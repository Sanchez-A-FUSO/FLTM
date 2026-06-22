"""
集中式多分类训练脚本，用于与联邦学习 FedAvg 做对比实验。

该脚本复用联邦版 CNN 模型结构与数据格式，直接将 10 个客户端的
`client_*.npz` 合并后在单机上进行集中式监督训练，并导出与联邦版
一致的产物格式，方便后续推理/评估对比。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf


NUM_CLIENTS = 10


def load_metadata(processed_dir: Path) -> dict:
  with open(processed_dir / "metadata.json", "r", encoding="utf-8") as f:
    return json.load(f)


def parse_hidden_units(s: str) -> tuple[int, ...]:
  parts = [p.strip() for p in s.split(",") if p.strip()]
  if not parts:
    raise SystemExit("--hidden 至少包含一层宽度，例如 256,128")
  out: list[int] = []
  for p in parts:
    u = int(p)
    if u <= 0:
      raise SystemExit(f"--hidden 每层宽度须为正，得到 {u!r}")
    out.append(u)
  return tuple(out)


def _infer_2d_shape(num_features: int) -> tuple[int, int]:
  if num_features <= 0:
    raise SystemExit(f"num_features 必须为正，得到 {num_features!r}")
  side = int(np.sqrt(num_features))
  if side * side != num_features:
    raise SystemExit(
        f"预处理后的图像特征数必须是完全平方数，当前为 {num_features}。"
    )
  return side, side


def build_model(
    num_features: int,
    num_classes: int,
    *,
    dropout: float = 0.0,
    hidden_units: tuple[int, ...] = (256, 128),
) -> tf.keras.Model:
  """与联邦 CNN 版保持一致的模型结构。"""
  h, w = _infer_2d_shape(num_features)
  inputs = tf.keras.layers.Input(shape=(h, w, 1))
  x = inputs

  x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  if dropout > 0:
    x = tf.keras.layers.Dropout(dropout)(x)

  x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  if dropout > 0:
    x = tf.keras.layers.Dropout(dropout)(x)

  x = tf.keras.layers.Flatten()(x)
  for u in hidden_units:
    x = tf.keras.layers.Dense(u, activation="relu")(x)
    if dropout > 0:
      x = tf.keras.layers.Dropout(dropout)(x)

  outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs, name="centralized_2d_cnn")


def make_optimizer(
    name: str,
    learning_rate: float,
    *,
    clipnorm: float = 0.0,
) -> tf.keras.optimizers.Optimizer:
  n = name.strip().lower()
  clip_kw: dict[str, float] = {}
  if clipnorm > 0:
    clip_kw["clipnorm"] = float(clipnorm)
  if n == "sgd":
    return tf.keras.optimizers.SGD(learning_rate=learning_rate, **clip_kw)
  if n == "adam":
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, **clip_kw)
  raise SystemExit(f"--client-opt 仅支持 sgd、adam，得到 {name!r}")


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
  def __init__(self, gamma: float = 2.0, name: str = "sparse_categorical_focal"):
    super().__init__(name=name)
    self.gamma = float(gamma)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    batch = tf.shape(y_true)[0]
    idx = tf.stack([tf.range(batch, dtype=tf.int32), y_true], axis=1)
    pt = tf.gather_nd(y_pred, idx)
    ce = -tf.math.log(pt)
    return tf.pow(1.0 - pt, self.gamma) * ce

  def get_config(self) -> dict:
    base = super().get_config()
    base.update({"gamma": self.gamma})
    return base


def make_supervised_loss(loss_name: str, *, focal_gamma: float) -> tf.keras.losses.Loss:
  n = loss_name.strip().lower()
  if n in ("ce", "crossentropy", "sparse_ce"):
    return tf.keras.losses.SparseCategoricalCrossentropy()
  if n == "focal":
    if focal_gamma < 0:
      raise SystemExit("--focal-gamma 须 >= 0。")
    return SparseCategoricalFocalLoss(gamma=focal_gamma)
  raise SystemExit(f"--loss 仅支持 crossentropy、focal，得到 {loss_name!r}")


def compute_class_weight_map(
    ys: list[np.ndarray],
    num_classes: int,
    power: float = 1.0,
) -> dict[int, float]:
  y_all = np.concatenate(ys, axis=0)
  counts = np.bincount(y_all.astype(np.int64), minlength=num_classes).astype(np.float64)
  total = float(np.sum(counts))
  safe = np.where(counts <= 0.0, 1.0, counts)
  weights = total / (float(num_classes) * safe)
  if power != 1.0:
    weights = np.power(weights, float(power))
  return {i: float(weights[i]) for i in range(num_classes)}


def main() -> None:
  parser = argparse.ArgumentParser(description="集中式多分类训练脚本（用于对比联邦学习）")
  parser.add_argument("--data-dir", type=str, default="data/processed")
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument(
      "--artifacts-dir",
      type=str,
      default="artifacts_centralized",
      help="训练结束后写入 global_model.keras、metadata.json、global_scaler.npz 副本。",
  )
  parser.add_argument(
      "--no-class-weight",
      action="store_true",
      help="关闭按类别频次反比加权（默认开启）。不均衡数据下关闭易导致几乎总预测 BENIGN。",
  )
  parser.add_argument(
      "--class-weight-power",
      type=float,
      default=1.0,
      help="balanced 类权的幂次：<1 缓和稀有类权重；默认 1.0。",
  )
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.0,
      help="隐藏层后 Dropout 比例，建议 0~0.3。",
  )
  parser.add_argument(
      "--hidden",
      type=str,
      default="256,128",
      help="逗号分隔的隐藏层宽度，例如 256,128 或 512,256,128。",
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="adam",
      choices=("sgd", "adam"),
      help="集中式训练优化器。",
  )
  parser.add_argument(
      "--loss",
      type=str,
      default="crossentropy",
      choices=("crossentropy", "focal"),
      help="crossentropy=稀疏交叉熵；focal=Focal Loss。",
  )
  parser.add_argument(
      "--focal-gamma",
      type=float,
      default=2.0,
      help="仅 --loss focal 时有效，典型 1~3。",
  )
  parser.add_argument(
      "--clipnorm",
      type=float,
      default=0.0,
      help="优化器梯度裁剪 L2 范数上限；>0 启用，例如 1.0。",
  )
  parser.add_argument(
      "--val-fraction",
      type=float,
      default=0.1,
      help="从合并后的训练集切出验证集比例，默认 0.1。",
  )
  args = parser.parse_args()

  processed_dir = Path(args.data_dir)
  meta = load_metadata(processed_dir)
  if int(meta["n_clients"]) != NUM_CLIENTS:
    raise SystemExit(f"本脚本固定 {NUM_CLIENTS} 个客户端，但 metadata 中为 {meta['n_clients']}")

  xs: list[np.ndarray] = []
  ys: list[np.ndarray] = []
  for i in range(NUM_CLIENTS):
    path = processed_dir / f"client_{i}.npz"
    z = np.load(path)
    xs.append(z["x"])
    ys.append(z["y"])

  num_features = int(meta["n_features"])
  num_classes = int(meta["n_classes"])
  hidden_units = parse_hidden_units(args.hidden)

  if args.dropout < 0 or args.dropout >= 1:
    raise SystemExit("--dropout 须在 [0, 1) 内。")
  if args.clipnorm < 0:
    raise SystemExit("--clipnorm 须 >= 0。")
  if not (0.0 <= args.val_fraction < 1.0):
    raise SystemExit("--val-fraction 须在 [0, 1) 内。")
  if args.no_class_weight and args.class_weight_power != 1.0:
    print("[警告] 已关闭 class_weight，--class-weight-power 将被忽略。")
  if args.loss == "focal":
    print(f"[训练] 使用 Focal Loss（gamma={args.focal_gamma:g}）。")

  x_all = np.concatenate(xs, axis=0)
  y_all = np.concatenate(ys, axis=0)

  rng = np.random.default_rng(42)
  indices = np.arange(len(x_all))
  rng.shuffle(indices)
  x_all = x_all[indices]
  y_all = y_all[indices]

  split = int(len(x_all) * (1.0 - args.val_fraction))
  if split <= 0 or split >= len(x_all):
    raise SystemExit("--val-fraction 设置后训练/验证划分无效，请调整参数。")

  x_train, x_val = x_all[:split], x_all[split:]
  y_train, y_val = y_all[:split], y_all[split:]

  class_weight: dict[int, float] | None = None
  if not args.no_class_weight:
    if args.class_weight_power <= 0:
      raise SystemExit("--class-weight-power 须为正数。")
    class_weight = compute_class_weight_map(ys, num_classes, power=args.class_weight_power)
    pw = args.class_weight_power
    pw_note = "balanced" if pw == 1.0 else f"balanced^{pw:g}"
    print(f"[训练] 已启用 class_weight（{pw_note}）。")

  model = build_model(num_features, num_classes, dropout=args.dropout, hidden_units=hidden_units)
  model.compile(
      optimizer=make_optimizer(args.optimizer, args.learning_rate, clipnorm=args.clipnorm),
      loss=make_supervised_loss(args.loss, focal_gamma=args.focal_gamma),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
  )

  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          monitor="val_accuracy",
          patience=5,
          restore_best_weights=True,
          mode="max",
      )
  ]

  history = model.fit(
      x_train,
      y_train,
      validation_data=(x_val, y_val),
      epochs=args.epochs,
      batch_size=args.batch_size,
      shuffle=True,
      verbose=1,
      class_weight=class_weight,
      callbacks=callbacks,
  )

  best_val = max(history.history.get("val_accuracy", [float("nan")]))
  print(f"最佳验证准确率: {best_val:.4f}")

  eval_loss, eval_acc = model.evaluate(x_all, y_all, verbose=0, return_dict=False)
  print(f"全局拼接 evaluate: loss={eval_loss:.4f} accuracy={eval_acc:.4f}")

  art = Path(args.artifacts_dir)
  art.mkdir(parents=True, exist_ok=True)
  model.save(art / "global_model.keras")
  shutil.copy2(processed_dir / "metadata.json", art / "metadata.json")
  scaler_src = processed_dir / "global_scaler.npz"
  if scaler_src.is_file():
    shutil.copy2(scaler_src, art / "global_scaler.npz")
  print(f"已导出部署包: {art}（模型 + metadata + scaler）")


if __name__ == "__main__":
  main()
