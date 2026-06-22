"""
多分类联邦训练（FedAvg），客户端数固定为 10，数据为 IID（prepare 脚本中分层划分）。

默认使用 Keras 逐客户端训练 + 按样本数加权平均权重模拟 FedAvg，
并可通过环境变量 `FL_USE_TFF=1` 切换到 TensorFlow Federated。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
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
  """根据特征数推断正方形图像边长。"""
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
  """2D CNN：直接接收预处理好的灰度图输入。"""
  h, w = _infer_2d_shape(num_features)
  inputs = tf.keras.layers.Input(shape=(h, w, 1))
  x = inputs

  # 第一段卷积：提取局部二维模式
  x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  if dropout > 0:
    x = tf.keras.layers.Dropout(dropout)(x)

  # 第二段卷积：进一步抽取高层模式
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
  return tf.keras.Model(inputs=inputs, outputs=outputs, name="federated_2d_cnn")


def make_client_optimizer(
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
  """多分类稀疏标签 Focal：压低「已分对且置信度高」样本的梯度，让模型更关注难分样本与长尾类。"""

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


def weighted_average_weights(
    weight_sets: list[list[np.ndarray]],
    sample_sizes: list[int],
) -> list[np.ndarray]:
  total = float(sum(sample_sizes))
  coeffs = [s / total for s in sample_sizes]
  n_layers = len(weight_sets[0])
  out: list[np.ndarray] = []
  ref_dtype = weight_sets[0][0].dtype
  for li in range(n_layers):
    acc = np.zeros_like(weight_sets[0][li], dtype=np.float64)
    for c, wset in zip(coeffs, weight_sets):
      acc += c * wset[li].astype(np.float64, copy=False)
    out.append(acc.astype(ref_dtype, copy=False))
  return out


def blend_weights(
    global_w: list[np.ndarray],
    target_w: list[np.ndarray],
    server_lr: float,
) -> list[np.ndarray]:
  if server_lr >= 0.999999:
    return target_w
  out = []
  for g, t in zip(global_w, target_w):
    out.append(((1.0 - server_lr) * g.astype(np.float64) + server_lr * t.astype(np.float64)).astype(g.dtype))
  return out


def _read_loss_and_accuracy(metrics) -> tuple[float, float]:
  """TFF 聚合后的 metrics 结构可能嵌套或为 Struct，这里做宽松读取。"""
  try:
    from tensorflow_federated.python.common_libs import structure as struct_lib

    if isinstance(metrics, struct_lib.Struct):
      metrics = collections.OrderedDict(struct_lib.iter_elements(metrics))
  except Exception:
    pass

  if hasattr(metrics, "_asdict"):
    metrics = metrics._asdict()

  loss_v: float | None = None
  acc_v: float | None = None

  def _scalar(x):
    try:
      if hasattr(x, "numpy"):
        return float(x.numpy())
      return float(x)
    except (TypeError, ValueError):
      return None

  def walk(obj, depth: int = 0) -> None:
    nonlocal loss_v, acc_v
    if depth > 8 or (loss_v is not None and acc_v is not None):
      return
    if isinstance(obj, dict):
      for k, v in obj.items():
        lk = str(k).lower()
        if loss_v is None and "loss" in lk:
          s = _scalar(v)
          if s is not None:
            loss_v = s
        if acc_v is None and ("accuracy" in lk or "sparse_categorical_accuracy" in lk):
          s = _scalar(v)
          if s is not None:
            acc_v = s
        if isinstance(v, (dict, list, tuple)) or hasattr(v, "_asdict"):
          walk(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
      for it in obj:
        walk(it, depth + 1)
    elif hasattr(obj, "_asdict"):
      walk(obj._asdict(), depth + 1)

  walk(metrics)
  return (loss_v if loss_v is not None else float("nan"), acc_v if acc_v is not None else float("nan"))


def compute_class_weight_map(
    ys: list[np.ndarray],
    num_classes: int,
    power: float = 1.0,
) -> dict[int, float]:
  """与 sklearn `balanced` 一致：n_samples / (n_classes * count_k)。

  `power`<1 时对权重做逐元幂次（如 0.5≈平方根），拉平极端比值，减轻「过度压低 P(BENIGN)」导致的误报。
  """
  y_all = np.concatenate(ys, axis=0)
  counts = np.bincount(y_all.astype(np.int64), minlength=num_classes).astype(np.float64)
  total = float(np.sum(counts))
  safe = np.where(counts <= 0.0, 1.0, counts)
  weights = total / (float(num_classes) * safe)
  if power != 1.0:
    weights = np.power(weights, float(power))
  return {i: float(weights[i]) for i in range(num_classes)}


def train_fedavg_keras(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    num_features: int,
    num_classes: int,
    rounds: int,
    batch_size: int,
    client_epochs: int,
    client_lr: float,
    server_lr: float,
    class_weight: dict[int, float] | None,
    dropout: float,
    hidden_units: tuple[int, ...],
    client_opt: str,
    supervised_loss: tf.keras.losses.Loss,
    round_lr_decay: float,
    clipnorm: float,
) -> tf.keras.Model:
  global_model = build_model(
      num_features, num_classes, dropout=dropout, hidden_units=hidden_units
  )
  global_model.compile(
      loss=supervised_loss,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
  )

  local = build_model(num_features, num_classes, dropout=dropout, hidden_units=hidden_units)

  x_all = np.concatenate(xs, axis=0)
  y_all = np.concatenate(ys, axis=0)
  sizes = [len(x) for x in xs]

  for r in range(rounds):
    eff_lr = float(client_lr) * (float(round_lr_decay) ** r)
    local.compile(
        optimizer=make_client_optimizer(client_opt, eff_lr, clipnorm=clipnorm),
        loss=supervised_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
    )
    client_weights: list[list[np.ndarray]] = []
    for i in range(NUM_CLIENTS):
      local.set_weights(global_model.get_weights())
      local.fit(
          xs[i],
          ys[i],
          batch_size=batch_size,
          epochs=client_epochs,
          shuffle=True,
          verbose=0,
          class_weight=class_weight,
      )
      client_weights.append(local.get_weights())

    w_avg = weighted_average_weights(client_weights, sizes)
    new_w = blend_weights(global_model.get_weights(), w_avg, server_lr)
    global_model.set_weights(new_w)

    loss, acc = global_model.evaluate(x_all, y_all, verbose=0, return_dict=False)
    print(f"round {r + 1:02d}/{rounds}  loss={loss:.4f}  sparse_categorical_accuracy={acc:.4f}")

  return global_model


def train_fedavg_tff(
    client_ds: list[tf.data.Dataset],
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    num_features: int,
    num_classes: int,
    rounds: int,
    batch_size: int,
    client_epochs: int,
    client_lr: float,
    server_lr: float,
    dropout: float,
    hidden_units: tuple[int, ...],
    client_opt: str,
    supervised_loss: tf.keras.losses.Loss,
    clipnorm: float,
) -> tf.keras.Model:
  import tensorflow_federated as tff

  example_ds = tf.data.Dataset.from_tensor_slices((xs[0], ys[0]))
  example_ds = example_ds.shuffle(1000).batch(batch_size).repeat(client_epochs)
  input_spec = example_ds.element_spec

  def model_fn():
    keras_model = build_model(
        num_features, num_classes, dropout=dropout, hidden_units=hidden_units
    )
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=supervised_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

  trainer = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=lambda: make_client_optimizer(
          client_opt, client_lr, clipnorm=clipnorm
      ),
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=server_lr),
  )

  state = trainer.initialize()
  for r in range(rounds):
    out = trainer.next(state, client_ds)
    state = out.state
    loss, train_acc = _read_loss_and_accuracy(out.metrics)
    print(f"round {r + 1:02d}/{rounds}  loss={loss:.4f}  sparse_categorical_accuracy={train_acc:.4f}")

  keras_model = build_model(num_features, num_classes, dropout=dropout, hidden_units=hidden_units)
  mw = trainer.get_model_weights(state)
  if hasattr(mw, "assign_weights_to"):
    mw.assign_weights_to(keras_model)
  else:
    tff.learning.ModelWeights.from_tff_result(mw).assign_weights_to(keras_model)
  return keras_model


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", type=str, default="data/processed")
  parser.add_argument("--rounds", type=int, default=15)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--client-epochs", type=int, default=1)
  parser.add_argument("--client-lr", type=float, default=0.02)
  parser.add_argument("--server-lr", type=float, default=1.0)
  parser.add_argument(
      "--artifacts-dir",
      type=str,
      default="artifacts",
      help="训练结束后写入 global_model.keras、metadata.json、global_scaler.npz 副本，供推理部署。",
  )
  parser.add_argument(
      "--no-class-weight",
      action="store_true",
      help="关闭按类别频次反比加权（默认开启）。不均衡数据下关闭易导致几乎总预测 BENIGN、异常召回为 0。",
  )
  parser.add_argument(
      "--class-weight-power",
      type=float,
      default=1.0,
      help="balanced 类权的幂次：<1 缓和稀有类权重（减轻误报、常能抬升整体准确率），例如 0.5~0.75；默认 1.0。",
  )
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.0,
      help="隐藏层后 Dropout 比例，缓解过拟合；联邦场景建议 0~0.3，例如 0.2。默认 0（与旧行为一致）。",
  )
  parser.add_argument(
      "--hidden",
      type=str,
      default="256,128",
      help="逗号分隔的隐藏层宽度，例如 256,128 或 512,256,128。",
  )
  parser.add_argument(
      "--client-opt",
      type=str,
      default="sgd",
      choices=("sgd", "adam"),
      help="客户端优化器：adam 时常需更小 --client-lr（如 1e-3）。",
  )
  parser.add_argument(
      "--loss",
      type=str,
      default="crossentropy",
      choices=("crossentropy", "focal"),
      help="crossentropy=稀疏交叉熵；focal=Focal Loss（难分样本权重更大，常与 class_weight 组合试）。",
  )
  parser.add_argument(
      "--focal-gamma",
      type=float,
      default=2.0,
      help="仅 --loss focal 时有效，典型 1~3；越大越强调难分样本。",
  )
  parser.add_argument(
      "--round-lr-decay",
      type=float,
      default=1.0,
      help="每轮客户端学习率乘子：第 r 轮 lr = client_lr * (round-lr-decay)^r；"
      "典型 0.98~1.0，后期更小步长利于收敛（仅 Keras FedAvg 路径生效）。",
  )
  parser.add_argument(
      "--clipnorm",
      type=float,
      default=0.0,
      help="客户端优化器全局梯度裁剪 L2 范数上限；>0 启用，例如 1.0，可抑制爆梯与聚合不稳定。",
  )
  args = parser.parse_args()

  use_tff = os.environ.get("FL_USE_TFF", "").strip() in ("1", "true", "yes")

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
  if args.round_lr_decay <= 0:
    raise SystemExit("--round-lr-decay 须为正。")
  if args.clipnorm < 0:
    raise SystemExit("--clipnorm 须 >= 0。")
  supervised_loss = make_supervised_loss(args.loss, focal_gamma=args.focal_gamma)
  if args.loss == "focal":
    print(f"[训练] 使用 Focal Loss（gamma={args.focal_gamma:g}）。")

  class_weight: dict[int, float] | None = None
  if not args.no_class_weight:
    if args.class_weight_power <= 0:
      raise SystemExit("--class-weight-power 须为正数。")
    class_weight = compute_class_weight_map(ys, num_classes, power=args.class_weight_power)
    pw = args.class_weight_power
    pw_note = "balanced" if pw == 1.0 else f"balanced^{pw:g}"
    print(f"[训练] 已启用 class_weight（{pw_note}）；若需纯交叉熵请加 --no-class-weight。")
    if use_tff:
      print(
          "[警告] FL_USE_TFF=1 时当前实现未把 class_weight 传入 TFF；"
          "不均衡场景请改用默认 Keras FedAvg，否则易「全判 BENIGN」。"
      )

  if use_tff and args.round_lr_decay != 1.0:
    print("[警告] --round-lr-decay 仅在 Keras FedAvg 路径生效，TFF 下客户端 lr 仍为固定 --client-lr。")

  if use_tff:
    client_ds = []
    for i in range(NUM_CLIENTS):
      ds = tf.data.Dataset.from_tensor_slices((xs[i], ys[i]))
      ds = ds.shuffle(min(10000, len(xs[i]))).batch(args.batch_size).repeat(args.client_epochs)
      client_ds.append(ds)
    keras_model = train_fedavg_tff(
        client_ds,
        xs,
        ys,
        num_features,
        num_classes,
        args.rounds,
        args.batch_size,
        args.client_epochs,
        args.client_lr,
        args.server_lr,
        args.dropout,
        hidden_units,
        args.client_opt,
        supervised_loss,
        args.clipnorm,
    )
  else:
    keras_model = train_fedavg_keras(
        xs,
        ys,
        num_features,
        num_classes,
        args.rounds,
        args.batch_size,
        args.client_epochs,
        args.client_lr,
        args.server_lr,
        class_weight,
        args.dropout,
        hidden_units,
        args.client_opt,
        supervised_loss,
        args.round_lr_decay,
        args.clipnorm,
    )

  x_all = np.concatenate(xs, axis=0)
  y_all = np.concatenate(ys, axis=0)
  keras_model.compile(
      loss=supervised_loss,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
  )
  ev = keras_model.evaluate(x_all, y_all, verbose=0, return_dict=False)
  print(f"全局拼接 evaluate: loss={ev[0]:.4f} accuracy={ev[1]:.4f}")

  art = Path(args.artifacts_dir)
  art.mkdir(parents=True, exist_ok=True)
  # Keras 3 原生 .keras 格式不支持 include_optimizer；推理仅需权重，加载时忽略优化器状态即可。
  keras_model.save(art / "global_model.keras")
  shutil.copy2(processed_dir / "metadata.json", art / "metadata.json")
  scaler_src = processed_dir / "global_scaler.npz"
  if scaler_src.is_file():
    shutil.copy2(scaler_src, art / "global_scaler.npz")
  print(f"已导出部署包: {art}（模型 + metadata + scaler）")


if __name__ == "__main__":
  main()
