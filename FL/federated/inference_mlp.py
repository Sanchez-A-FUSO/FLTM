"""
加载联邦聚合后的全局 MLP 模型，对流量特征做多分类预测，并给出「异常」二值判定（非 BENIGN 即异常）。

该脚本用于与 CNN 版本做对比实验；默认读取 `artifacts_mlp` 下的模型与元数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, recall_score


def _benign_index(label_classes: list[str]) -> int | None:
  for i, name in enumerate(label_classes):
    if str(name).strip().upper() == "BENIGN":
      return i
  return None


@dataclass
class PredictionRow:
  predicted_class: str
  predicted_id: int
  probabilities: np.ndarray
  is_anomaly: bool
  anomaly_score: float
  """1 - P(BENIGN)；若无 BENIGN 类则退化为基于最大概率的简化异常分数。"""


class FederatedFlowDetectorMLP:
  def __init__(
      self,
      artifacts_dir: str | Path,
      processed_dir: str | Path | None = None,
  ) -> None:
    art = Path(artifacts_dir)
    proc = Path(processed_dir) if processed_dir is not None else Path("data/processed")

    model_path = art / "global_model.keras"
    if not model_path.is_file():
      raise FileNotFoundError(
          f"缺少全局模型 {model_path}。\n"
          "请先训练 MLP 版本模型，确保 artifacts 目录下存在 global_model.keras。"
      )

    meta_path = art / "metadata.json"
    if not meta_path.is_file():
      meta_path = proc / "metadata.json"
    if not meta_path.is_file():
      raise FileNotFoundError(
          "找不到 metadata.json。请先运行预处理脚本或完成一次 MLP 联邦训练。"
      )

    scaler_path = art / "global_scaler.npz"
    if not scaler_path.is_file():
      scaler_path = proc / "global_scaler.npz"
    if not scaler_path.is_file():
      raise FileNotFoundError(
          "找不到 global_scaler.npz。请先运行预处理脚本或完成一次 MLP 联邦训练。"
      )

    with open(meta_path, "r", encoding="utf-8") as f:
      self.metadata: dict[str, Any] = json.load(f)

    z = np.load(scaler_path)
    self._mean = z["mean"].astype(np.float64)
    self._scale = z["scale"].astype(np.float64)
    self._scale_safe = np.where(self._scale < 1e-12, 1.0, self._scale)

    self.label_classes: list[str] = list(self.metadata["label_classes"])
    self.n_features = int(self.metadata["n_features"])
    self._benign_i = _benign_index(self.label_classes)

    self.model = tf.keras.models.load_model(model_path, compile=False)

  def transform(self, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
      x = x[np.newaxis, :]
    if x.ndim != 2:
      raise ValueError(f"MLP 仅支持二维特征矩阵 (n_samples, n_features)，当前 shape={x.shape}")
    if x.shape[-1] != self.n_features:
      raise ValueError(f"期望特征维 {self.n_features}，得到 {x.shape[-1]}")
    x64 = x.astype(np.float64)
    norm = (x64 - self._mean) / self._scale_safe
    return norm.astype(np.float32)

  def predict_proba(self, x_raw: np.ndarray) -> np.ndarray:
    x = self.transform(x_raw)
    return self.model.predict(x, verbose=0)

  def predict_rows(self, x_raw: np.ndarray, anomaly_threshold: float = 0.5) -> list[PredictionRow]:
    prob = self.predict_proba(x_raw)
    if x_raw.ndim == 1:
      prob = prob[0:1]
    rows: list[PredictionRow] = []
    for p in prob:
      pid = int(np.argmax(p))
      name = self.label_classes[pid] if pid < len(self.label_classes) else str(pid)
      if self._benign_i is not None:
        p_benign = float(p[self._benign_i])
        score = float(np.clip(1.0 - p_benign, 0.0, 1.0))
        is_anom = score >= anomaly_threshold
      else:
        is_anom = True
        score = float(np.max(p))
      rows.append(
          PredictionRow(
              predicted_class=name,
              predicted_id=pid,
              probabilities=p.astype(np.float64),
              is_anomaly=is_anom,
              anomaly_score=score,
          )
      )
    return rows


def _short_label(s: str, width: int = 26) -> str:
  t = str(s).strip()
  return t if len(t) <= width else t[: width - 2] + ".."


def print_rigorous_eval(
    y_true: np.ndarray,
    pred_multiclass: np.ndarray,
    pred_attack: np.ndarray,
    label_classes: list[str],
    benign_i: int | None,
    dataset_name: str,
    show_confusion: bool,
) -> None:
  n = len(y_true)
  labels_idx = np.arange(len(label_classes))

  print(f"\n========== 评估：{dataset_name} | 样本数 n={n} ==========")
  acc = float(np.mean(pred_multiclass == y_true))
  print(f"多分类准确率 (Accuracy): {acc:.4f}")

  if benign_i is not None:
    y_attack = (y_true != benign_i).astype(np.int32)
    p_attack = pred_attack.astype(np.int32)
    tn = int(np.sum((p_attack == 0) & (y_attack == 0)))
    fp = int(np.sum((p_attack == 1) & (y_attack == 0)))
    fn = int(np.sum((p_attack == 0) & (y_attack == 1)))
    tp = int(np.sum((p_attack == 1) & (y_attack == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print("\n--- 异常检测（正类 = 非 BENIGN，即攻击/异常流量）---")
    print("混淆（行=真实，列=预测）：")
    print(f"                 预测正常(BENIGN)  预测异常")
    print(f"  真实正常(BENIGN)      TN={tn:6d} ←判对    FP={fp:6d} ←误报(将正常判为异常)")
    print(f"  真实异常              FN={fn:6d} ←漏报(将异常判为正常)    TP={tp:6d} ←命中(将攻击判为异常)")
    print(f"  精确率 Precision = TP/(TP+FP) = {prec:.4f}")
    print(f"  召回率 Recall    = TP/(TP+FN) = {rec:.4f}   （检出真实异常的比例，越高漏报越少）")
    print(f"  F1               = {f1:.4f}")
    print(f"  误报率 FPR       = FP/(FP+TN) = {fpr:.4f}   （正常流量被误判为异常的比例）")
    n_attack = int(y_attack.sum())
    n_benign = n - n_attack
    print(f"  真实分布: 正常 {n_benign} | 异常 {n_attack}")

  names = [_short_label(c) for c in label_classes]
  print("\n--- 多分类：按类别精确率 / 召回率 / F1（support=该类真实样本数）---")
  print(
      classification_report(
          y_true,
          pred_multiclass,
          labels=labels_idx,
          target_names=names,
          digits=4,
          zero_division=0,
      )
  )

  r_macro = recall_score(y_true, pred_multiclass, average="macro", zero_division=0, labels=labels_idx)
  r_weighted = recall_score(y_true, pred_multiclass, average="weighted", zero_division=0, labels=labels_idx)
  r_micro = recall_score(y_true, pred_multiclass, average="micro", zero_division=0, labels=labels_idx)
  print(f"召回率汇总 — micro（与准确率一致）: {r_micro:.4f}")
  print(f"召回率汇总 — macro（各类平等）:     {r_macro:.4f}")
  print(f"召回率汇总 — weighted（按 support 加权）: {r_weighted:.4f}")

  if show_confusion:
    cm = confusion_matrix(y_true, pred_multiclass, labels=labels_idx)
    print("\n--- 混淆矩阵（行=真实类别，列=预测类别；单元格为样本数）---")
    header = "".join(f"{names[j]:>12}" for j in range(len(names)))
    print(f"{'':>14}{header}")
    for i, row_name in enumerate(names):
      row = "".join(f"{cm[i, j]:12d}" for j in range(len(names)))
      print(f"{row_name:>14}{row}")


def main() -> None:
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

  parser = argparse.ArgumentParser(description="联邦 MLP 全局模型推理 / 异常流量判定")
  parser.add_argument("--artifacts-dir", type=str, default="artifacts_mlp")
  parser.add_argument(
      "--processed-dir",
      type=str,
      default="data/processed",
      help="当 artifacts 内没有 metadata.json / global_scaler.npz 时，从此目录回退加载（需先 prepare）。",
  )
  parser.add_argument(
      "--eval-npz",
      type=str,
      default="",
      help="含 x,y 的 NPZ（如 data/processed/client_0.npz 或 test.npz），输出召回率等指标。",
  )
  parser.add_argument("--eval-limit", type=int, default=5000, help="评估最多使用的样本数")
  parser.add_argument(
      "--brief",
      action="store_true",
      help="仅打印一行准确率 + 异常 F1（旧版简短输出）。",
  )
  parser.add_argument(
      "--confusion",
      action="store_true",
      help="额外打印完整多分类混淆矩阵（类别多时较宽）。",
  )
  parser.add_argument(
      "--anomaly-threshold",
      type=float,
      default=0.5,
      help="异常判定阈值：score=1-P(BENIGN)，当 score >= 阈值 时判为异常（默认 0.5）。",
  )
  parser.add_argument(
      "--eval-diagnostics",
      action="store_true",
      help="打印 P(BENIGN) / argmax 在「真实正常 vs 真实攻击」上的分布，用于判断是否塌缩为全 BENIGN（与 metadata 是否一致无关）。",
  )
  args = parser.parse_args()

  det = FederatedFlowDetectorMLP(args.artifacts_dir, processed_dir=args.processed_dir)
  anomaly_threshold = float(np.clip(args.anomaly_threshold, 0.0, 1.0))

  if not args.eval_npz:
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((2, det.n_features)).astype(np.float32)
    for i, row in enumerate(det.predict_rows(dummy, anomaly_threshold=anomaly_threshold)):
      print(f"样本{i}: class={row.predicted_class} anomaly={row.is_anomaly} score={row.anomaly_score:.4f}")
    print("未指定 --eval-npz，仅作 smoke test。可对 client NPZ 做评估。")
    return

  path = Path(args.eval_npz)
  z = np.load(path)
  x = np.asarray(z["x"])
  y = np.asarray(z["y"])
  if x.ndim != 2:
    raise ValueError(f"MLP 评估 NPZ 中 x 应为二维 (n_samples, n_features)，当前 shape={x.shape}")
  y = y.astype(np.int64, copy=False).reshape(-1)
  if x.shape[0] != y.shape[0]:
    raise ValueError(f"x、y 样本数不一致: x.shape[0]={x.shape[0]} len(y)={len(y)}")
  n_classes_meta = len(det.label_classes)
  if y.size and (y.min() < 0 or y.max() >= n_classes_meta):
    raise ValueError(
        f"标签 id 超出 metadata 类别数 [0,{n_classes_meta - 1}]：min={int(y.min())} max={int(y.max())}"
    )

  n = min(x.shape[0], args.eval_limit)
  x = x[:n]
  y = y[:n]

  prob = det.predict_proba(x)
  if prob.shape[1] != n_classes_meta:
    raise ValueError(
        f"模型输出维 {prob.shape[1]} 与 metadata 类别数 {n_classes_meta} 不一致，"
        "请确认 artifacts 与 data/processed 来自同一次 prepare / 训练。"
    )
  pred_multiclass = np.argmax(prob, axis=1).astype(np.int64, copy=False)
  benign_i = det._benign_i

  if benign_i is not None:
    anomaly_score = np.clip(1.0 - prob[:, benign_i], 0.0, 1.0)
    pred_attack = (anomaly_score >= anomaly_threshold).astype(np.int32)
  else:
    anomaly_score = np.max(prob, axis=1)
    pred_attack = np.ones(len(prob), dtype=np.int32)

  if args.eval_diagnostics and benign_i is not None:
    pb = prob[:, benign_i].astype(np.float64)
    am = pred_multiclass
    mask_b = y == benign_i
    mask_a = ~mask_b
    nb = int(mask_b.sum())
    na = int(mask_a.sum())
    print("\n--- 诊断：模型是否把「攻击」也当成 BENIGN（与文件是否对齐无关）---")
    print(f"评估子集 n={n} | 模型 softmax 维={prob.shape[1]} | benign 列索引={benign_i}")
    if nb > 0:
      print(
          f"真实正常: n={nb}  mean/median P(BENIGN)={pb[mask_b].mean():.4f}/{float(np.median(pb[mask_b])):.4f}  "
          f"argmax 为 BENIGN 的比例={float(np.mean(am[mask_b] == benign_i)):.4f}"
      )
    if na > 0:
      print(
          f"真实攻击: n={na}  mean/median P(BENIGN)={pb[mask_a].mean():.4f}/{float(np.median(pb[mask_a])):.4f}  "
          f"argmax 为 BENIGN 的比例={float(np.mean(am[mask_a] == benign_i)):.4f}"
      )
    print(
        "若「真实攻击」上 argmax 为 BENIGN 比例≈1 且 mean P(BENIGN) 很高，说明训练未分开攻击，"
        "需加强类权/轮次/网络或检查训练日志是否启用 class_weight。"
    )

  if args.brief:
    acc = float(np.mean(pred_multiclass == y))
    if benign_i is not None:
      y_bin = (y != benign_i).astype(np.int32)
      tp = int(np.sum((pred_attack == 1) & (y_bin == 1)))
      fp = int(np.sum((pred_attack == 1) & (y_bin == 0)))
      fn = int(np.sum((pred_attack == 0) & (y_bin == 1)))
      prec = tp / (tp + fp) if (tp + fp) else 0.0
      rec = tp / (tp + fn) if (tp + fn) else 0.0
      f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
      print(
          f"评估 {path.name} n={n}  accuracy={acc:.4f}  异常阈值={anomaly_threshold:.2f}  "
          f"异常 F1={f1:.4f} (P={prec:.4f} R={rec:.4f})"
      )
    else:
      print(f"评估 {path.name} n={n}  accuracy={acc:.4f}（无 BENIGN，无二分类指标）")
  else:
    print(f"异常判定阈值: {anomaly_threshold:.2f}（score=1-P(BENIGN)）")
    print_rigorous_eval(
        y,
        pred_multiclass,
        pred_attack,
        det.label_classes,
        benign_i,
        dataset_name=path.name,
        show_confusion=args.confusion,
    )


if __name__ == "__main__":
  main()
