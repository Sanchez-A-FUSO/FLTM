"""Predict classes for a real-world flow-feature CSV.

This script is intended for datasets produced by
`scripts/pcap_to_simple_flow_csv.py` or any other CICIDS2017-style CSV.

It loads the trained model, metadata, and scaler, aligns CSV columns to the
training feature schema, applies the saved normalization, and prints only the
predicted class for each row.

Unlike the evaluation scripts, it does not require ground-truth labels and
it does not compute accuracy / recall / confusion-matrix statistics.

This version intentionally avoids pandas and uses only the Python standard
library for CSV parsing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


ALIASES = {
    "Destination Port": ["Destination Port", "Dst Port"],
    "Flow Duration": ["Flow Duration"],
    "Total Fwd Packets": ["Total Fwd Packets", "Tot Fwd Pkts"],
    "Total Backward Packets": ["Total Backward Packets", "Tot Bwd Pkts"],
    "Total Length of Fwd Packets": ["Total Length of Fwd Packets", "TotLen Fwd Pkts"],
    "Total Length of Bwd Packets": ["Total Length of Bwd Packets", "TotLen Bwd Pkts"],
    "Fwd Packet Length Max": ["Fwd Packet Length Max"],
    "Fwd Packet Length Min": ["Fwd Packet Length Min"],
    "Fwd Packet Length Mean": ["Fwd Packet Length Mean"],
    "Fwd Packet Length Std": ["Fwd Packet Length Std"],
    "Bwd Packet Length Max": ["Bwd Packet Length Max"],
    "Bwd Packet Length Min": ["Bwd Packet Length Min"],
    "Bwd Packet Length Mean": ["Bwd Packet Length Mean"],
    "Bwd Packet Length Std": ["Bwd Packet Length Std"],
    "Flow Bytes/s": ["Flow Bytes/s", "Flow Byts/s"],
    "Flow Packets/s": ["Flow Packets/s", "Flow Pkts/s"],
    "Flow IAT Mean": ["Flow IAT Mean"],
    "Flow IAT Std": ["Flow IAT Std"],
    "Flow IAT Max": ["Flow IAT Max"],
    "Flow IAT Min": ["Flow IAT Min"],
    "Fwd IAT Total": ["Fwd IAT Total"],
    "Fwd IAT Mean": ["Fwd IAT Mean"],
    "Fwd IAT Std": ["Fwd IAT Std"],
    "Fwd IAT Max": ["Fwd IAT Max"],
    "Fwd IAT Min": ["Fwd IAT Min"],
    "Bwd IAT Total": ["Bwd IAT Total"],
    "Bwd IAT Mean": ["Bwd IAT Mean"],
    "Bwd IAT Std": ["Bwd IAT Std"],
    "Bwd IAT Max": ["Bwd IAT Max"],
    "Bwd IAT Min": ["Bwd IAT Min"],
    "Fwd PSH Flags": ["Fwd PSH Flags"],
    "Bwd PSH Flags": ["Bwd PSH Flags"],
    "Fwd URG Flags": ["Fwd URG Flags"],
    "Bwd URG Flags": ["Bwd URG Flags"],
    "Fwd Header Length": ["Fwd Header Length"],
    "Bwd Header Length": ["Bwd Header Length"],
    "Fwd Packets/s": ["Fwd Packets/s"],
    "Bwd Packets/s": ["Bwd Packets/s"],
    "Min Packet Length": ["Min Packet Length"],
    "Max Packet Length": ["Max Packet Length"],
    "Packet Length Mean": ["Packet Length Mean"],
    "Packet Length Std": ["Packet Length Std"],
    "Packet Length Variance": ["Packet Length Variance"],
    "FIN Flag Count": ["FIN Flag Count"],
    "SYN Flag Count": ["SYN Flag Count"],
    "RST Flag Count": ["RST Flag Count"],
    "PSH Flag Count": ["PSH Flag Count"],
    "ACK Flag Count": ["ACK Flag Count"],
    "URG Flag Count": ["URG Flag Count"],
    "CWE Flag Count": ["CWE Flag Count"],
    "ECE Flag Count": ["ECE Flag Count"],
    "Down/Up Ratio": ["Down/Up Ratio"],
    "Average Packet Size": ["Average Packet Size"],
    "Avg Fwd Segment Size": ["Avg Fwd Segment Size"],
    "Avg Bwd Segment Size": ["Avg Bwd Segment Size"],
    "Fwd Avg Bytes/Bulk": ["Fwd Avg Bytes/Bulk"],
    "Fwd Avg Packets/Bulk": ["Fwd Avg Packets/Bulk"],
    "Fwd Avg Bulk Rate": ["Fwd Avg Bulk Rate"],
    "Bwd Avg Bytes/Bulk": ["Bwd Avg Bytes/Bulk"],
    "Bwd Avg Packets/Bulk": ["Bwd Avg Packets/Bulk"],
    "Bwd Avg Bulk Rate": ["Bwd Avg Bulk Rate"],
    "Subflow Fwd Packets": ["Subflow Fwd Packets"],
    "Subflow Fwd Bytes": ["Subflow Fwd Bytes"],
    "Subflow Bwd Packets": ["Subflow Bwd Packets"],
    "Subflow Bwd Bytes": ["Subflow Bwd Bytes"],
    "Init_Win_bytes_forward": ["Init_Win_bytes_forward"],
    "Init_Win_bytes_backward": ["Init_Win_bytes_backward"],
    "act_data_pkt_fwd": ["act_data_pkt_fwd"],
    "min_seg_size_forward": ["min_seg_size_forward"],
    "Active Mean": ["Active Mean"],
    "Active Std": ["Active Std"],
    "Active Max": ["Active Max"],
    "Active Min": ["Active Min"],
    "Idle Mean": ["Idle Mean"],
    "Idle Std": ["Idle Std"],
    "Idle Max": ["Idle Max"],
    "Idle Min": ["Idle Min"],
}


def _load_json(path: Path) -> dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def _load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
  z = np.load(path)
  mean = z["mean"].astype(np.float64)
  scale = z["scale"].astype(np.float64)
  scale = np.where(scale < 1e-12, 1.0, scale)
  return mean, scale


def _safe_float(value: str | None) -> float:
  if value is None:
    return 0.0
  s = str(value).strip()
  if s == "":
    return 0.0
  try:
    x = float(s)
    if np.isfinite(x):
      return x
  except Exception:
    pass
  return 0.0


def _resolve_label_name(label_id: int, label_classes: list[str]) -> str:
  if 0 <= label_id < len(label_classes):
    return str(label_classes[label_id])
  return str(label_id)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  with open(path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    header = reader.fieldnames or []
  header = [h.strip() for h in header]
  return header, rows


def _align_features(headers: list[str], rows: list[dict[str, str]], feature_names: list[str]) -> np.ndarray:
  header_map = {h.strip(): h for h in headers}
  out = np.zeros((len(rows), len(feature_names)), dtype=np.float64)

  for j, feat in enumerate(feature_names):
    candidates = ALIASES.get(feat, [feat])
    matched_key = None
    for cand in candidates:
      cand = cand.strip()
      if cand in header_map:
        matched_key = header_map[cand]
        break
    if matched_key is None:
      continue

    for i, row in enumerate(rows):
      out[i, j] = _safe_float(row.get(matched_key))

  out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
  return out


def main() -> None:
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

  parser = argparse.ArgumentParser(description="Real CSV inference: print predicted classes only")
  parser.add_argument("--csv", required=True, help="Input flow-feature CSV")
  parser.add_argument("--artifacts-dir", default="artifacts", help="Model artifacts directory")
  parser.add_argument("--processed-dir", default="data/processed", help="Fallback metadata/scaler directory")
  parser.add_argument("--label-col", default="Label", help="Label column in the CSV, ignored by default")
  parser.add_argument("--topk", type=int, default=1, help="Print top-k predicted classes (default 1)")
  parser.add_argument("--limit", type=int, default=0, help="Only predict on first N rows (0 = all)")
  parser.add_argument(
      "--show-prob",
      action="store_true",
      help="Also print the top-1 probability",
  )
  parser.add_argument(
      "--summary-json",
      type=str,
      default="",
      help="Save prediction summary to JSON for UI charts.",
  )
  args = parser.parse_args()

  csv_path = Path(args.csv)
  if not csv_path.is_file():
    raise SystemExit(f"CSV not found: {csv_path}")

  art = Path(args.artifacts_dir)
  proc = Path(args.processed_dir)

  model_path = art / "global_model.keras"
  if not model_path.is_file():
    raise SystemExit(f"Missing model: {model_path}")

  meta_path = art / "metadata.json"
  if not meta_path.is_file():
    meta_path = proc / "metadata.json"
  if not meta_path.is_file():
    raise SystemExit("Missing metadata.json in artifacts or processed dir")

  scaler_path = art / "global_scaler.npz"
  if not scaler_path.is_file():
    scaler_path = proc / "global_scaler.npz"
  if not scaler_path.is_file():
    raise SystemExit("Missing global_scaler.npz in artifacts or processed dir")

  meta = _load_json(meta_path)
  feature_names = list(meta.get("feature_names", []))
  feature_order = [int(i) for i in meta.get("feature_order", [])]
  image_shape = tuple(int(v) for v in meta.get("image_shape", [12, 12, 1]))
  label_classes = list(meta.get("label_classes", []))

  if not feature_names:
    raise SystemExit("metadata.json does not contain feature_names")

  model = tf.keras.models.load_model(model_path, compile=False)
  mean, scale = _load_scaler(scaler_path)

  headers, rows = _read_csv_rows(csv_path)
  if args.limit > 0:
    rows = rows[: args.limit]

  aligned = _align_features(headers, rows, feature_names)
  if aligned.shape[1] != mean.shape[0]:
    raise SystemExit(
        f"Feature width mismatch: aligned CSV has {aligned.shape[1]} columns, scaler expects {mean.shape[0]}"
    )

  x_scaled = ((aligned - mean) / scale).astype(np.float32)

  input_shape = model.input_shape
  if isinstance(input_shape, list):
    input_shape = input_shape[0]
  expected = tuple(input_shape[1:]) if input_shape is not None else ()

  if len(expected) == 1:
    # MLP-like model: feed the raw feature vector.
    if x_scaled.shape[1] != expected[0]:
      raise SystemExit(f"模型期望输入维度 {expected[0]}，但当前对齐后的特征维度为 {x_scaled.shape[1]}")
    x_model = x_scaled
  elif len(expected) == 3:
    # CNN-like model: reorder and reshape to image.
    if feature_order:
      x_scaled = x_scaled[:, feature_order]
    h, w, c = expected
    need = h * w
    if x_scaled.shape[1] > need:
      raise SystemExit(f"Feature width {x_scaled.shape[1]} exceeds image capacity {need}")
    if x_scaled.shape[1] < need:
      x_scaled = np.pad(x_scaled, ((0, 0), (0, need - x_scaled.shape[1])), mode="constant", constant_values=0.0)
    x_model = x_scaled.reshape(-1, h, w, c).astype(np.float32)
  else:
    raise SystemExit(f"模型期望的输入形状不受支持: {input_shape}")

  prob = model.predict(x_model, verbose=0)
  pred_ids = np.argmax(prob, axis=1).astype(np.int64)

  topk = max(1, int(args.topk))
  topk = min(topk, prob.shape[1])
  topk_ids = np.argsort(-prob, axis=1)[:, :topk]

  pred_counts = np.bincount(pred_ids, minlength=len(label_classes)).tolist() if label_classes else []
  max_probs = np.max(prob, axis=1) if len(prob) else np.array([], dtype=np.float32)
  summary = {
      "csv": str(csv_path.name),
      "n_samples": int(len(pred_ids)),
      "label_classes": label_classes,
      "pred_class_counts": pred_counts,
      "avg_top1_prob": float(np.mean(max_probs)) if len(max_probs) else 0.0,
      "min_top1_prob": float(np.min(max_probs)) if len(max_probs) else 0.0,
      "max_top1_prob": float(np.max(max_probs)) if len(max_probs) else 0.0,
      "topk": int(topk),
      "topk_examples": [],
  }

  for i in range(len(pred_ids)):
    pred_name = _resolve_label_name(int(pred_ids[i]), label_classes)
    if topk == 1:
      if args.show_prob:
        print(f"row={i}  pred={pred_name}  prob={float(prob[i, pred_ids[i]]):.6f}")
      else:
        print(f"row={i}  pred={pred_name}")
      if len(summary["topk_examples"]) < 10:
        summary["topk_examples"].append({"row": int(i), "pred": pred_name, "prob": float(prob[i, pred_ids[i]])})
      continue

    parts = []
    example_parts = []
    for j in topk_ids[i]:
      label = _resolve_label_name(int(j), label_classes)
      score = float(prob[i, j])
      parts.append(f"{label}:{score:.6f}")
      example_parts.append({"label": label, "prob": score})
    if len(summary["topk_examples"]) < 10:
      summary["topk_examples"].append({"row": int(i), "pred": pred_name, "topk": example_parts})
    print(f"row={i}  pred={pred_name}  top{topk}={', '.join(parts)}")

  if args.summary_json:
    with open(args.summary_json, "w", encoding="utf-8") as f:
      json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
  main()
