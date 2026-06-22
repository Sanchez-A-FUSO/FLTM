"""Convert a flow-feature CSV into a test NPZ for the current model.

This script is meant to sit after `pcap_to_simple_flow_csv.py`.
It loads the training metadata and global scaler produced by
`prepare_cicids2017_train_split.py`, aligns CSV columns to the expected
training feature space, applies the saved standardization, reorders features
with the saved feature order, and exports a `test.npz` containing:

- x: float32 image tensor shaped (n_samples, image_side, image_side, 1)
- y: int64 label ids when a label column exists, otherwise zeros

If the input CSV has CICIDS2017-style column names, this script will try to
map them to the training schema. Missing columns are filled with zeros and
extra columns are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALIASES = {
    "Destination Port": ["Destination Port", "Dst Port", "Dst Port"],
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


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
  for name in candidates:
    if name in df.columns:
      return pd.to_numeric(df[name], errors="coerce")
  return None


def _resolve_label_column(df: pd.DataFrame, label_col: str) -> pd.Series | None:
  if label_col in df.columns:
    return df[label_col].astype(str)
  for cand in ("Label", "label", "class", "Class", "y"):
    if cand in df.columns:
      return df[cand].astype(str)
  return None


def _align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
  aligned = pd.DataFrame(index=df.index)
  for feat in feature_names:
    series = _pick_column(df, ALIASES.get(feat, [feat]))
    if series is None:
      aligned[feat] = 0.0
    else:
      aligned[feat] = series
  aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
  return aligned.astype(np.float64)


def _to_images(x: np.ndarray, image_shape: tuple[int, int, int], feature_order: list[int]) -> np.ndarray:
  x = x[:, feature_order]
  h, w, c = image_shape
  need = h * w
  if x.shape[1] > need:
    raise SystemExit(f"Feature dimension {x.shape[1]} exceeds image capacity {need}.")
  if x.shape[1] < need:
    x = np.pad(x, ((0, 0), (0, need - x.shape[1])), mode="constant", constant_values=0.0)
  return x.reshape(-1, h, w, c).astype(np.float32)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert flow CSV into test NPZ")
  parser.add_argument("--csv", required=True, help="Input flow CSV")
  parser.add_argument("--output", required=True, help="Output test NPZ path")
  parser.add_argument("--metadata", required=True, help="metadata.json from training")
  parser.add_argument("--scaler", required=True, help="global_scaler.npz from training")
  parser.add_argument("--label-col", default="Label", help="Label column name in the CSV")
  parser.add_argument(
      "--label-override",
      default="",
      help="If set, overwrite all labels with this constant value (e.g. BENIGN)",
  )
  parser.add_argument(
      "--label-map",
      default="",
      help="Optional JSON mapping from string labels to integer ids. If absent, labels are factorized.",
  )
  parser.add_argument(
      "--drop-label",
      action="store_true",
      help="Ignore input labels and write y as zeros.",
  )
  args = parser.parse_args()

  csv_path = Path(args.csv)
  meta_path = Path(args.metadata)
  scaler_path = Path(args.scaler)
  out_path = Path(args.output)

  if not csv_path.is_file():
    raise SystemExit(f"CSV not found: {csv_path}")
  if not meta_path.is_file():
    raise SystemExit(f"metadata not found: {meta_path}")
  if not scaler_path.is_file():
    raise SystemExit(f"scaler not found: {scaler_path}")

  meta = _load_json(meta_path)
  feature_names = list(meta.get("feature_names", []))
  feature_order = [int(i) for i in meta.get("feature_order", [])]
  image_shape = tuple(int(v) for v in meta.get("image_shape", [12, 12, 1]))
  raw_feature_dim = int(meta.get("raw_feature_dim", len(feature_names)))
  label_classes = list(meta.get("label_classes", []))

  if not feature_names:
    raise SystemExit("metadata.json does not contain feature_names")
  if feature_order and len(feature_order) != raw_feature_dim:
    raise SystemExit("feature_order length does not match raw_feature_dim")

  df = pd.read_csv(csv_path, low_memory=False)
  df.columns = [c.strip() for c in df.columns]

  label_series = None if args.drop_label else _resolve_label_column(df, args.label_col)
  if args.label_override:
    label_series = pd.Series([args.label_override] * len(df), index=df.index)

  aligned = _align_features(df, feature_names)
  x_raw = aligned.to_numpy(dtype=np.float64)
  mean, scale = _load_scaler(scaler_path)

  if x_raw.shape[1] != mean.shape[0]:
    raise SystemExit(
        f"Feature width mismatch: csv has {x_raw.shape[1]} columns after alignment, "
        f"but scaler expects {mean.shape[0]}"
    )

  x_scaled = ((x_raw - mean) / scale).astype(np.float32)
  if feature_order:
    x_img = _to_images(x_scaled, image_shape, feature_order)
  else:
    need = image_shape[0] * image_shape[1]
    if x_scaled.shape[1] < need:
      x_scaled = np.pad(x_scaled, ((0, 0), (0, need - x_scaled.shape[1])), mode="constant", constant_values=0.0)
    x_img = x_scaled.reshape(-1, *image_shape).astype(np.float32)

  if label_series is None:
    y = np.zeros(len(df), dtype=np.int64)
  else:
    labels = label_series.astype(str).fillna("UNKNOWN")
    if args.label_map:
      mapping = json.loads(args.label_map)
      if not isinstance(mapping, dict):
        raise SystemExit("--label-map must be a JSON object")
      y = np.array([int(mapping.get(lbl, -1)) for lbl in labels], dtype=np.int64)
    elif label_classes and all(lbl in label_classes for lbl in labels.unique()):
      class_to_id = {name: i for i, name in enumerate(label_classes)}
      y = np.array([class_to_id.get(lbl, -1) for lbl in labels], dtype=np.int64)
    else:
      codes, uniques = pd.factorize(labels, sort=True)
      y = codes.astype(np.int64)
      if len(uniques) > 0:
        print("[csv2npz] label factorization:")
        for i, u in enumerate(uniques[:20]):
          print(f"  {i}: {u}")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, x=x_img, y=y)

  print(f"Wrote {out_path}")
  print(f"  samples: {len(y)}")
  print(f"  x shape: {x_img.shape}")
  print(f"  y shape: {y.shape}")
  print(f"  feature width: {x_raw.shape[1]}")
  print(f"  image shape: {image_shape}")


if __name__ == "__main__":
  main()
