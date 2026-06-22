"""
CICIDS2017 预处理脚本。

目标：
- 读取 CICIDS2017 原始 CSV，或在 --demo 模式下生成合成数据；
- 清洗特征并编码标签；
- 剔除样本数过少的类别，避免后续 10 折分层切分失败；
- 从全部候选样本中抽取 10% 作为联邦训练集；
- 再从剩余样本中抽取 20% 作为独立测试集；
- 将训练集按 10 份做分层划分，生成联邦训练所需的客户端 NPZ；
- 保存标准化器参数与元数据，便于复现。

用法示例：
  python scripts/prepare_cicids2017_train_split.py --csv-dir MachineLearningCSV --output-dir data/processed

若未下载数据集，可使用 --demo 生成合成分类数据用于联调。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _rank_features_by_correlation(X: np.ndarray, feature_names: list[str]) -> tuple[list[int], np.ndarray]:
  """按绝对相关性对特征做近似分组排序，先把相关性高的特征排到一起。"""
  if X.ndim != 2:
    raise ValueError(f"X 期望为二维矩阵，得到 shape={X.shape}")
  n_features = X.shape[1]
  if n_features == 0:
    return [], np.empty((0, 0), dtype=np.float64)
  if n_features == 1:
    return [0], np.ones((1, 1), dtype=np.float64)

  corr = np.corrcoef(X, rowvar=False)
  corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
  abs_corr = np.abs(corr)
  np.fill_diagonal(abs_corr, 0.0)

  # 先把总相关性高的特征作为“中心点”。
  start = int(np.argmax(abs_corr.sum(axis=1)))
  order = [start]
  remaining = set(range(n_features))
  remaining.remove(start)

  while remaining:
    last = order[-1]
    next_idx = max(remaining, key=lambda j: (abs_corr[last, j], abs_corr[:, j].sum(), -j))
    order.append(int(next_idx))
    remaining.remove(next_idx)

  # 再做一次局部微调：把与前一个特征相关性更高的特征尽量靠前。
  ranked: list[int] = [order[0]]
  for idx in order[1:]:
    ranked.append(int(idx))
  return ranked, corr

N_CLIENTS = 10
DEFAULT_TRAIN_FRACTION = 0.10
MAX_TRAIN_CANDIDATE_SAMPLES = 100_000
IMAGE_SIDE = 12


def _read_cicids_csvs(csv_dir: Path, sample_per_file: int | None, random_state: int) -> pd.DataFrame:
  paths = sorted(csv_dir.rglob("*.csv"))
  if not paths:
    raise FileNotFoundError(f"未在目录及其子目录中找到 CSV: {csv_dir}")

  chunks: List[pd.DataFrame] = []
  for p in paths:
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if sample_per_file is not None and sample_per_file > 0 and len(df) > sample_per_file:
      df = df.sample(n=sample_per_file, random_state=random_state)
    chunks.append(df)
  return pd.concat(chunks, ignore_index=True)


def _make_demo(n_samples: int, n_features: int, n_classes: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
  X, y = make_classification(
      n_samples=n_samples,
      n_features=n_features,
      n_informative=min(n_features, max(2, n_features // 2)),
      n_redundant=max(0, n_features // 4),
      n_classes=n_classes,
      n_clusters_per_class=2,
      random_state=random_state,
  )
  return X.astype(np.float32), y.astype(np.int64)


def preprocess_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, list[str]]:
  feature_df = df.drop(columns=[label_col], errors="ignore")
  drop_cols = [c for c in feature_df.columns if feature_df[c].dtype == object]
  feature_df = feature_df.drop(columns=drop_cols, errors="ignore")

  for c in feature_df.columns:
    feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")

  feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
  return feature_df, list(feature_df.columns)


def _drop_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    label_classes: list[str],
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
  unique, counts = np.unique(y, return_counts=True)
  rare_ids = unique[counts < min_count]
  if rare_ids.size == 0:
    return X, y, label_classes

  print(f"[prepare] 以下类别样本数 < {min_count}，已剔除：")
  for rid in rare_ids:
    rid_i = int(rid)
    c = int(np.sum(y == rid))
    name = label_classes[rid_i] if rid_i < len(label_classes) else str(rid_i)
    print(f"  - {name!r} (id={rid_i}, n={c})")

  keep = ~np.isin(y, rare_ids)
  X = X[keep]
  y = y[keep]
  if len(y) == 0:
    raise SystemExit("剔除稀有类后无剩余样本，请增大抽样或检查数据。")

  le2 = LabelEncoder()
  y_new = le2.fit_transform(y).astype(np.int64)
  new_names = [label_classes[int(c)] for c in le2.classes_]
  return X, y_new, new_names


def _print_class_counts(y: np.ndarray, label_classes: list[str], title: str) -> None:
  counts = np.bincount(y.astype(np.int64), minlength=len(label_classes))
  total = int(counts.sum())
  print(f"\n[prepare] {title}（总样本数={total}）")
  for i, name in enumerate(label_classes):
    print(f"  - {name!r}: {int(counts[i])}")


def _to_gray_images(
    X: np.ndarray,
    side: int = IMAGE_SIDE,
    feature_order: list[int] | None = None,
) -> np.ndarray:
  """按给定特征顺序重排后，补零并变成灰度图张量 (n, side, side, 1)。"""
  if X.ndim != 2:
    raise ValueError(f"X 期望为二维矩阵，得到 shape={X.shape}")
  if feature_order is not None:
    X = X[:, feature_order]
  need = side * side
  if X.shape[1] > need:
    raise SystemExit(
      f"特征维度 {X.shape[1]} 超过 {side}x{side}={need}，请增大 IMAGE_SIDE。"
    )
  pad = need - X.shape[1]
  if pad > 0:
    X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
  return X.reshape(-1, side, side, 1).astype(np.float32)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv-dir", type=str, default="", help="CICIDS2017 MachineLearningCSV 目录")
  parser.add_argument("--label-col", type=str, default="Label", help="标签列名（清洗后不含首尾空格）")
  parser.add_argument("--sample-per-file", type=int, default=0, help="每个 CSV 最多抽样行数；<=0 表示全量")
  parser.add_argument("--output-dir", type=str, default="data/processed")
  parser.add_argument("--random-state", type=int, default=42)
  parser.add_argument("--demo", action="store_true", help="不读 CSV，生成演示用表格数据")
  parser.add_argument("--demo-samples", type=int, default=20000)
  parser.add_argument("--demo-features", type=int, default=32)
  parser.add_argument("--demo-classes", type=int, default=8)
  parser.add_argument(
      "--min-per-class",
      type=int,
      default=10,
      help="每类至少保留的样本数（与客户端数一致时才能 StratifiedKFold）；不足的类会被剔除。",
  )
  parser.add_argument(
      "--train-fraction",
      type=float,
      default=DEFAULT_TRAIN_FRACTION,
      help="从全部候选样本中抽取用于训练的比例，默认 0.10。",
  )
  parser.add_argument(
      "--max-train-samples",
      type=int,
      default=MAX_TRAIN_CANDIDATE_SAMPLES,
      help="限制训练候选集总量的上限，默认 100000。<=0 表示不限制。",
  )
  args = parser.parse_args()

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if not (0.0 < args.train_fraction < 1.0):
    raise SystemExit("--train-fraction 必须在 0 与 1 之间。")

  min_total_per_class = int(np.ceil(N_CLIENTS / args.train_fraction))
  min_per = max(int(args.min_per_class), min_total_per_class)

  if args.demo:
    X, y = _make_demo(args.demo_samples, args.demo_features, args.demo_classes, args.random_state)
    label_classes = [str(i) for i in range(int(y.max()) + 1)]
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X = X.astype(np.float64)
  else:
    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_dir():
      raise SystemExit("请提供有效的 --csv-dir，或加 --demo 进行联调。")

    spf = args.sample_per_file if args.sample_per_file > 0 else None
    df = _read_cicids_csvs(csv_dir, spf, args.random_state)
    df.columns = [c.strip() for c in df.columns]
    if args.label_col not in df.columns:
      raise SystemExit(f"找不到标签列 {args.label_col!r}，实际列包含: {list(df.columns)[:30]} ...")

    labels_raw = df[args.label_col].astype(str)
    feature_df, feature_names = preprocess_features(df, args.label_col)
    labels_raw = labels_raw.loc[feature_df.index]

    le = LabelEncoder()
    y = le.fit_transform(labels_raw).astype(np.int64)
    X = feature_df.to_numpy(dtype=np.float64)
    label_classes = [str(c) for c in le.classes_]

  _print_class_counts(y, label_classes, "原始数据读取后")

  X, y, label_classes = _drop_rare_classes(X, y, label_classes, min_per)
  _print_class_counts(y, label_classes, "剔除稀有类后")

  if args.max_train_samples > 0 and len(y) > args.max_train_samples:
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=args.max_train_samples,
        stratify=y,
        random_state=args.random_state,
    )
    print(f"[prepare] 已将候选集总量限制为 {len(y)} 条（上限={int(args.max_train_samples)}）。")
    _print_class_counts(y, label_classes, "限制总量后")

  if len(y) == 0:
    raise SystemExit("混合样本为空，请检查原始数据与筛选条件。")

  n_train = max(1, int(round(len(y) * args.train_fraction)))
  if n_train >= len(y):
    n_train = max(1, len(y) - 1)
  X_train, X_remain, y_train, y_remain = train_test_split(
      X,
      y,
      train_size=n_train,
      stratify=y,
      random_state=args.random_state,
  )

  _print_class_counts(y_train, label_classes, "真正用于联邦训练的 10% 训练集")
  _print_class_counts(y_remain, label_classes, "训练后剩余样本")

  if len(y_remain) == 0:
    raise SystemExit("训练后没有剩余样本，无法再抽取测试集。")
  if len(y_remain) == 1:
    X_test_pool = X_remain
    y_test_pool = y_remain
    X_unused = np.empty((0, X_remain.shape[1]), dtype=X_remain.dtype)
    y_unused = np.empty((0,), dtype=y_remain.dtype)
  else:
    n_test = int(round(len(y_remain) * 0.20))
    n_test = max(1, min(n_test, len(y_remain) - 1))
    X_test_pool, X_unused, y_test_pool, y_unused = train_test_split(
        X_remain,
        y_remain,
        train_size=n_test,
        stratify=y_remain,
        random_state=args.random_state,
    )

  _print_class_counts(y_test_pool, label_classes, "最终独立测试集（剩余样本的 20%）")
  if len(y_unused) > 0:
    _print_class_counts(y_unused, label_classes, "测试集抽取后未使用的剩余样本")

  scaler = StandardScaler()
  scaler.fit(X_train)
  X_work_flat = scaler.transform(X_train).astype(np.float32)
  y_work = y_train
  X_test_flat = scaler.transform(X_test_pool).astype(np.float32)
  y_test_out = y_test_pool

  feature_order, corr_matrix = _rank_features_by_correlation(X_work_flat, feature_names)
  ordered_feature_names = [feature_names[i] for i in feature_order]

  X_work = _to_gray_images(X_work_flat, feature_order=feature_order) if len(X_work_flat) > 0 else X_work_flat.reshape(0, IMAGE_SIDE, IMAGE_SIDE, 1)
  X_test_out = _to_gray_images(X_test_flat, feature_order=feature_order) if len(X_test_flat) > 0 else X_test_flat.reshape(0, IMAGE_SIDE, IMAGE_SIDE, 1)

  _, c_tr = np.unique(y_work, return_counts=True)
  if int(c_tr.min()) < N_CLIENTS:
    raise SystemExit("训练部分无法满足 10 折分层（某类样本过少）。")

  skf = StratifiedKFold(n_splits=N_CLIENTS, shuffle=True, random_state=args.random_state)
  splits = list(skf.split(np.zeros(len(y_work)), y_work))

  meta = {
      "n_clients": N_CLIENTS,
      "raw_feature_dim": int(X_work_flat.shape[1]),
      "n_features": int(X_work.shape[1] * X_work.shape[2]),
      "image_shape": [int(X_work.shape[1]), int(X_work.shape[2]), int(X_work.shape[3])],
      "n_classes": int(y_work.max() + 1),
      "label_classes": label_classes,
      "feature_names": feature_names,
      "feature_order": [int(i) for i in feature_order],
      "ordered_feature_names": ordered_feature_names,
      "feature_correlation_matrix": corr_matrix.astype(np.float64).tolist(),
      "feature_reordering": {
          "enabled": True,
          "method": "greedy_absolute_correlation_chain",
          "description": "先在标准化后的训练集上计算特征相关性，再按相关性把相近特征重排后映射为灰度图。",
      },
      "iid": "stratified_10_fold_partition",
      "selected_classes_policy": {
          "benign": "keep",
          "attack_classes": "keep_all",
      },
      "train_fraction": float(args.train_fraction),
      "min_samples_per_class": min_per,
      "total_rows_after_filter": int(len(y)),
      "train_rows_for_fl": int(len(y_work)),
      "test_split": {
          "enabled": True,
          "fraction_of_remaining": 0.20,
          "test_rows": int(len(y_test_out)),
          "remaining_rows_before_test_split": int(len(y_remain)),
          "file": "test.npz",
          "source": "remaining_after_train_split",
      },
      "max_train_samples": {
          "enabled": int(args.max_train_samples) > 0,
          "limit": int(args.max_train_samples),
      },
  }
  with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

  np.savez(
      out_dir / "global_scaler.npz",
      mean=scaler.mean_.astype(np.float64),
      scale=scaler.scale_.astype(np.float64),
  )

  for client_id, (_, idx) in enumerate(splits):
    np.savez(out_dir / f"client_{client_id}.npz", x=X_work[idx], y=y_work[idx])

  np.savez(out_dir / "test.npz", x=X_test_out, y=y_test_out)

  print(
      f"已写入 {out_dir}：{N_CLIENTS} 个客户端 NPZ、metadata.json、global_scaler.npz、test.npz。"
      f" 保留类别数={len(label_classes)}，联邦训练样本={len(y_work)}，测试集={len(y_test_out)}。"
      f" 原始候选集总量={len(X)}，训练后剩余样本={len(y_remain)}。"
      f" 特征重排已启用，前 10 个顺序特征索引={feature_order[:10]}。"
  )


if __name__ == "__main__":
  main()
