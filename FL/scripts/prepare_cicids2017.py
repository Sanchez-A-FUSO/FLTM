"""
将 CICIDS2017（MachineLearningCSV）抽样、清洗后，按 10 份做分层划分（IID），
保存为 federated 训练可直接读取的 NPZ。

用法示例（支持子目录中的 CSV，例如 MachineLearningCSV/MachineLearningCVE）：
  python scripts/prepare_cicids2017.py --csv-dir MachineLearningCSV --sample-per-file 50000 --output-dir data/processed
  # 留出 20% 分层测试集（test.npz，不参与联邦划分；标准化仅在训练部分 fit）：
  python scripts/prepare_cicids2017.py --csv-dir MachineLearningCSV --test-fraction 0.2 --output-dir data/processed

若未下载数据集，可使用 --demo 生成合成分类数据用于联调。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _read_cicids_csvs(csv_dir: Path, sample_per_file: int | None, random_state: int) -> pd.DataFrame:
  paths = sorted(csv_dir.rglob("*.csv"))
  if not paths:
    raise FileNotFoundError(f"未在目录及其子目录中找到 CSV: {csv_dir}")

  chunks: List[pd.DataFrame] = []
  for p in paths:
    df = pd.read_csv(p, low_memory=False)
    # 常见列名带前导空格
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


def _drop_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    label_classes: list[str],
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
  """Drop rows whose label count < min_count (needed for StratifiedKFold n_splits=min_count). Remap y to 0..K-1."""
  unique, counts = np.unique(y, return_counts=True)
  rare_ids = unique[counts < min_count]
  if rare_ids.size == 0:
    return X, y, label_classes

  print(
      f"以下类别样本数 < {min_count}，已剔除（否则无法满足 10 客户端分层划分对每类最小样本数的要求）。"
      "若需保留稀有类，可增大 --sample-per-file、改为全量（--sample-per-file 0），或减小 --test-fraction。"
  )
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


def rebalance_benign_cap(
    X: np.ndarray,
    y: np.ndarray,
    label_classes: list[str],
    benign_max_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Optionally cap BENIGN samples to improve class balance.

  benign_max_ratio 表示 BENIGN 最多保留为「全部攻击样本数」的多少倍。
  例如 2.0 表示 BENIGN 至多是攻击总数的 2 倍；<=0 表示不下采样。
  """
  if benign_max_ratio <= 0:
    return X, y

  benign_i = None
  for i, name in enumerate(label_classes):
    if str(name).strip().upper() == "BENIGN":
      benign_i = i
      break
  if benign_i is None:
    print("[prepare] 未找到 BENIGN 类，跳过 --benign-max-ratio 下采样。")
    return X, y

  idx_b = np.where(y == benign_i)[0]
  idx_a = np.where(y != benign_i)[0]
  n_b, n_a = len(idx_b), len(idx_a)
  if n_a <= 0:
    print("[prepare] 数据中无攻击类，跳过 --benign-max-ratio 下采样。")
    return X, y

  max_b = int(np.floor(benign_max_ratio * n_a))
  if max_b <= 0:
    raise SystemExit("--benign-max-ratio 过小，导致 BENIGN 最大保留数为 0。请调大该值。")
  if n_b <= max_b:
    print(
        f"[prepare] BENIGN={n_b}，攻击={n_a}，已满足 benign<=ratio*attack（ratio={benign_max_ratio:g}），不做下采样。"
    )
    return X, y

  rng = np.random.default_rng(random_state)
  keep_b = rng.choice(idx_b, size=max_b, replace=False)
  keep = np.concatenate([keep_b, idx_a], axis=0)
  rng.shuffle(keep)
  print(
      f"[prepare] 已启用 BENIGN 下采样：BENIGN {n_b} -> {max_b}，攻击 {n_a}，"
      f"下采样后 benign/attack={max_b / float(n_a):.4f}（目标上限={benign_max_ratio:g}）。"
  )
  return X[keep], y[keep]


def preprocess_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, list[str]]:
  feature_df = df.drop(columns=[label_col], errors="ignore")
  # 去掉明显非数值列
  drop_cols = []
  for c in feature_df.columns:
    if feature_df[c].dtype == object:
      drop_cols.append(c)
  feature_df = feature_df.drop(columns=drop_cols, errors="ignore")

  for c in feature_df.columns:
    feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")

  feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
  return feature_df, list(feature_df.columns)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv-dir", type=str, default="", help="CICIDS2017 MachineLearningCSV 目录")
  parser.add_argument("--label-col", type=str, default="Label", help="标签列名（清洗后不含首尾空格）")
  parser.add_argument("--sample-per-file", type=int, default=20000, help="每个 CSV 最多抽样行数；<=0 表示全量")
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
      "--test-fraction",
      type=float,
      default=0.0,
      help="留出测试集比例 (0,1)，分层抽样；标准化仅在训练部分上 fit。写出 test.npz，不参与联邦客户端划分。0 表示不划分（与旧行为一致）。",
  )
  parser.add_argument(
      "--benign-max-ratio",
      type=float,
      default=0.0,
      help=(
          "可选：对 BENIGN 做下采样，限制 BENIGN <= ratio * (全部攻击样本)。"
          "例如 2.0 表示 BENIGN 最多是攻击总数 2 倍；<=0 表示关闭（默认）。"
      ),
  )
  args = parser.parse_args()

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  n_clients = 10
  test_fraction = float(args.test_fraction)
  if test_fraction != 0.0 and not (0.0 < test_fraction < 1.0):
    raise SystemExit("--test-fraction 必须在 0 与 1 之间（不含端点），或设为 0 表示不留测试集。")

  min_per = max(args.min_per_class, n_clients)
  if 0.0 < test_fraction < 1.0:
    # 分层留出测试后，训练集中每类至少约 (1-test_fraction)*n_class 条；须 ≥ n_clients 才能 StratifiedKFold(10)
    min_total_per_class = int(np.ceil(n_clients / (1.0 - test_fraction)))
    if min_total_per_class > min_per:
      print(
          f"[prepare] 因 test-fraction={test_fraction:g}，每类在全量数据中至少需 "
          f"{min_total_per_class} 条，留出测试后训练侧才能保证每类 ≥{n_clients} 条（10 折分层）。"
          f"已将稀有类剔除门槛提高到 {min_total_per_class}（原为 {min_per}）。"
      )
    min_per = max(min_per, min_total_per_class)

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

  X, y = rebalance_benign_cap(
      X,
      y,
      label_classes,
      benign_max_ratio=float(args.benign_max_ratio),
      random_state=int(args.random_state),
  )
  X, y, label_classes = _drop_rare_classes(X, y, label_classes, min_per)

  _, counts = np.unique(y, return_counts=True)
  if int(counts.min()) < n_clients:
    raise SystemExit(
        f"剔除稀有类后仍无法满足分层划分：最少类样本数为 {int(counts.min())}，需要 >= {n_clients}。"
        "请增大 --sample-per-file 或使用 --sample-per-file 0 全量读取。"
    )

  X_test_out: np.ndarray | None = None
  y_test_out: np.ndarray | None = None

  if test_fraction > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        stratify=y,
        random_state=args.random_state,
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_work = scaler.transform(X_train).astype(np.float32)
    y_work = y_train
    X_test_out = scaler.transform(X_test).astype(np.float32)
    y_test_out = y_test
    _, c_tr = np.unique(y_work, return_counts=True)
    if int(c_tr.min()) < n_clients:
      raise SystemExit(
          "留出测试集后，训练部分无法满足 10 折分层（某类样本过少）。"
          "请减小 --test-fraction 或增大 --sample-per-file。"
      )
  else:
    scaler = StandardScaler()
    X_work = scaler.fit_transform(X).astype(np.float32)
    y_work = y

  skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=args.random_state)
  splits = list(skf.split(np.zeros(len(y_work)), y_work))

  meta = {
      "n_clients": n_clients,
      "n_features": int(X_work.shape[1]),
      "n_classes": int(y_work.max() + 1),
      "label_classes": label_classes,
      "feature_names": feature_names,
      "iid": "stratified_10_fold_partition",
      "min_samples_per_class": min_per,
      "total_rows_after_filter": int(len(y)),
      "train_rows_for_fl": int(len(y_work)),
      "test_split": {
          "enabled": test_fraction > 0,
          "fraction": test_fraction,
          "test_rows": int(len(y_test_out)) if y_test_out is not None else 0,
          "file": "test.npz" if test_fraction > 0 else "",
      },
      "rebalance": {
          "benign_max_ratio": float(args.benign_max_ratio),
          "enabled": float(args.benign_max_ratio) > 0.0,
      },
  }
  with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

  # 保存 scaler（均值方差）便于推理端一致处理
  np.savez(
      out_dir / "global_scaler.npz",
      mean=scaler.mean_.astype(np.float64),
      scale=scaler.scale_.astype(np.float64),
  )

  for client_id, (_, idx) in enumerate(splits):
    np.savez(
        out_dir / f"client_{client_id}.npz",
        x=X_work[idx],
        y=y_work[idx],
    )

  if X_test_out is not None and y_test_out is not None:
    np.savez(out_dir / "test.npz", x=X_test_out, y=y_test_out)

  msg = (
      f"已写入 {out_dir} ，共 {n_clients} 个客户端 NPZ + metadata.json + global_scaler.npz；"
      f"保留 {len(label_classes)} 类；联邦训练用样本 {len(y_work)} 条。"
  )
  if test_fraction > 0:
    msg += f" 测试集 test.npz：{len(y_test_out)} 条（未参与训练划分）。"
  else:
    msg += f"（全量 {len(y)} 条均参与客户端划分，无独立测试集；需测试集请加 --test-fraction，例如 0.2）"
  print(msg)


if __name__ == "__main__":
  main()
