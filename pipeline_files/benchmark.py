#!/usr/bin/env python3
"""
Benchmark all Sharpe-AD models on the same feature set.

Usage:
    cd pipeline_files
    python benchmark.py

Produces a sorted leaderboard of train/validation Sharpe ratios.
"""

import time
import numpy as np
import pandas as pd
from pipeline import split_and_train_pipeline, calc_sharpe
from all_features import extract_features
from all_models import MODEL_REGISTRY


def run_benchmark(data_dir="../data"):
    # ------------------------------------------------------------------
    # 1. Load data & features ONCE  (expensive step)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  LOADING DATA & EXTRACTING FEATURES (one-time cost)")
    print("=" * 60)

    train_seen = pd.read_parquet(f"{data_dir}/bars_seen_train.parquet")
    train_unseen = pd.read_parquet(f"{data_dir}/bars_unseen_train.parquet")

    X = extract_features(train_seen)

    train_halfway_close = train_seen.groupby("session")["close"].last()
    train_end_close = train_unseen.groupby("session")["close"].last()
    y = (train_end_close / train_halfway_close) - 1.0
    X = X.loc[y.index]

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_features = X_train.shape[1]
    print(f"\nFeatures: {n_features}  |  Train: {len(X_train)}  |  Val: {len(X_val)}")
    print(f"Feature list: {list(X_train.columns)}\n")

    # ------------------------------------------------------------------
    # 2. Train each model & collect Sharpe metrics
    # ------------------------------------------------------------------
    rows = []

    for name, ModelClass in MODEL_REGISTRY.items():
        print("-" * 60)
        print(f"  Training:  {name}")
        print("-" * 60)

        model = ModelClass(n_features)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_sharpe = calc_sharpe(train_preds, y_train.values)
        val_sharpe = calc_sharpe(val_preds, y_val.values)

        rows.append(
            {
                "model": name,
                "train_sharpe": round(train_sharpe, 4),
                "val_sharpe": round(val_sharpe, 4),
                "overfit_gap": round(train_sharpe - val_sharpe, 4),
                "time_sec": round(elapsed, 1),
            }
        )
        print(
            f"  => Train Sharpe: {train_sharpe:.4f}  "
            f"|  Val Sharpe: {val_sharpe:.4f}  "
            f"|  Gap: {train_sharpe - val_sharpe:.4f}  "
            f"|  {elapsed:.1f}s\n"
        )

    # ------------------------------------------------------------------
    # 3. Print the leaderboard
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking

    print("\n")
    print("=" * 70)
    print("               SHARPE-AD MODEL BENCHMARK LEADERBOARD")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    # Also save to CSV for later reference
    df.to_csv("benchmark_results.csv", index_label="rank")
    print("\nResults saved to benchmark_results.csv")

    return df


if __name__ == "__main__":
    run_benchmark()
