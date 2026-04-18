#!/usr/bin/env python3
"""Authoritative CV for pipeline_files models using src session folds."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_files.all_features import extract_features
from pipeline_files.all_models import MODEL_REGISTRY
from src.data.load import load_train_bars
from src.data.splits import make_session_folds
from src.data.targets import build_train_targets
from src.evaluation.metrics import pnl, sharpe_from_positions
from src.settings import CV_FOLDS, RANDOM_SEED


CLASS_NAME_REGISTRY = {model_cls.__name__: (name, model_cls) for name, model_cls in MODEL_REGISTRY.items()}


def resolve_models(model_name: str):
    if model_name.lower() == "all":
        return list(MODEL_REGISTRY.items())
    if model_name in MODEL_REGISTRY:
        return [(model_name, MODEL_REGISTRY[model_name])]
    if model_name in CLASS_NAME_REGISTRY:
        return [CLASS_NAME_REGISTRY[model_name]]
    available = sorted(list(MODEL_REGISTRY) + list(CLASS_NAME_REGISTRY))
    raise ValueError(f"Unknown model '{model_name}'. Available options: {available}")


def load_training_matrix():
    train_seen, train_unseen = load_train_bars()
    features = extract_features(train_seen).fillna(0.0).sort_index()
    target_return = build_train_targets(train_seen, train_unseen)["target_return"].sort_index()
    features = features.loc[target_return.index]
    return features, target_return


def evaluate_model(model_label: str, model_cls, features: pd.DataFrame, target_return: pd.Series, *, n_folds: int, seed: int):
    fold_rows: list[dict[str, float]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id, train_sessions, valid_sessions in make_session_folds(features.index, n_folds=n_folds, seed=seed):
        X_train = features.loc[train_sessions]
        X_valid = features.loc[valid_sessions]
        y_train = target_return.loc[train_sessions]
        y_valid = target_return.loc[valid_sessions]

        model = model_cls(features.shape[1])
        started_at = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - started_at

        train_position = pd.Series(np.asarray(model.predict(X_train)).reshape(-1), index=X_train.index, dtype=float)
        valid_position = pd.Series(np.asarray(model.predict(X_valid)).reshape(-1), index=X_valid.index, dtype=float)

        train_sharpe = sharpe_from_positions(train_position, y_train)
        valid_sharpe = sharpe_from_positions(valid_position, y_valid)
        valid_pnl = pnl(valid_position, y_valid)

        fold_rows.append(
            {
                "model": model_label,
                "fold": float(fold_id),
                "n_train": float(len(train_sessions)),
                "n_valid": float(len(valid_sessions)),
                "train_sharpe": train_sharpe,
                "valid_sharpe": valid_sharpe,
                "gap": train_sharpe - valid_sharpe,
                "fit_seconds": elapsed,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "model": model_label,
                    "target_position": valid_position,
                    "realized_return": y_valid,
                    "pnl": valid_pnl,
                    "fold": fold_id,
                }
            )
        )

        print(
            f"  Fold {fold_id + 1}/{n_folds}: "
            f"train={train_sharpe:+.4f} "
            f"valid={valid_sharpe:+.4f} "
            f"gap={train_sharpe - valid_sharpe:+.4f} "
            f"time={elapsed:.1f}s"
        )

    fold_summary = pd.DataFrame(fold_rows)
    oof_predictions = pd.concat(oof_frames).sort_index()
    return fold_summary, oof_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run session-fold CV for pipeline_files models.")
    parser.add_argument(
        "--model",
        default="all",
        help="MODEL_REGISTRY label, model class name, or 'all'. Defaults to all.",
    )
    parser.add_argument("--n-folds", type=int, default=CV_FOLDS, help="Number of session folds.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Session fold seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for the aggregated fold summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features, target_return = load_training_matrix()
    print(f"Loaded {len(features)} sessions with {features.shape[1]} features.")

    summaries: list[pd.DataFrame] = []
    for model_label, model_cls in resolve_models(args.model):
        print(f"\nEvaluating {model_label}")
        fold_summary, _ = evaluate_model(
            model_label,
            model_cls,
            features,
            target_return,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        summaries.append(fold_summary)

    all_folds = pd.concat(summaries, ignore_index=True)
    leaderboard = (
        all_folds.groupby("model", as_index=False)
        .agg(
            mean_train_sharpe=("train_sharpe", "mean"),
            mean_valid_sharpe=("valid_sharpe", "mean"),
            std_valid_sharpe=("valid_sharpe", "std"),
            mean_gap=("gap", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .sort_values("mean_valid_sharpe", ascending=False)
    )

    print("\n" + "=" * 86)
    print("PIPELINE_FILES FINAL CV LEADERBOARD")
    print("=" * 86)
    print(leaderboard.to_string(index=False))
    print("=" * 86)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(args.output, index=False)
        print(f"Saved leaderboard to {args.output}")


if __name__ == "__main__":
    main()
