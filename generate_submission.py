#!/usr/bin/env python3
"""Train a pipeline_files model on all train data and export a validated submission."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_files.all_features import extract_features
from pipeline_files.all_models import MODEL_REGISTRY
from src.data.load import load_bars, load_train_bars
from src.data.targets import build_train_targets
from src.paths import SUBMISSIONS_DIR, ensure_output_dirs
from src.submission import (
    build_submission,
    build_submission_metadata,
    combine_split_submissions,
    expected_competition_sessions,
    save_submission,
)


CLASS_NAME_REGISTRY = {model_cls.__name__: model_cls for model_cls in MODEL_REGISTRY.values()}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a validated competition submission from pipeline_files.")
    parser.add_argument(
        "--model-class",
        choices=sorted(CLASS_NAME_REGISTRY),
        default="BestDeep84",
        help="pipeline_files model class to train on all training sessions.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Global multiplier applied to predicted positions before export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to outputs/submissions/<timestamp>_pipeline_<model>_competition.csv",
    )
    return parser.parse_args()


def build_output_path(model_class_name: str, output: Path | None) -> Path:
    if output is not None:
        return output
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return SUBMISSIONS_DIR / f"{timestamp}_pipeline_{model_class_name}_competition.csv"


def main():
    args = parse_args()
    ensure_output_dirs()

    print("Loading training data...")
    train_seen, train_unseen = load_train_bars()
    train_targets = build_train_targets(train_seen, train_unseen)["target_return"].sort_index()

    print("Extracting training features...")
    X_train = extract_features(train_seen).fillna(0.0).sort_index()
    X_train = X_train.loc[train_targets.index]
    print(f"  Training samples: {len(X_train)}, features: {X_train.shape[1]}")

    model_cls = CLASS_NAME_REGISTRY[args.model_class]
    print(f"Training {args.model_class} on all training sessions...")
    model = model_cls(X_train.shape[1])
    model.fit(X_train, train_targets)

    print("Loading test data...")
    public_test_seen = load_bars("public_test", "seen")
    private_test_seen = load_bars("private_test", "seen")

    print("Extracting test features...")
    X_public = extract_features(public_test_seen).fillna(0.0).sort_index()
    X_private = extract_features(private_test_seen).fillna(0.0).sort_index()

    print(f"  Public test sessions:  {len(X_public)}")
    print(f"  Private test sessions: {len(X_private)}")

    public_positions = np.asarray(model.predict(X_public)).reshape(-1) * args.scale
    private_positions = np.asarray(model.predict(X_private)).reshape(-1) * args.scale

    public_submission = build_submission(X_public.index, public_positions)
    private_submission = build_submission(X_private.index, private_positions)
    submission = combine_split_submissions(public_submission, private_submission)

    output_path = build_output_path(args.model_class, args.output)
    metadata = build_submission_metadata(
        model_name=f"pipeline_{args.model_class}",
        test_split="competition",
        feature_count=int(X_train.shape[1]),
        notes=f"pipeline_files model_class={args.model_class}; scale={args.scale}",
    )
    saved_path = save_submission(
        submission,
        output_path,
        expected_sessions=expected_competition_sessions(),
        metadata=metadata,
        latest_alias=f"latest_pipeline_{args.model_class}_competition.csv",
    )

    print(f"\nSubmission saved to {saved_path}")
    print(f"  Total rows: {len(submission)}")
    print(f"  Sessions: {submission['session'].min()} .. {submission['session'].max()}")
    print(
        "  Position range: "
        f"[{submission['target_position'].min():.2f}, {submission['target_position'].max():.2f}]"
    )
    print(f"  Mean position: {submission['target_position'].mean():.4f}")
    print(f"  Std position:  {submission['target_position'].std():.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
