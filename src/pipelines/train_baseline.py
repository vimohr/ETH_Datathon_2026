import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.load import load_bars, load_headlines, load_train_bars
from src.data.targets import build_train_targets
from src.evaluation.validation import run_cross_validation
from src.features.headlines import build_headline_features
from src.features.price import build_price_features
from src.models.baseline import LinearBaselineModel
from src.models.uncertainty import size_positions
from src.paths import OOF_DIR, SUBMISSIONS_DIR, ensure_output_dirs
from src.submission import build_submission, build_submission_metadata, save_submission


def build_feature_matrix(split: str, include_headlines: bool = False) -> pd.DataFrame:
    bars = load_bars(split, "seen")
    features = build_price_features(bars)
    if include_headlines:
        headlines = load_headlines(split, "seen")
        features = features.join(build_headline_features(headlines, sessions=features.index), how="left")
    return features.fillna(0.0).sort_index()


def run_pipeline(test_split: str, include_headlines: bool, output_path=None, cv_only: bool = False):
    ensure_output_dirs()

    train_seen_bars, train_unseen_bars = load_train_bars()
    train_targets = build_train_targets(train_seen_bars, train_unseen_bars)
    train_features = build_price_features(train_seen_bars)

    if include_headlines:
        train_headlines = load_headlines("train", "seen")
        train_features = train_features.join(
            build_headline_features(train_headlines, sessions=train_features.index),
            how="left",
        )

    train_features = train_features.fillna(0.0).sort_index()
    target_return = train_targets["target_return"].sort_index()

    fold_summary, oof_predictions = run_cross_validation(train_features, target_return)
    print(fold_summary.to_string(index=False))
    print(f"Mean CV Sharpe: {fold_summary['sharpe'].mean():.4f}")

    oof_name = "baseline_price_headlines_oof.csv" if include_headlines else "baseline_price_oof.csv"
    oof_path = OOF_DIR / oof_name
    oof_predictions.to_csv(oof_path, index_label="session")
    print(f"Saved OOF predictions to {oof_path}")

    if cv_only:
        return None

    model = LinearBaselineModel().fit(train_features, target_return)
    test_features = build_feature_matrix(test_split, include_headlines=include_headlines)
    predicted_return = model.predict_expected_return(test_features)
    predicted_uncertainty = model.predict_uncertainty(test_features)
    target_position = size_positions(predicted_return, predicted_uncertainty)

    submission = build_submission(test_features.index, target_position)
    expected_sessions = load_bars(test_split, "seen")["session"].unique()

    model_name = "baseline_price_headlines" if include_headlines else "baseline_price"
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = SUBMISSIONS_DIR / f"{timestamp}_{model_name}_{test_split}.csv"
    output_path = Path(output_path)
    metadata = build_submission_metadata(
        model_name=model_name,
        test_split=test_split,
        feature_count=int(train_features.shape[1]),
        mean_cv_sharpe=float(fold_summary["sharpe"].mean()),
        include_headlines=include_headlines,
    )
    saved_path = save_submission(
        submission,
        output_path,
        expected_sessions=expected_sessions,
        metadata=metadata,
        latest_alias=f"latest_{test_split}.csv",
    )
    print(f"Saved submission to {saved_path}")
    return submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline model and create a submission.")
    parser.add_argument(
        "--test-split",
        choices=["public_test", "private_test"],
        default="public_test",
        help="Which test split to score and export.",
    )
    parser.add_argument(
        "--include-headlines",
        action="store_true",
        help="Join simple headline features into the baseline feature matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submission output path. Defaults to outputs/submissions/...",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run cross-validation and write OOF predictions without fitting on all data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        test_split=args.test_split,
        include_headlines=args.include_headlines,
        output_path=args.output,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
