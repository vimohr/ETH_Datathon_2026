from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.load import load_bars, load_headlines, load_train_bars
from src.data.targets import build_train_targets
from src.experiments.config import FeatureSpec
from src.experiments.features import build_feature_block, build_feature_matrix
from src.experiments.selection import (
    SelectionModelSpec,
    build_selection_model,
    evaluate_feature_subset,
    make_fixed_folds,
)
from src.kaggle_utils import DEFAULT_COMPETITION, submit_competition_file
from src.models.uncertainty import size_positions
from src.paths import OOF_DIR, SUBMISSIONS_DIR, REPORTS_DIR, ROOT, ensure_output_dirs
from src.submission import (
    build_submission,
    build_submission_metadata,
    combine_split_submissions,
    expected_competition_sessions,
    expected_sessions,
    save_submission,
)


def _timestamp_string() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "subset"


def _resolve_subset_columns(report_path: Path, subset_name: str) -> list[str]:
    report = pd.read_csv(report_path)
    required_columns = {"subset_name", "feature"}
    missing = required_columns - set(report.columns)
    if missing:
        raise ValueError(f"Subset feature report missing required columns: {sorted(missing)}")

    subset_rows = report.loc[report["subset_name"].astype(str) == subset_name].copy()
    if subset_rows.empty:
        available = ", ".join(sorted(report["subset_name"].astype(str).unique()))
        raise ValueError(f"Unknown subset_name={subset_name!r}. Available values: {available}")

    if "feature_order" in subset_rows.columns:
        subset_rows = subset_rows.sort_values("feature_order")

    features = subset_rows["feature"].astype(str).tolist()
    if not features:
        raise ValueError(f"Subset {subset_name!r} resolved to zero features.")
    return features


def _infer_feature_blocks(selected_columns: list[str]) -> tuple[FeatureSpec, ...]:
    blocks = sorted({column.split("__", 1)[0] for column in selected_columns})
    return tuple(FeatureSpec(name=block) for block in blocks)


def _fit_feature_blocks(feature_specs: tuple[FeatureSpec, ...], *, train_bars: pd.DataFrame, train_headlines: pd.DataFrame, sessions):
    block_instances = [build_feature_block(spec) for spec in feature_specs]
    for block in block_instances:
        block.fit(bars=train_bars, headlines=train_headlines, sessions=sessions)
    return block_instances


def _build_selected_frame(
    feature_specs: tuple[FeatureSpec, ...],
    block_instances,
    *,
    bars: pd.DataFrame,
    headlines: pd.DataFrame,
    sessions,
    selected_columns: list[str],
) -> pd.DataFrame:
    frame = build_feature_matrix(
        feature_specs,
        block_instances,
        bars=bars,
        headlines=headlines,
        sessions=sessions,
    ).sort_index()
    overlapping_columns = [column for column in selected_columns if column in frame.columns]
    if not overlapping_columns:
        raise ValueError("None of the selected columns were present in the built feature matrix.")
    return frame.reindex(columns=selected_columns, fill_value=0.0).copy()


def _save_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index_label="session")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train and submit one selected feature subset.")
    parser.add_argument(
        "--subset-features-report",
        type=Path,
        default=REPORTS_DIR / "latest_feature_selection_subset_features.csv",
        help="CSV file produced by src.pipelines.select_features with one row per subset feature.",
    )
    parser.add_argument(
        "--subset-name",
        required=True,
        help="Subset name from the subset-features report, for example top_pruned_total_30.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of fixed session folds for the CV summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the fixed session folds.",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "ridge", "weighted_linear", "weighted_ridge"],
        default="linear",
        help="Model to fit on the selected subset.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=2000.0,
        help="Internal ridge penalty for model=linear.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Penalty used by ridge / weighted_ridge.",
    )
    parser.add_argument(
        "--weight-power",
        type=float,
        default=0.5,
        help="Magnitude weighting exponent for weighted models.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.25,
        help="Minimum sample weight for weighted models.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run CV and fit the full model, but do not write test submissions.",
    )
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Generate both split submissions plus a combined competition file.",
    )
    parser.add_argument(
        "--test-split",
        choices=["public_test", "private_test"],
        default="public_test",
        help="Which split to export when --competition is not used.",
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=None,
        help="Output path for the public-test submission when --competition is used.",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=None,
        help="Output path for the private-test submission when --competition is used.",
    )
    parser.add_argument(
        "--competition-output",
        type=Path,
        default=None,
        help="Output path for the combined competition submission when --competition is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the single split submission when --competition is not used.",
    )
    parser.add_argument(
        "--submit-kaggle",
        action="store_true",
        help="Submit the combined competition file to Kaggle after generation.",
    )
    parser.add_argument(
        "--submission-message",
        default=None,
        help="Submission message used when --submit-kaggle is enabled.",
    )
    parser.add_argument(
        "--competition-name",
        default=DEFAULT_COMPETITION,
        help="Kaggle competition slug.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load Kaggle credentials from.",
    )
    args = parser.parse_args()
    if args.submit_kaggle and not args.competition:
        parser.error("--submit-kaggle requires --competition.")
    if args.submit_kaggle and not args.submission_message:
        parser.error("--submission-message is required when --submit-kaggle is used.")
    return args


def main():
    args = parse_args()
    ensure_output_dirs()

    subset_report = args.subset_features_report
    if not subset_report.is_absolute():
        subset_report = ROOT / subset_report
    selected_columns = _resolve_subset_columns(subset_report, args.subset_name)
    feature_specs = _infer_feature_blocks(selected_columns)
    model_spec = SelectionModelSpec(
        name=args.model,
        ridge_alpha=args.ridge_alpha,
        alpha=args.alpha,
        weight_power=args.weight_power,
        min_weight=args.min_weight,
    )

    train_seen_bars, train_unseen_bars = load_train_bars()
    train_headlines = load_headlines("train", "seen")
    target_return = build_train_targets(train_seen_bars, train_unseen_bars)["target_return"].sort_index()
    sessions = target_return.index

    block_instances = _fit_feature_blocks(
        feature_specs,
        train_bars=train_seen_bars,
        train_headlines=train_headlines,
        sessions=sessions,
    )
    train_features = _build_selected_frame(
        feature_specs,
        block_instances,
        bars=train_seen_bars,
        headlines=train_headlines,
        sessions=sessions,
        selected_columns=selected_columns,
    )
    folds = make_fixed_folds(train_features.index, n_folds=args.cv_folds, seed=args.seed)
    cv_result = evaluate_feature_subset(
        train_features,
        target_return.loc[train_features.index],
        columns=train_features.columns,
        folds=folds,
        model_spec=model_spec,
    )

    subset_slug = _slugify(args.subset_name)
    timestamp = _timestamp_string()
    oof_output = OOF_DIR / f"{timestamp}_{subset_slug}_oof.csv"
    _save_frame(cv_result.oof_predictions, oof_output)

    model = build_selection_model(model_spec).fit(train_features, target_return.loc[train_features.index])

    notes = json.dumps(
        {
            "subset_name": args.subset_name,
            "subset_features_report": str(subset_report),
            "feature_blocks": [spec.name for spec in feature_specs],
            "model": {
                "name": model_spec.name,
                "ridge_alpha": model_spec.ridge_alpha,
                "alpha": model_spec.alpha,
                "weight_power": model_spec.weight_power,
                "min_weight": model_spec.min_weight,
            },
        },
        sort_keys=True,
    )
    metadata_base = build_submission_metadata(
        model_name=f"selected_subset_{subset_slug}",
        test_split="train",
        feature_count=len(selected_columns),
        mean_cv_sharpe=cv_result.mean_cv_sharpe,
        include_headlines=any(spec.name.startswith("headline") for spec in feature_specs),
        notes=notes,
    )

    print(
        f"Subset {args.subset_name}: features={len(selected_columns)} "
        f"mean_cv_sharpe={cv_result.mean_cv_sharpe:.4f} "
        f"oof_sharpe={cv_result.oof_sharpe:.4f}"
    )
    print(f"Saved OOF predictions to {oof_output}")

    if args.cv_only:
        return

    def predict_split(split: str) -> pd.DataFrame:
        bars = load_bars(split, "seen")
        headlines = load_headlines(split, "seen")
        split_sessions = pd.Index(sorted(bars["session"].unique()))
        split_features = _build_selected_frame(
            feature_specs,
            block_instances,
            bars=bars,
            headlines=headlines,
            sessions=split_sessions,
            selected_columns=selected_columns,
        )
        predicted_return = model.predict_expected_return(split_features)
        predicted_uncertainty = model.predict_uncertainty(split_features)
        target_position = size_positions(predicted_return, predicted_uncertainty)
        return build_submission(split_features.index, target_position)

    if not args.competition:
        submission = predict_split(args.test_split)
        output_path = args.output or SUBMISSIONS_DIR / f"{timestamp}_{subset_slug}_{args.test_split}.csv"
        metadata = dict(metadata_base)
        metadata["test_split"] = args.test_split
        saved_path = save_submission(
            submission,
            output_path,
            expected_sessions=expected_sessions(args.test_split),
            metadata=metadata,
            latest_alias=f"latest_{subset_slug}_{args.test_split}.csv",
        )
        print(f"Saved submission to {saved_path}")
        return

    public_submission = predict_split("public_test")
    private_submission = predict_split("private_test")
    public_output = args.public_output or SUBMISSIONS_DIR / f"{timestamp}_{subset_slug}_public_test.csv"
    private_output = args.private_output or SUBMISSIONS_DIR / f"{timestamp}_{subset_slug}_private_test.csv"
    public_metadata = dict(metadata_base)
    public_metadata["test_split"] = "public_test"
    private_metadata = dict(metadata_base)
    private_metadata["test_split"] = "private_test"
    saved_public = save_submission(
        public_submission,
        public_output,
        expected_sessions=expected_sessions("public_test"),
        metadata=public_metadata,
        latest_alias=f"latest_{subset_slug}_public_test.csv",
    )
    saved_private = save_submission(
        private_submission,
        private_output,
        expected_sessions=expected_sessions("private_test"),
        metadata=private_metadata,
        latest_alias=f"latest_{subset_slug}_private_test.csv",
    )

    competition_submission = combine_split_submissions(public_submission, private_submission)
    competition_output = args.competition_output or SUBMISSIONS_DIR / f"{timestamp}_{subset_slug}_competition.csv"
    competition_metadata = dict(metadata_base)
    competition_metadata.update(
        {
            "test_split": "competition",
            "competition_name": args.competition_name,
            "source_public_submission": str(saved_public),
            "source_private_submission": str(saved_private),
            "row_count": int(len(competition_submission)),
        }
    )
    saved_competition = save_submission(
        competition_submission,
        competition_output,
        expected_sessions=expected_competition_sessions(),
        metadata=competition_metadata,
        latest_alias=f"latest_{subset_slug}_competition.csv",
    )

    print(f"Saved public submission to {saved_public}")
    print(f"Saved private submission to {saved_private}")
    print(f"Saved competition submission to {saved_competition}")

    if args.submit_kaggle:
        message, loaded_files = submit_competition_file(
            saved_competition,
            message=args.submission_message,
            competition=args.competition_name,
            env_file=args.env_file,
        )
        if loaded_files:
            print("Loaded Kaggle credentials from:")
            for path in loaded_files:
                print(f"- {path}")
        print(message)


if __name__ == "__main__":
    main()
