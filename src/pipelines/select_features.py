from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.experiments.catalog import FEATURE_BLOCK_CATALOG, MODEL_CATALOG, format_catalog
from src.experiments.selection import (
    SelectionModelSpec,
    build_frozen_selection_data,
    build_forward_subset_candidates,
    build_subset_candidates,
    clean_candidate_features,
    evaluate_named_subsets,
    greedy_forward_select,
    make_fixed_folds,
    prune_ranked_features,
    rank_candidate_features,
)
from src.paths import REPORTS_DIR, ensure_output_dirs


def _timestamp_string() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _save_csv(frame: pd.DataFrame, path: Path, *, latest_alias: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if latest_alias:
        frame.to_csv(path.parent / latest_alias, index=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank and prune feature columns using fixed-fold Sharpe over the competition target."
    )
    parser.add_argument(
        "--core-block",
        action="append",
        default=None,
        help="Feature block to lock into every subset. Default: price",
    )
    parser.add_argument(
        "--no-core",
        action="store_true",
        help="Do not lock any core feature block. Rank all candidate features on equal footing.",
    )
    parser.add_argument(
        "--candidate-block",
        action="append",
        default=None,
        help="Feature block to rank and prune. Default: price_technical + headline_parser",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of fixed session folds to use.",
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
        help="Model used for feature ranking and subset evaluation.",
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
        "--missing-rate-threshold",
        type=float,
        default=0.40,
        help="Drop candidate columns above this missing-rate threshold.",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=1e-10,
        help="Drop candidate columns below this standard-deviation threshold.",
    )
    parser.add_argument(
        "--keep-top",
        type=int,
        default=60,
        help="How many ranked candidate columns to keep before correlation pruning and optional forward selection.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Absolute-correlation threshold for pruning ranked candidates.",
    )
    parser.add_argument(
        "--subset-strategy",
        choices=["ranked", "forward"],
        default="ranked",
        help="How to construct the final evaluated subsets from the pruned candidate pool.",
    )
    parser.add_argument(
        "--forward-score",
        choices=["oof_sharpe", "mean_cv_sharpe"],
        default="oof_sharpe",
        help="Selection metric used when --subset-strategy forward is enabled.",
    )
    parser.add_argument(
        "--forward-progress-every",
        type=int,
        default=1,
        help="Print forward-selection progress every N selected features. Use 0 to disable.",
    )
    parser.add_argument(
        "--total-size",
        type=int,
        action="append",
        default=None,
        help="Total feature count to evaluate after pruning. May be passed multiple times. Default: 30 and 40",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional debug cap on how many cleaned candidate columns to rank.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print ranking progress every N candidate columns. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="feature_selection",
        help="Filename prefix for saved reports.",
    )
    parser.add_argument(
        "--ranking-output",
        type=Path,
        default=None,
        help="CSV path for the ranked candidate report.",
    )
    parser.add_argument(
        "--cleaning-output",
        type=Path,
        default=None,
        help="CSV path for the candidate cleaning report.",
    )
    parser.add_argument(
        "--pruned-output",
        type=Path,
        default=None,
        help="CSV path for the correlation-pruned ranking report.",
    )
    parser.add_argument(
        "--subset-output",
        type=Path,
        default=None,
        help="CSV path for the final subset comparison report.",
    )
    parser.add_argument(
        "--subset-features-output",
        type=Path,
        default=None,
        help="CSV path listing every feature in every evaluated subset.",
    )
    parser.add_argument(
        "--forward-output",
        type=Path,
        default=None,
        help="CSV path for the greedy forward-selection step report when --subset-strategy forward is used.",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print the supported feature blocks and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the supported models and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_features:
        print(format_catalog("Supported Feature Blocks", FEATURE_BLOCK_CATALOG))
    if args.list_models:
        if args.list_features:
            print("")
        print(format_catalog("Supported Models", MODEL_CATALOG))
    if args.list_features or args.list_models:
        return

    ensure_output_dirs()
    timestamp = _timestamp_string()

    core_blocks = [] if args.no_core else (args.core_block or ["price"])
    candidate_blocks = args.candidate_block or (
        ["price", "price_technical", "headline_parser"] if args.no_core else ["price_technical", "headline_parser"]
    )
    total_sizes = args.total_size or ([10, 15, 20] if args.no_core else [30, 40])

    if any("tfidf" in block for block in [*core_blocks, *candidate_blocks]):
        print("Warning: this selector freezes one full X matrix, so TF-IDF blocks will leak fold vocabulary information.")

    model_spec = SelectionModelSpec(
        name=args.model,
        ridge_alpha=args.ridge_alpha,
        alpha=args.alpha,
        weight_power=args.weight_power,
        min_weight=args.min_weight,
    )
    selection_data = build_frozen_selection_data(
        core_feature_blocks=core_blocks,
        candidate_feature_blocks=candidate_blocks,
    )
    folds = make_fixed_folds(selection_data.features.index, n_folds=args.cv_folds, seed=args.seed)

    print(
        f"Frozen matrix: sessions={len(selection_data.features)} "
        f"core_features={len(selection_data.core_columns)} "
        f"candidate_features={len(selection_data.extra_columns)}"
    )

    cleaning = clean_candidate_features(
        selection_data.features,
        candidate_columns=selection_data.extra_columns,
        missing_rate_threshold=args.missing_rate_threshold,
        variance_floor=args.variance_floor,
    )
    cleaned_candidate_columns = cleaning.loc[cleaning["status"].eq("keep"), "feature"].tolist()
    if args.max_candidates is not None:
        cleaned_candidate_columns = cleaned_candidate_columns[: max(args.max_candidates, 0)]

    print(
        f"Candidates after cleanup: {len(cleaned_candidate_columns)} "
        f"(dropped {len(selection_data.extra_columns) - len(cleaned_candidate_columns)})"
    )
    if not cleaned_candidate_columns:
        raise ValueError("No candidate features survived cleanup.")

    base_result, ranking = rank_candidate_features(
        selection_data.features,
        selection_data.target_return,
        core_columns=selection_data.core_columns,
        candidate_columns=cleaned_candidate_columns,
        folds=folds,
        model_spec=model_spec,
        progress_every=args.progress_every,
    )

    keep_top = max(args.keep_top, 0)
    ranked_keep = ranking.head(keep_top) if keep_top else ranking.copy()
    pruned, kept_columns = prune_ranked_features(
        selection_data.features,
        ranked_keep,
        correlation_threshold=args.corr_threshold,
    )

    forward_steps = None
    if args.subset_strategy == "forward":
        forward_steps, selected_columns = greedy_forward_select(
            selection_data.features,
            selection_data.target_return,
            core_columns=selection_data.core_columns,
            candidate_columns=kept_columns,
            total_feature_sizes=total_sizes,
            folds=folds,
            model_spec=model_spec,
            score_metric=args.forward_score,
            progress_every=args.forward_progress_every,
        )
        subset_map = build_forward_subset_candidates(
            core_columns=selection_data.core_columns,
            selected_extra_columns=selected_columns,
            total_feature_sizes=total_sizes,
        )
    else:
        subset_map = build_subset_candidates(
            core_columns=selection_data.core_columns,
            kept_extra_columns=kept_columns,
            cleaned_candidate_columns=cleaned_candidate_columns,
            total_feature_sizes=total_sizes,
        )
    subset_summary, subset_features = evaluate_named_subsets(
        selection_data.features,
        selection_data.target_return,
        subset_map=subset_map,
        folds=folds,
        model_spec=model_spec,
        core_columns=selection_data.core_columns,
    )

    ranking = ranking.copy()
    ranking.insert(1, "base_mean_cv_sharpe", base_result.mean_cv_sharpe)
    ranking.insert(2, "base_oof_sharpe", base_result.oof_sharpe)

    output_prefix = f"{timestamp}_{args.output_prefix}"
    cleaning_output = args.cleaning_output or REPORTS_DIR / f"{output_prefix}_cleaning.csv"
    ranking_output = args.ranking_output or REPORTS_DIR / f"{output_prefix}_ranking.csv"
    pruned_output = args.pruned_output or REPORTS_DIR / f"{output_prefix}_pruned.csv"
    subset_output = args.subset_output or REPORTS_DIR / f"{output_prefix}_subsets.csv"
    subset_features_output = args.subset_features_output or REPORTS_DIR / f"{output_prefix}_subset_features.csv"
    forward_output = args.forward_output or REPORTS_DIR / f"{output_prefix}_forward_steps.csv"

    saved_cleaning = _save_csv(cleaning, cleaning_output, latest_alias="latest_feature_selection_cleaning.csv")
    saved_ranking = _save_csv(ranking, ranking_output, latest_alias="latest_feature_selection_ranking.csv")
    saved_pruned = _save_csv(pruned, pruned_output, latest_alias="latest_feature_selection_pruned.csv")
    saved_subsets = _save_csv(subset_summary, subset_output, latest_alias="latest_feature_selection_subsets.csv")
    saved_subset_features = _save_csv(
        subset_features,
        subset_features_output,
        latest_alias="latest_feature_selection_subset_features.csv",
    )
    saved_forward = None
    if forward_steps is not None:
        saved_forward = _save_csv(
            forward_steps,
            forward_output,
            latest_alias="latest_feature_selection_forward_steps.csv",
        )

    print(
        f"Base core Sharpe: mean_cv={base_result.mean_cv_sharpe:.4f} "
        f"oof={base_result.oof_sharpe:.4f}"
    )
    print("\nTop Ranked Candidate Features")
    print(ranking.head(10).to_string(index=False))
    print("\nSubset Comparison")
    print(subset_summary.to_string(index=False))
    print(f"\nSaved cleaning report to {saved_cleaning}")
    print(f"Saved ranking report to {saved_ranking}")
    print(f"Saved pruned report to {saved_pruned}")
    print(f"Saved subset report to {saved_subsets}")
    print(f"Saved subset feature list to {saved_subset_features}")
    if saved_forward is not None:
        print(f"Saved forward-selection report to {saved_forward}")


if __name__ == "__main__":
    main()
