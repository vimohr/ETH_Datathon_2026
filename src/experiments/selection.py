from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.load import load_headlines, load_train_bars
from src.data.splits import make_session_folds
from src.data.targets import build_train_targets
from src.evaluation.metrics import pnl, sharpe_from_positions
from src.experiments.config import FeatureSpec
from src.experiments.features import build_feature_block, build_feature_matrix
from src.models.baseline import LinearBaselineModel
from src.models.ridge import RidgeRegressionModel
from src.models.uncertainty import size_positions
from src.models.weighted import WeightedRidgeRegressionModel


@dataclass(frozen=True)
class SelectionModelSpec:
    name: str = "linear"
    ridge_alpha: float = 2000.0
    alpha: float = 10.0
    weight_power: float = 0.5
    min_weight: float = 0.25


@dataclass(frozen=True)
class FrozenSelectionData:
    features: pd.DataFrame
    target_return: pd.Series
    core_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]


@dataclass
class CrossValidationResult:
    columns: tuple[str, ...]
    fold_summary: pd.DataFrame
    oof_predictions: pd.DataFrame

    @property
    def mean_cv_sharpe(self) -> float:
        return float(self.fold_summary["sharpe"].mean()) if not self.fold_summary.empty else 0.0

    @property
    def std_cv_sharpe(self) -> float:
        return float(self.fold_summary["sharpe"].std(ddof=0)) if not self.fold_summary.empty else 0.0

    @property
    def oof_sharpe(self) -> float:
        if self.oof_predictions.empty:
            return 0.0
        return sharpe_from_positions(
            self.oof_predictions["target_position"],
            self.oof_predictions["realized_return"],
        )


def _resolved_feature_specs(feature_block_names: Iterable[str]) -> tuple[FeatureSpec, ...]:
    return tuple(FeatureSpec(name=str(name).strip()) for name in feature_block_names if str(name).strip())


def _validate_unique_aliases(feature_specs: tuple[FeatureSpec, ...]) -> None:
    aliases = [spec.resolved_alias for spec in feature_specs]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Feature block aliases must be unique within one selection run.")


def build_frozen_selection_data(
    *,
    core_feature_blocks: Iterable[str],
    candidate_feature_blocks: Iterable[str],
) -> FrozenSelectionData:
    core_specs = _resolved_feature_specs(core_feature_blocks)
    candidate_specs = _resolved_feature_specs(candidate_feature_blocks)
    if not candidate_specs:
        raise ValueError("At least one candidate feature block is required.")

    all_specs = core_specs + candidate_specs
    _validate_unique_aliases(all_specs)

    train_seen_bars, train_unseen_bars = load_train_bars()
    train_headlines = load_headlines("train", "seen")
    target_return = build_train_targets(train_seen_bars, train_unseen_bars)["target_return"].sort_index()
    sessions = target_return.index

    block_instances = [build_feature_block(spec) for spec in all_specs]
    for block in block_instances:
        block.fit(bars=train_seen_bars, headlines=train_headlines, sessions=sessions)

    features = build_feature_matrix(
        all_specs,
        block_instances,
        bars=train_seen_bars,
        headlines=train_headlines,
        sessions=sessions,
    ).sort_index()

    core_prefixes = {f"{spec.resolved_alias}__" for spec in core_specs}
    candidate_prefixes = {f"{spec.resolved_alias}__" for spec in candidate_specs}
    core_columns = tuple(
        column for column in features.columns if any(column.startswith(prefix) for prefix in core_prefixes)
    )
    extra_columns = tuple(
        column for column in features.columns if any(column.startswith(prefix) for prefix in candidate_prefixes)
    )

    if not extra_columns:
        raise ValueError("Candidate feature blocks produced zero columns.")

    return FrozenSelectionData(
        features=features,
        target_return=target_return.loc[features.index],
        core_columns=core_columns,
        extra_columns=extra_columns,
    )


def make_fixed_folds(sessions, *, n_folds: int, seed: int):
    return list(make_session_folds(sessions, n_folds=n_folds, seed=seed))


def build_zero_baseline_result(target_return: pd.Series, *, folds) -> CrossValidationResult:
    target_series = pd.Series(target_return).astype(float).sort_index()
    fold_rows: list[dict[str, float]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id, _, valid_sessions in folds:
        valid_target = target_series.loc[valid_sessions]
        zero_series = pd.Series(0.0, index=valid_target.index, dtype=float)
        one_series = pd.Series(1.0, index=valid_target.index, dtype=float)
        fold_rows.append(
            {
                "fold": float(fold_id),
                "n_train": float(len(target_series) - len(valid_sessions)),
                "n_valid": float(len(valid_sessions)),
                "feature_count": 0.0,
                "sharpe": 0.0,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "predicted_return": zero_series,
                    "predicted_uncertainty": one_series,
                    "target_position": zero_series,
                    "realized_return": valid_target,
                    "pnl": zero_series,
                    "fold": fold_id,
                }
            )
        )

    return CrossValidationResult(
        columns=tuple(),
        fold_summary=pd.DataFrame(fold_rows),
        oof_predictions=pd.concat(oof_frames).sort_index(),
    )


def build_selection_model(model_spec: SelectionModelSpec):
    if model_spec.name == "linear":
        return LinearBaselineModel(ridge_alpha=model_spec.ridge_alpha)
    if model_spec.name == "ridge":
        return RidgeRegressionModel(alpha=model_spec.alpha)
    if model_spec.name == "weighted_linear":
        return WeightedRidgeRegressionModel(
            alpha=0.0,
            weight_power=model_spec.weight_power,
            min_weight=model_spec.min_weight,
        )
    if model_spec.name == "weighted_ridge":
        return WeightedRidgeRegressionModel(
            alpha=model_spec.alpha,
            weight_power=model_spec.weight_power,
            min_weight=model_spec.min_weight,
        )
    raise ValueError(f"Unsupported model={model_spec.name!r}.")


def evaluate_feature_subset(
    features: pd.DataFrame,
    target_return: pd.Series,
    *,
    columns: Iterable[str],
    folds,
    model_spec: SelectionModelSpec,
) -> CrossValidationResult:
    selected_columns = tuple(columns)
    if not selected_columns:
        raise ValueError("At least one feature column is required.")

    feature_frame = features.loc[:, selected_columns].fillna(0.0).sort_index()
    target_series = pd.Series(target_return, index=feature_frame.index).astype(float).sort_index()

    fold_rows: list[dict[str, float]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id, train_sessions, valid_sessions in folds:
        train_features = feature_frame.loc[train_sessions]
        valid_features = feature_frame.loc[valid_sessions]
        train_target = target_series.loc[train_sessions]
        valid_target = target_series.loc[valid_sessions]

        model = build_selection_model(model_spec).fit(train_features, train_target)
        predicted_return = model.predict_expected_return(valid_features)
        predicted_uncertainty = model.predict_uncertainty(valid_features)
        target_position = size_positions(predicted_return, predicted_uncertainty)
        fold_pnl = pnl(target_position, valid_target)
        fold_sharpe = sharpe_from_positions(target_position, valid_target)

        fold_rows.append(
            {
                "fold": float(fold_id),
                "n_train": float(len(train_sessions)),
                "n_valid": float(len(valid_sessions)),
                "feature_count": float(len(selected_columns)),
                "sharpe": fold_sharpe,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "predicted_return": predicted_return,
                    "predicted_uncertainty": predicted_uncertainty,
                    "target_position": target_position,
                    "realized_return": valid_target,
                    "pnl": fold_pnl,
                    "fold": fold_id,
                }
            )
        )

    return CrossValidationResult(
        columns=selected_columns,
        fold_summary=pd.DataFrame(fold_rows),
        oof_predictions=pd.concat(oof_frames).sort_index(),
    )


def clean_candidate_features(
    features: pd.DataFrame,
    *,
    candidate_columns: Iterable[str],
    missing_rate_threshold: float = 0.40,
    variance_floor: float = 1e-10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen_signatures: dict[bytes, str] = {}

    for column in candidate_columns:
        series = pd.to_numeric(features[column], errors="coerce")
        missing_rate = float(series.isna().mean())
        filled = series.fillna(0.0)
        std = float(filled.std(ddof=0))
        nunique = int(filled.nunique(dropna=False))
        nonzero_rate = float((filled.abs() > 1e-12).mean())
        duplicate_of = ""
        status = "keep"
        drop_reason = ""

        if missing_rate > missing_rate_threshold:
            status = "drop"
            drop_reason = "missing_rate"
        elif nunique <= 1 or std <= variance_floor:
            status = "drop"
            drop_reason = "near_zero_variance"
        else:
            signature = filled.to_numpy(dtype=float, copy=False).tobytes()
            duplicate_of = seen_signatures.get(signature, "")
            if duplicate_of:
                status = "drop"
                drop_reason = "exact_duplicate"
            else:
                seen_signatures[signature] = column

        rows.append(
            {
                "feature": column,
                "block": column.split("__", 1)[0],
                "feature_name": column.split("__", 1)[1] if "__" in column else column,
                "missing_rate": missing_rate,
                "std": std,
                "nunique": float(nunique),
                "nonzero_rate": nonzero_rate,
                "status": status,
                "drop_reason": drop_reason,
                "duplicate_of": duplicate_of,
            }
        )

    return pd.DataFrame(rows).sort_values(["status", "block", "feature_name"]).reset_index(drop=True)


def rank_candidate_features(
    features: pd.DataFrame,
    target_return: pd.Series,
    *,
    core_columns: Iterable[str],
    candidate_columns: Iterable[str],
    folds,
    model_spec: SelectionModelSpec,
    progress_every: int = 25,
) -> tuple[CrossValidationResult, pd.DataFrame]:
    core_columns = tuple(core_columns)
    candidate_columns = tuple(candidate_columns)
    if not candidate_columns:
        raise ValueError("At least one candidate feature is required.")

    if core_columns:
        base_result = evaluate_feature_subset(
            features,
            target_return,
            columns=core_columns,
            folds=folds,
            model_spec=model_spec,
        )
    else:
        base_result = build_zero_baseline_result(target_return, folds=folds)
    base_fold_sharpes = base_result.fold_summary["sharpe"].to_numpy(dtype=float)

    spearman_corr = features.loc[:, candidate_columns].corrwith(target_return, method="spearman")
    pearson_corr = features.loc[:, candidate_columns].corrwith(target_return, method="pearson")

    rows: list[dict[str, object]] = []
    total = len(candidate_columns)
    for index, column in enumerate(candidate_columns, start=1):
        result = evaluate_feature_subset(
            features,
            target_return,
            columns=core_columns + (column,),
            folds=folds,
            model_spec=model_spec,
        )
        fold_sharpes = result.fold_summary["sharpe"].to_numpy(dtype=float)
        fold_delta = fold_sharpes - base_fold_sharpes
        feature_values = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

        rows.append(
            {
                "feature": column,
                "block": column.split("__", 1)[0],
                "feature_name": column.split("__", 1)[1] if "__" in column else column,
                "feature_count": float(len(result.columns)),
                "mean_cv_sharpe": result.mean_cv_sharpe,
                "std_cv_sharpe": result.std_cv_sharpe,
                "oof_sharpe": result.oof_sharpe,
                "delta_mean_cv_sharpe": result.mean_cv_sharpe - base_result.mean_cv_sharpe,
                "delta_oof_sharpe": result.oof_sharpe - base_result.oof_sharpe,
                "positive_fold_count": float(np.sum(fold_delta > 0.0)),
                "non_negative_fold_count": float(np.sum(fold_delta >= 0.0)),
                "mean_fold_delta_sharpe": float(fold_delta.mean()),
                "std_fold_delta_sharpe": float(fold_delta.std(ddof=0)),
                "spearman_corr": float(spearman_corr.get(column, np.nan)),
                "abs_spearman_corr": float(abs(spearman_corr.get(column, np.nan))),
                "pearson_corr": float(pearson_corr.get(column, np.nan)),
                "abs_pearson_corr": float(abs(pearson_corr.get(column, np.nan))),
                "nonzero_rate": float((feature_values.abs() > 1e-12).mean()),
                "std": float(feature_values.std(ddof=0)),
            }
        )

        if progress_every > 0 and (index == total or index % progress_every == 0):
            print(f"Ranked {index}/{total} candidate features")

    ranking = pd.DataFrame(rows).sort_values(
        ["delta_oof_sharpe", "positive_fold_count", "abs_spearman_corr", "feature"],
        ascending=[False, False, False, True],
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    return base_result, ranking.reset_index(drop=True)


def _selection_metric_value(result: CrossValidationResult, metric_name: str) -> float:
    if metric_name == "oof_sharpe":
        return result.oof_sharpe
    if metric_name == "mean_cv_sharpe":
        return result.mean_cv_sharpe
    raise ValueError(f"Unsupported selection metric={metric_name!r}.")


def prune_ranked_features(
    features: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    correlation_threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str]]:
    if ranking.empty:
        return ranking.copy(), []

    ordered_features = ranking["feature"].tolist()
    abs_corr = features.loc[:, ordered_features].corr().abs().fillna(0.0)

    kept: list[str] = []
    kept_rank = 0
    rows: list[dict[str, object]] = []
    for row in ranking.to_dict(orient="records"):
        feature = str(row["feature"])
        correlated_with = ""
        max_abs_corr = 0.0
        status = "keep"
        pruned_rank = np.nan
        if kept:
            corr_to_kept = abs_corr.loc[feature, kept]
            max_abs_corr = float(corr_to_kept.max())
            if max_abs_corr > correlation_threshold:
                correlated_with = str(corr_to_kept.idxmax())
                status = "drop"
            else:
                kept.append(feature)
                kept_rank += 1
                pruned_rank = float(kept_rank)
        else:
            kept.append(feature)
            kept_rank += 1
            pruned_rank = float(kept_rank)

        rows.append(
            {
                **row,
                "max_abs_corr_to_kept": max_abs_corr,
                "correlated_with": correlated_with,
                "prune_status": status,
                "pruned_rank": pruned_rank,
            }
        )

    pruned = pd.DataFrame(rows)
    return pruned, kept


def greedy_forward_select(
    features: pd.DataFrame,
    target_return: pd.Series,
    *,
    core_columns: Iterable[str],
    candidate_columns: Iterable[str],
    total_feature_sizes: Iterable[int],
    folds,
    model_spec: SelectionModelSpec,
    score_metric: str = "oof_sharpe",
    progress_every: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    core_columns = tuple(core_columns)
    requested_sizes = sorted({int(size) for size in total_feature_sizes if int(size) > 0})
    if not requested_sizes:
        raise ValueError("At least one positive total feature size is required for forward selection.")

    remaining = [column for column in candidate_columns if column not in core_columns]
    max_total_size = max(requested_sizes)
    extra_budget = max(max_total_size - len(core_columns), 0)
    total_steps = min(extra_budget, len(remaining))
    selected_extra: list[str] = []
    rows: list[dict[str, object]] = []

    for step in range(1, total_steps + 1):
        current_columns = core_columns + tuple(selected_extra)
        best_feature = ""
        best_result: CrossValidationResult | None = None
        best_score = float("-inf")
        best_secondary = float("-inf")
        best_std = float("inf")

        for feature in remaining:
            result = evaluate_feature_subset(
                features,
                target_return,
                columns=current_columns + (feature,),
                folds=folds,
                model_spec=model_spec,
            )
            score = _selection_metric_value(result, score_metric)
            secondary = result.mean_cv_sharpe if score_metric != "mean_cv_sharpe" else result.oof_sharpe
            is_better = False
            if best_result is None or score > best_score + 1e-12:
                is_better = True
            elif abs(score - best_score) <= 1e-12 and secondary > best_secondary + 1e-12:
                is_better = True
            elif (
                abs(score - best_score) <= 1e-12
                and abs(secondary - best_secondary) <= 1e-12
                and result.std_cv_sharpe < best_std - 1e-12
            ):
                is_better = True
            elif (
                abs(score - best_score) <= 1e-12
                and abs(secondary - best_secondary) <= 1e-12
                and abs(result.std_cv_sharpe - best_std) <= 1e-12
                and (not best_feature or feature < best_feature)
            ):
                is_better = True

            if is_better:
                best_feature = feature
                best_result = result
                best_score = score
                best_secondary = secondary
                best_std = result.std_cv_sharpe

        if best_result is None:
            break

        selected_extra.append(best_feature)
        remaining.remove(best_feature)
        selected_columns = core_columns + tuple(selected_extra)
        rows.append(
            {
                "step": step,
                "added_feature": best_feature,
                "feature_count": len(selected_columns),
                "extra_feature_count": len(selected_extra),
                "selection_metric": score_metric,
                "selection_score": best_score,
                "mean_cv_sharpe": best_result.mean_cv_sharpe,
                "std_cv_sharpe": best_result.std_cv_sharpe,
                "oof_sharpe": best_result.oof_sharpe,
                "selected_features": " | ".join(selected_columns),
            }
        )

        if progress_every > 0 and (step == total_steps or step % progress_every == 0):
            print(
                f"Forward-selected {step}/{total_steps} features "
                f"(last add={best_feature}, {score_metric}={best_score:.4f})"
            )

    return pd.DataFrame(rows), selected_extra


def build_subset_candidates(
    *,
    core_columns: Iterable[str],
    kept_extra_columns: Iterable[str],
    cleaned_candidate_columns: Iterable[str],
    total_feature_sizes: Iterable[int],
) -> dict[str, tuple[str, ...]]:
    core_columns = tuple(core_columns)
    kept_extra_columns = tuple(kept_extra_columns)
    cleaned_candidate_columns = tuple(cleaned_candidate_columns)
    subset_map: dict[str, tuple[str, ...]] = {}

    if core_columns:
        subset_map["core_only"] = core_columns

    technical_columns = tuple(
        column for column in cleaned_candidate_columns if column.startswith("price_technical__")
    )
    if technical_columns:
        if core_columns:
            subset_map[f"core_plus_all_technical_total_{len(core_columns) + len(technical_columns)}"] = (
                core_columns + technical_columns
            )
        else:
            subset_map[f"all_technical_total_{len(technical_columns)}"] = technical_columns

    for total_size in sorted({int(size) for size in total_feature_sizes if int(size) > 0}):
        extra_budget = max(total_size - len(core_columns), 0)
        subset_name = f"top_pruned_total_{min(total_size, len(core_columns) + len(kept_extra_columns))}"
        if core_columns:
            subset_map[subset_name] = core_columns + kept_extra_columns[:extra_budget]
        else:
            subset_map[subset_name] = kept_extra_columns[:total_size]

    unique_subsets: dict[tuple[str, ...], str] = {}
    deduped: dict[str, tuple[str, ...]] = {}
    for subset_name, columns in subset_map.items():
        key = tuple(columns)
        if not key:
            continue
        if key in unique_subsets:
            continue
        unique_subsets[key] = subset_name
        deduped[subset_name] = key
    return deduped


def build_forward_subset_candidates(
    *,
    core_columns: Iterable[str],
    selected_extra_columns: Iterable[str],
    total_feature_sizes: Iterable[int],
) -> dict[str, tuple[str, ...]]:
    core_columns = tuple(core_columns)
    selected_extra_columns = tuple(selected_extra_columns)
    subset_map: dict[str, tuple[str, ...]] = {}

    if core_columns:
        subset_map["core_only"] = core_columns

    for total_size in sorted({int(size) for size in total_feature_sizes if int(size) > 0}):
        capped_total = min(max(total_size, len(core_columns)), len(core_columns) + len(selected_extra_columns))
        extra_budget = max(capped_total - len(core_columns), 0)
        if extra_budget <= 0:
            continue
        subset_map[f"forward_total_{capped_total}"] = core_columns + selected_extra_columns[:extra_budget]

    unique_subsets: dict[tuple[str, ...], str] = {}
    deduped: dict[str, tuple[str, ...]] = {}
    for subset_name, columns in subset_map.items():
        key = tuple(columns)
        if not key:
            continue
        if key in unique_subsets:
            continue
        unique_subsets[key] = subset_name
        deduped[subset_name] = key
    return deduped


def evaluate_named_subsets(
    features: pd.DataFrame,
    target_return: pd.Series,
    *,
    subset_map: dict[str, tuple[str, ...]],
    folds,
    model_spec: SelectionModelSpec,
    core_columns: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_set = set(core_columns)
    summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for subset_name, columns in subset_map.items():
        result = evaluate_feature_subset(
            features,
            target_return,
            columns=columns,
            folds=folds,
            model_spec=model_spec,
        )
        extra_columns = [column for column in columns if column not in core_set]
        summary_rows.append(
            {
                "subset_name": subset_name,
                "feature_count": float(len(columns)),
                "extra_feature_count": float(len(extra_columns)),
                "technical_feature_count": float(sum(column.startswith("price_technical__") for column in columns)),
                "parser_feature_count": float(sum(column.startswith("headline_parser__") for column in columns)),
                "mean_cv_sharpe": result.mean_cv_sharpe,
                "std_cv_sharpe": result.std_cv_sharpe,
                "oof_sharpe": result.oof_sharpe,
            }
        )
        for order, column in enumerate(columns, start=1):
            feature_rows.append(
                {
                    "subset_name": subset_name,
                    "feature_order": float(order),
                    "feature": column,
                    "block": column.split("__", 1)[0],
                    "feature_name": column.split("__", 1)[1] if "__" in column else column,
                    "is_core": float(column in core_set),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["oof_sharpe", "mean_cv_sharpe", "feature_count"],
        ascending=[False, False, True],
    )
    summary.insert(0, "subset_rank", np.arange(1, len(summary) + 1, dtype=int))
    return summary.reset_index(drop=True), pd.DataFrame(feature_rows)
