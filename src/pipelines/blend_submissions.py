import argparse
import itertools
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import pnl, sharpe_from_positions
from src.models.uncertainty import size_positions
from src.paths import OOF_DIR, SUBMISSIONS_DIR, ensure_output_dirs
from src.settings import POSITION_CLIP, POSITION_SCORE_PERCENTILE
from src.submission import build_submission, build_submission_metadata, save_submission


@dataclass
class BlendComponent:
    name: str
    oof_path: Path
    submission_path: Path


def _parse_component(spec: str) -> BlendComponent:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "Each --component must be 'name,oof_path,submission_path'. "
            f"Received: {spec!r}"
        )
    name, oof_path, submission_path = parts
    return BlendComponent(name=name, oof_path=Path(oof_path), submission_path=Path(submission_path))


def _parse_float_list(value: str) -> list[float]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    return [float(item) for item in values]


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return weights / total


def _weight_grid(n_components: int, step: float) -> list[np.ndarray]:
    if n_components < 1:
        return []
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("weight_step must divide 1.0 cleanly, e.g. 0.05 or 0.1.")

    if n_components == 1:
        return [np.array([1.0], dtype=float)]

    grids: list[np.ndarray] = []
    for combo in itertools.product(range(units + 1), repeat=n_components):
        if sum(combo) != units:
            continue
        grids.append(np.asarray(combo, dtype=float) / units)
    return grids


def _load_oof_frame(component: BlendComponent) -> pd.DataFrame:
    frame = pd.read_csv(component.oof_path).copy()
    required = {"session", "target_position", "realized_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF file {component.oof_path} is missing columns: {sorted(missing)}")
    return frame.loc[:, ["session", "target_position", "realized_return"]].rename(
        columns={"target_position": component.name}
    )


def _load_submission_frame(component: BlendComponent) -> pd.DataFrame:
    frame = pd.read_csv(component.submission_path).copy()
    required = {"session", "target_position"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Submission file {component.submission_path} is missing columns: {sorted(missing)}")
    return frame.loc[:, ["session", "target_position"]].rename(columns={"target_position": component.name})


def _merge_oof_components(components: list[BlendComponent]) -> pd.DataFrame:
    merged = None
    for component in components:
        frame = _load_oof_frame(component)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="session", how="inner", suffixes=("", f"__{component.name}"))
            duplicate_column = f"realized_return__{component.name}"
            if duplicate_column in merged.columns:
                same_returns = np.allclose(
                    merged["realized_return"].to_numpy(dtype=float),
                    merged[duplicate_column].to_numpy(dtype=float),
                    atol=1e-12,
                    equal_nan=True,
                )
                if not same_returns:
                    raise ValueError(
                        f"Realized returns do not match between OOF files when merging component {component.name}."
                    )
                merged = merged.drop(columns=[duplicate_column])

    if merged is None:
        raise ValueError("No components were provided.")
    return merged.sort_values("session").reset_index(drop=True)


def _merge_submission_components(components: list[BlendComponent]) -> pd.DataFrame:
    merged = None
    for component in components:
        frame = _load_submission_frame(component)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="session", how="inner")

    if merged is None:
        raise ValueError("No components were provided.")
    return merged.sort_values("session").reset_index(drop=True)


def _blend_positions(
    position_matrix: pd.DataFrame,
    *,
    weights: np.ndarray,
    neutral_band: float,
    dispersion_gamma: float,
    max_abs_position: float = POSITION_CLIP,
    percentile: float = POSITION_SCORE_PERCENTILE,
) -> pd.Series:
    aligned_weights = _normalize_weights(np.asarray(weights, dtype=float))
    score = position_matrix.to_numpy(dtype=float) @ aligned_weights
    score_series = pd.Series(score, index=position_matrix.index, name="blend_score")

    uncertainty = None
    if dispersion_gamma > 0.0 and position_matrix.shape[1] > 1:
        dispersion = position_matrix.std(axis=1, ddof=0)
        uncertainty = 1.0 + dispersion_gamma * dispersion / max_abs_position

    positions = size_positions(
        score_series,
        uncertainty=uncertainty,
        max_abs_position=max_abs_position,
        neutral_band=neutral_band,
        percentile=percentile,
    )
    return positions


def _evaluate_candidate(
    position_matrix: pd.DataFrame,
    realized_return: pd.Series,
    *,
    weights: np.ndarray,
    neutral_band: float,
    dispersion_gamma: float,
    max_abs_position: float,
    percentile: float,
) -> dict[str, object]:
    target_position = _blend_positions(
        position_matrix,
        weights=weights,
        neutral_band=neutral_band,
        dispersion_gamma=dispersion_gamma,
        max_abs_position=max_abs_position,
        percentile=percentile,
    )
    return {
        "weights": np.asarray(weights, dtype=float),
        "neutral_band": float(neutral_band),
        "dispersion_gamma": float(dispersion_gamma),
        "sharpe": sharpe_from_positions(target_position, realized_return),
        "target_position": target_position,
    }


def _search_candidates(
    position_matrix: pd.DataFrame,
    realized_return: pd.Series,
    *,
    candidate_weights: list[np.ndarray],
    neutral_bands: list[float],
    dispersion_gammas: list[float],
    max_abs_position: float,
    percentile: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for weights in candidate_weights:
        for neutral_band in neutral_bands:
            for dispersion_gamma in dispersion_gammas:
                candidate = _evaluate_candidate(
                    position_matrix,
                    realized_return,
                    weights=weights,
                    neutral_band=neutral_band,
                    dispersion_gamma=dispersion_gamma,
                    max_abs_position=max_abs_position,
                    percentile=percentile,
                )
                rows.append(
                    {
                        "weights": json.dumps([round(float(weight), 4) for weight in candidate["weights"]]),
                        "neutral_band": float(neutral_band),
                        "dispersion_gamma": float(dispersion_gamma),
                        "sharpe": float(candidate["sharpe"]),
                    }
                )
                if best is None or candidate["sharpe"] > best["sharpe"]:
                    best = candidate

    if best is None:
        raise ValueError("No blend candidates were evaluated.")

    leaderboard = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return best, leaderboard


def _format_weight_map(components: list[BlendComponent], weights: np.ndarray) -> dict[str, float]:
    return {component.name: round(float(weight), 6) for component, weight in zip(components, weights, strict=True)}


def run_pipeline(
    *,
    components: list[BlendComponent],
    test_split: str,
    output_path: Path | None,
    weights: list[float] | None,
    search_weights: bool,
    weight_step: float,
    neutral_bands: list[float],
    dispersion_gammas: list[float],
    max_abs_position: float,
    percentile: float,
    cv_only: bool,
) -> pd.DataFrame | None:
    ensure_output_dirs()

    merged_oof = _merge_oof_components(components)
    component_names = [component.name for component in components]
    position_matrix = merged_oof.set_index("session")[component_names]
    realized_return = merged_oof.set_index("session")["realized_return"]

    if search_weights:
        candidate_weights = _weight_grid(len(components), step=weight_step)
    else:
        if weights is None:
            candidate_weights = [np.full(len(components), 1.0 / len(components), dtype=float)]
        else:
            if len(weights) != len(components):
                raise ValueError("Number of --weights values must match the number of components.")
            candidate_weights = [_normalize_weights(np.asarray(weights, dtype=float))]

    best, leaderboard = _search_candidates(
        position_matrix,
        realized_return,
        candidate_weights=candidate_weights,
        neutral_bands=neutral_bands,
        dispersion_gammas=dispersion_gammas,
        max_abs_position=max_abs_position,
        percentile=percentile,
    )
    print(leaderboard.head(10).to_string(index=False))
    print(f"Best CV Sharpe: {best['sharpe']:.4f}")
    print(f"Best weights: {_format_weight_map(components, best['weights'])}")
    print(f"Best neutral_band: {best['neutral_band']}")
    print(f"Best dispersion_gamma: {best['dispersion_gamma']}")

    model_slug = "_".join(component.name for component in components)
    oof_name = f"blend_{model_slug}_oof.csv"
    oof_output = pd.DataFrame(
        {
            "predicted_return": best["target_position"],
            "predicted_uncertainty": 1.0,
            "target_position": best["target_position"],
            "realized_return": realized_return,
            "pnl": pnl(best["target_position"], realized_return),
        }
    )
    oof_path = OOF_DIR / oof_name
    oof_output.to_csv(oof_path, index_label="session")
    print(f"Saved blended OOF predictions to {oof_path}")

    if cv_only:
        return None

    merged_submission = _merge_submission_components(components)
    submission_matrix = merged_submission.set_index("session")[component_names]
    target_position = _blend_positions(
        submission_matrix,
        weights=best["weights"],
        neutral_band=best["neutral_band"],
        dispersion_gamma=best["dispersion_gamma"],
        max_abs_position=max_abs_position,
        percentile=percentile,
    )
    submission = build_submission(submission_matrix.index, target_position)

    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = SUBMISSIONS_DIR / f"{timestamp}_blend_{model_slug}_{test_split}.csv"

    metadata = build_submission_metadata(
        model_name=f"blend_{model_slug}",
        test_split=test_split,
        feature_count=int(len(components)),
        mean_cv_sharpe=float(best["sharpe"]),
        include_headlines=True,
        notes=json.dumps(
            {
                "components": [component.name for component in components],
                "weights": _format_weight_map(components, best["weights"]),
                "neutral_band": float(best["neutral_band"]),
                "dispersion_gamma": float(best["dispersion_gamma"]),
                "weight_step": float(weight_step),
                "search_weights": bool(search_weights),
            },
            sort_keys=True,
        ),
    )
    saved_path = save_submission(
        submission,
        output_path,
        expected_sessions=submission_matrix.index.to_numpy(),
        metadata=metadata,
        latest_alias=f"latest_{test_split}.csv",
    )
    print(f"Saved blended submission to {saved_path}")
    return submission


def parse_args():
    parser = argparse.ArgumentParser(description="Blend model OOFs and submissions into a new submission.")
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="Blend component in the form 'name,oof_path,submission_path'. Repeat once per model.",
    )
    parser.add_argument(
        "--test-split",
        choices=["public_test", "private_test"],
        default="public_test",
        help="Which test split the component submissions correspond to.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submission output path. Defaults to outputs/submissions/...",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated blend weights. If omitted and --search-weights is not set, use equal weights.",
    )
    parser.add_argument(
        "--search-weights",
        action="store_true",
        help="Grid-search nonnegative weights that sum to 1. Recommended for 2-3 components.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.1,
        help="Weight grid step when --search-weights is enabled.",
    )
    parser.add_argument(
        "--neutral-bands",
        type=str,
        default="0",
        help="Comma-separated neutral-band candidates in target-position units, e.g. '0,5,10'.",
    )
    parser.add_argument(
        "--dispersion-gammas",
        type=str,
        default="0",
        help="Comma-separated disagreement shrinkage candidates, e.g. '0,0.5,1.0'.",
    )
    parser.add_argument(
        "--max-abs-position",
        type=float,
        default=POSITION_CLIP,
        help="Final position clip for the blended submission.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=POSITION_SCORE_PERCENTILE,
        help="Percentile used when rescaling blended positions.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run OOF blending only and skip test submission export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    components = [_parse_component(spec) for spec in args.component]
    weights = _parse_float_list(args.weights) if args.weights is not None else None
    neutral_bands = _parse_float_list(args.neutral_bands)
    dispersion_gammas = _parse_float_list(args.dispersion_gammas)
    run_pipeline(
        components=components,
        test_split=args.test_split,
        output_path=args.output,
        weights=weights,
        search_weights=args.search_weights,
        weight_step=args.weight_step,
        neutral_bands=neutral_bands,
        dispersion_gammas=dispersion_gammas,
        max_abs_position=args.max_abs_position,
        percentile=args.percentile,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
