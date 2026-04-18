import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.experiments.catalog import format_full_catalog
from src.experiments.config import load_experiment_config
from src.experiments.runner import cross_validate_experiment
from src.paths import REPORTS_DIR, ROOT, ensure_output_dirs


def _resolve_paths(config_globs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for config_glob in config_globs:
        matches = sorted(ROOT.glob(config_glob))
        if not matches:
            raise FileNotFoundError(f"No configs matched pattern: {config_glob}")
        for path in matches:
            if path in seen:
                continue
            seen.add(path)
            resolved.append(path)
    return resolved


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _save_report(frame: pd.DataFrame, output_path: Path, latest_alias: str | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    if latest_alias:
        frame.to_csv(output_path.parent / latest_alias, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run K-fold CV across many experiment configs.")
    parser.add_argument(
        "--config-glob",
        action="append",
        default=None,
        help="Glob pattern relative to the repo root. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path for the leaderboard output.",
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=None,
        help="CSV path for the per-fold output.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many leaderboard rows to print.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first config error instead of continuing.",
    )
    parser.add_argument(
        "--list-catalog",
        action="store_true",
        help="Print the supported feature blocks and models and exit.",
    )
    args = parser.parse_args()
    if not args.list_catalog and not args.config_glob:
        parser.error("At least one --config-glob is required unless --list-catalog is used.")
    return args


def main():
    args = parse_args()
    if args.list_catalog:
        print(format_full_catalog())
        return

    ensure_output_dirs()
    config_paths = _resolve_paths(args.config_glob)

    summary_rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    for config_path in config_paths:
        display_path = _display_path(config_path)
        try:
            config = load_experiment_config(config_path)
            started_at = time.time()
            training = cross_validate_experiment(config)
            elapsed = time.time() - started_at
        except Exception as exc:
            if args.fail_fast:
                raise
            summary_rows.append(
                {
                    "experiment_name": config_path.stem,
                    "config_path": display_path,
                    "model": "",
                    "feature_blocks": "",
                    "mean_cv_sharpe": float("nan"),
                    "std_cv_sharpe": float("nan"),
                    "mean_feature_count": float("nan"),
                    "cv_folds": float("nan"),
                    "elapsed_seconds": float("nan"),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"ERROR  {display_path}: {exc}")
            continue

        feature_block_names = ",".join(spec.resolved_alias for spec in config.feature_blocks)
        mean_cv_sharpe = float(training.fold_summary["sharpe"].mean())
        std_cv_sharpe = float(training.fold_summary["sharpe"].std(ddof=0))
        mean_feature_count = float(training.fold_summary["feature_count"].mean())

        summary_rows.append(
            {
                "experiment_name": config.experiment_name,
                "config_path": display_path,
                "model": config.model.name,
                "feature_blocks": feature_block_names,
                "mean_cv_sharpe": mean_cv_sharpe,
                "std_cv_sharpe": std_cv_sharpe,
                "mean_feature_count": mean_feature_count,
                "cv_folds": float(config.cv_folds),
                "elapsed_seconds": elapsed,
                "status": "ok",
                "error": "",
            }
        )

        fold_frame = training.fold_summary.copy()
        fold_frame.insert(0, "experiment_name", config.experiment_name)
        fold_frame.insert(1, "config_path", display_path)
        fold_frame.insert(2, "model", config.model.name)
        fold_frame.insert(3, "feature_blocks", feature_block_names)
        fold_frames.append(fold_frame)

        print(
            f"OK     {display_path}: mean={mean_cv_sharpe:.4f} std={std_cv_sharpe:.4f} "
            f"features={mean_feature_count:.0f} time={elapsed:.1f}s"
        )

    leaderboard = pd.DataFrame(summary_rows)
    success_mask = leaderboard["status"].eq("ok") if not leaderboard.empty else pd.Series(dtype=bool)
    success_rows = leaderboard.loc[success_mask].sort_values("mean_cv_sharpe", ascending=False)
    error_rows = leaderboard.loc[~success_mask]
    ordered_leaderboard = pd.concat([success_rows, error_rows], ignore_index=True)

    if args.output is None:
        args.output = REPORTS_DIR / f"{timestamp}_experiment_sweep.csv"
    saved_leaderboard_path = _save_report(ordered_leaderboard, args.output, latest_alias="latest_experiment_sweep.csv")

    if fold_frames:
        all_folds = pd.concat(fold_frames, ignore_index=True)
        fold_output = args.fold_output or REPORTS_DIR / f"{timestamp}_experiment_sweep_folds.csv"
        saved_folds_path = _save_report(all_folds, fold_output, latest_alias="latest_experiment_sweep_folds.csv")
    else:
        saved_folds_path = None

    print("\nEXPERIMENT LEADERBOARD")
    if ordered_leaderboard.empty:
        print("No configs were evaluated.")
    else:
        print(ordered_leaderboard.head(max(args.top, 0)).to_string(index=False))
    print(f"\nSaved leaderboard to {saved_leaderboard_path}")
    if saved_folds_path is not None:
        print(f"Saved per-fold results to {saved_folds_path}")


if __name__ == "__main__":
    main()
