import argparse
from pathlib import Path

from src.experiments.config import ExperimentConfig, FeatureSpec, ModelSpec
from src.experiments.runner import run_experiment_pipeline


def build_baseline_config(include_headlines: bool, cv_repeats: int = 1) -> ExperimentConfig:
    feature_blocks = [FeatureSpec(name="price")]
    experiment_name = "baseline_price"

    if include_headlines:
        feature_blocks.append(FeatureSpec(name="headline_parser"))
        experiment_name = "baseline_price_headlines"

    return ExperimentConfig(
        experiment_name=experiment_name,
        feature_blocks=tuple(feature_blocks),
        model=ModelSpec(name="linear"),
        cv_repeats=int(cv_repeats),
    )


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
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=1,
        help="How many random-seed CV repeats to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_baseline_config(include_headlines=args.include_headlines, cv_repeats=args.cv_repeats)
    run_experiment_pipeline(
        config,
        test_split=args.test_split,
        output_path=args.output,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
