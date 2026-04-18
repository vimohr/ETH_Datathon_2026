import argparse
from pathlib import Path

from src.experiments.config import ExperimentConfig, FeatureSpec, ModelSpec
from src.experiments.runner import run_experiment_pipeline


def build_text_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="text_only",
        feature_blocks=(FeatureSpec(name="headline_parser"),),
        model=ModelSpec(name="linear"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the text-only baseline model and create a submission.")
    parser.add_argument(
        "--test-split",
        choices=["public_test", "private_test"],
        default="public_test",
        help="Which test split to score and export.",
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
    config = build_text_config()
    run_experiment_pipeline(
        config,
        test_split=args.test_split,
        output_path=args.output,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
