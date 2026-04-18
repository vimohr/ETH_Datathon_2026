import argparse
from pathlib import Path

from src.experiments.catalog import FEATURE_BLOCK_CATALOG, MODEL_CATALOG, format_catalog
from src.experiments.config import load_experiment_config
from src.experiments.runner import run_experiment_competition_pipeline, run_experiment_pipeline
from src.kaggle_utils import DEFAULT_COMPETITION


def parse_args():
    parser = argparse.ArgumentParser(description="Train a configurable experiment and create submissions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the JSON experiment config.",
    )
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
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Generate both split submissions plus a combined competition file.",
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
        "--submit-kaggle",
        action="store_true",
        help="Submit the combined competition file to Kaggle after generating it.",
    )
    parser.add_argument(
        "--submission-message",
        default=None,
        help="Submission message to use when --submit-kaggle is enabled.",
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
    args = parser.parse_args()
    if args.cv_only and args.competition:
        parser.error("--cv-only cannot be combined with --competition.")
    if args.submit_kaggle and not args.competition:
        parser.error("--submit-kaggle requires --competition.")
    if args.submit_kaggle and not args.submission_message:
        parser.error("--submission-message is required when --submit-kaggle is used.")
    if not (args.list_features or args.list_models) and args.config is None:
        parser.error("--config is required unless a list option is used.")
    return args


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

    config = load_experiment_config(args.config)
    if args.competition:
        run_experiment_competition_pipeline(
            config,
            public_output_path=args.public_output,
            private_output_path=args.private_output,
            competition_output_path=args.competition_output,
            competition=args.competition_name,
            submit_kaggle=args.submit_kaggle,
            submission_message=args.submission_message,
            env_file=args.env_file,
        )
        return

    run_experiment_pipeline(
        config,
        test_split=args.test_split,
        output_path=args.output,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
