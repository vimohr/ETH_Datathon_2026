import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.kaggle_utils import DEFAULT_COMPETITION, list_competition_submissions, submit_competition_file
from src.paths import SUBMISSIONS_DIR, ensure_output_dirs
from src.submission import (
    build_submission_metadata,
    combine_submission_files,
    expected_competition_sessions,
    save_submission,
)


def combine_command(args) -> Path:
    ensure_output_dirs()

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = SUBMISSIONS_DIR / f"{timestamp}_{args.model_name}_competition.csv"

    combined_submission = combine_submission_files(args.public, args.private)
    metadata = build_submission_metadata(
        model_name=args.model_name,
        test_split="competition",
        notes=args.notes,
    )
    metadata.update(
        {
            "competition_name": args.competition,
            "source_public_submission": str(Path(args.public)),
            "source_private_submission": str(Path(args.private)),
            "row_count": int(len(combined_submission)),
        }
    )
    saved_path = save_submission(
        combined_submission,
        output_path,
        expected_sessions=expected_competition_sessions(),
        metadata=metadata,
        latest_alias="latest_competition.csv",
    )
    print(f"Saved combined competition submission to {saved_path}")
    return saved_path


def submit_command(args) -> None:
    message, loaded_files = submit_competition_file(
        args.file,
        message=args.message,
        competition=args.competition,
        env_file=args.env_file,
    )
    if loaded_files:
        print("Loaded Kaggle credentials from:")
        for path in loaded_files:
            print(f"- {path}")
    print(message)


def status_command(args) -> None:
    rows, loaded_files = list_competition_submissions(
        competition=args.competition,
        env_file=args.env_file,
        page_size=args.limit,
    )
    if loaded_files:
        print("Loaded Kaggle credentials from:")
        for path in loaded_files:
            print(f"- {path}")
    if not rows:
        print("No submissions found.")
        return

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition submission utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    combine_parser = subparsers.add_parser("combine", help="Combine public and private split submissions.")
    combine_parser.add_argument(
        "--public",
        type=Path,
        default=SUBMISSIONS_DIR / "latest_public_test.csv",
        help="Path to the public-test submission CSV.",
    )
    combine_parser.add_argument(
        "--private",
        type=Path,
        default=SUBMISSIONS_DIR / "latest_private_test.csv",
        help="Path to the private-test submission CSV.",
    )
    combine_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the combined competition submission CSV.",
    )
    combine_parser.add_argument(
        "--model-name",
        default="competition_submission",
        help="Model name to store in metadata for the combined file.",
    )
    combine_parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Competition slug for metadata purposes.",
    )
    combine_parser.add_argument(
        "--notes",
        default=None,
        help="Optional notes to store in the combined submission metadata.",
    )
    combine_parser.set_defaults(func=combine_command)

    submit_parser = subparsers.add_parser("submit", help="Submit a competition file to Kaggle.")
    submit_parser.add_argument(
        "--file",
        type=Path,
        default=SUBMISSIONS_DIR / "latest_competition.csv",
        help="Path to the combined competition submission CSV.",
    )
    submit_parser.add_argument(
        "--message",
        required=True,
        help="Kaggle submission message.",
    )
    submit_parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Competition slug.",
    )
    submit_parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load Kaggle credentials from.",
    )
    submit_parser.set_defaults(func=submit_command)

    status_parser = subparsers.add_parser("status", help="List recent Kaggle submissions with error details.")
    status_parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Competition slug.",
    )
    status_parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load Kaggle credentials from.",
    )
    status_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent submissions to show.",
    )
    status_parser.set_defaults(func=status_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
