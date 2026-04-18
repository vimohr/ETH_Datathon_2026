import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.paths import ROOT, SUBMISSIONS_DIR


def build_submission(sessions, target_position) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "session": pd.Index(sessions, name="session").astype(int),
            "target_position": pd.Series(target_position, index=sessions).astype(float).to_numpy(),
        }
    )
    return submission.sort_values("session").reset_index(drop=True)


def validate_submission(submission: pd.DataFrame, expected_sessions=None) -> pd.DataFrame:
    required_columns = {"session", "target_position"}
    missing = required_columns - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing required columns: {sorted(missing)}")

    if submission["session"].duplicated().any():
        raise ValueError("Submission contains duplicate session ids.")

    if expected_sessions is not None:
        expected_index = pd.Index(expected_sessions).astype(int).sort_values()
        submitted_index = pd.Index(submission["session"]).astype(int).sort_values()
        if not submitted_index.equals(expected_index):
            raise ValueError("Submission sessions do not match the expected test sessions.")

    return submission


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_submission_metadata(
    *,
    model_name: str,
    test_split: str,
    feature_count: int,
    mean_cv_sharpe: float | None = None,
    include_headlines: bool = False,
    notes: str | None = None,
) -> dict:
    metadata = {
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": current_git_commit(),
        "model_name": model_name,
        "test_split": test_split,
        "feature_count": feature_count,
        "include_headlines": include_headlines,
    }
    if mean_cv_sharpe is not None:
        metadata["mean_cv_sharpe"] = float(mean_cv_sharpe)
    if notes:
        metadata["notes"] = notes
    return metadata


def append_submission_registry(output_path: Path, metadata: dict) -> Path:
    registry_path = SUBMISSIONS_DIR / "registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()
    record = {"submission_path": str(output_path.relative_to(ROOT)), **metadata}
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return registry_path


def save_submission(
    submission: pd.DataFrame,
    output_path,
    expected_sessions=None,
    metadata: dict | None = None,
    latest_alias: str | None = None,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission(submission, expected_sessions=expected_sessions)
    submission.to_csv(output_path, index=False)

    if latest_alias:
        latest_path = output_path.parent / latest_alias
        submission.to_csv(latest_path, index=False)

    if metadata:
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_submission_registry(output_path, metadata)
        if latest_alias:
            latest_metadata_path = (output_path.parent / latest_alias).with_suffix(".json")
            latest_metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    return output_path
