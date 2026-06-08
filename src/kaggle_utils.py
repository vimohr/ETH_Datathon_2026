import os
from contextlib import contextmanager
from pathlib import Path

from src.paths import ROOT

DEFAULT_COMPETITION = "hrt-eth-zurich-datathon-2026"
KAGGLE_ENV_KEYS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR")


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
    return env


def load_kaggle_environment(env_file: str | Path | None = None) -> tuple[dict[str, str], list[Path]]:
    loaded_files: list[Path] = []
    resolved_env: dict[str, str] = {}

    if env_file is None:
        candidate_files = [ROOT.parent / ".env", ROOT / ".env"]
    else:
        candidate = Path(env_file)
        candidate_files = [candidate if candidate.is_absolute() else ROOT / candidate]

    for candidate in candidate_files:
        if candidate.exists():
            resolved_env.update(_parse_env_file(candidate))
            loaded_files.append(candidate)

    for key in KAGGLE_ENV_KEYS:
        if os.environ.get(key):
            resolved_env[key] = os.environ[key]

    if resolved_env.get("KAGGLE_API_TOKEN") and not resolved_env.get("KAGGLE_KEY"):
        resolved_env["KAGGLE_KEY"] = resolved_env["KAGGLE_API_TOKEN"]

    missing = [key for key in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not resolved_env.get(key)]
    if missing:
        searched = ", ".join(str(path) for path in candidate_files)
        raise ValueError(
            "Missing Kaggle credentials: "
            + ", ".join(missing)
            + f". Checked environment and .env files at: {searched}"
        )

    return {key: value for key, value in resolved_env.items() if key in KAGGLE_ENV_KEYS and value}, loaded_files


@contextmanager
def authenticated_kaggle_api(env_file: str | Path | None = None):
    kaggle_env, loaded_files = load_kaggle_environment(env_file=env_file)
    previous_values = {key: os.environ.get(key) for key in kaggle_env}
    try:
        os.environ.update(kaggle_env)
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        yield api, loaded_files
    finally:
        for key, old_value in previous_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def submit_competition_file(
    file_path,
    *,
    message: str,
    competition: str = DEFAULT_COMPETITION,
    env_file: str | Path | None = None,
) -> tuple[str, list[Path]]:
    resolved_path = Path(file_path)
    if not resolved_path.is_absolute():
        resolved_path = ROOT / resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {resolved_path}")

    with authenticated_kaggle_api(env_file=env_file) as (api, loaded_files):
        response = api.competition_submit(str(resolved_path), message, competition)
    return getattr(response, "message", str(response)), loaded_files


def list_competition_submissions(
    *,
    competition: str = DEFAULT_COMPETITION,
    env_file: str | Path | None = None,
    page_size: int = 20,
) -> tuple[list[dict[str, str]], list[Path]]:
    with authenticated_kaggle_api(env_file=env_file) as (api, loaded_files):
        submissions = api.competition_submissions(competition, page_size=page_size) or []

    rows: list[dict[str, str]] = []
    for submission in submissions:
        rows.append(
            {
                "ref": str(getattr(submission, "ref", "")),
                "file_name": getattr(submission, "file_name", ""),
                "date": str(getattr(submission, "date", "")),
                "description": getattr(submission, "description", ""),
                "status": str(getattr(submission, "status", "")),
                "public_score": getattr(submission, "public_score", ""),
                "private_score": getattr(submission, "private_score", ""),
                "error_description": getattr(submission, "error_description", ""),
            }
        )
    return rows, loaded_files
