from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NOTEBOOKS_DIR = ROOT / "notebooks"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
OOF_DIR = OUTPUTS_DIR / "oof"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DOCS_DIR = ROOT / "docs"


def ensure_output_dirs() -> None:
    for path in (OUTPUTS_DIR, MODELS_DIR, OOF_DIR, FIGURES_DIR, SUBMISSIONS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
