import pandas as pd

from src.paths import DATA_DIR

VALID_SPLITS = {"train", "public_test", "private_test"}
VALID_VISIBILITY = {"seen", "unseen"}


def _dataset_path(data_kind: str, visibility: str, split: str):
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if visibility not in VALID_VISIBILITY:
        raise ValueError(f"Unknown visibility: {visibility}")
    if split != "train" and visibility == "unseen":
        raise ValueError("Unseen data only exists for the training split.")
    return DATA_DIR / f"{data_kind}_{visibility}_{split}.parquet"


def load_bars(split: str, visibility: str = "seen") -> pd.DataFrame:
    return pd.read_parquet(_dataset_path("bars", visibility, split))


def load_headlines(split: str, visibility: str = "seen") -> pd.DataFrame:
    return pd.read_parquet(_dataset_path("headlines", visibility, split))


def load_train_bars():
    return load_bars("train", "seen"), load_bars("train", "unseen")


def load_train_headlines():
    return load_headlines("train", "seen"), load_headlines("train", "unseen")
