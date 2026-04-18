import pandas as pd


def build_train_targets(train_seen_bars: pd.DataFrame, train_unseen_bars: pd.DataFrame) -> pd.DataFrame:
    halfway = (
        train_seen_bars.groupby("session", sort=True)["close"]
        .last()
        .rename("close_halfway")
        .to_frame()
    )
    end = (
        train_unseen_bars.groupby("session", sort=True)["close"]
        .last()
        .rename("close_end")
        .to_frame()
    )
    targets = halfway.join(end, how="inner")
    targets["target_return"] = targets["close_end"] / targets["close_halfway"] - 1.0
    return targets
