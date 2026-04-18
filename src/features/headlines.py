import re

import pandas as pd

KEYWORD_PATTERNS = {
    "headline_positive_count": r"\b(secures|wins|approval|upgrade|launches|record quarterly revenue|increase)\b",
    "headline_negative_count": r"\b(delays|withdraws|steps down|drop|loses key contract|unfavorable)\b",
    "headline_event_count": r"\b(contract|maintenance|award|alternatives|investor|sustainability|open letter)\b",
}


def _entity_prefix(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]+", text)
    return " ".join(tokens[:2]).lower()


def build_headline_features(headlines: pd.DataFrame, sessions=None) -> pd.DataFrame:
    if headlines.empty:
        index = pd.Index(sorted(pd.Index(sessions).unique())) if sessions is not None else pd.Index([])
        return pd.DataFrame(index=index)

    df = headlines.copy()
    df["headline"] = df["headline"].fillna("")
    grouped = df.groupby("session", sort=True)
    features = pd.DataFrame(index=sorted(df["session"].unique()))

    features["headline_count"] = grouped.size()
    features["headline_bar_count"] = grouped["bar_ix"].nunique()
    features["headline_char_mean"] = grouped["headline"].apply(lambda series: series.str.len().mean())
    features["headline_char_max"] = grouped["headline"].apply(lambda series: series.str.len().max())
    features["headline_entity_prefix_count"] = grouped["headline"].apply(
        lambda series: series.map(_entity_prefix).nunique()
    )

    recent_threshold = int(df["bar_ix"].max()) - 9
    recent_counts = df.loc[df["bar_ix"] >= recent_threshold].groupby("session").size()
    features["recent_headline_count"] = recent_counts

    headline_text = df["headline"].str.lower()
    for feature_name, pattern in KEYWORD_PATTERNS.items():
        keyword_counts = headline_text.str.contains(pattern, regex=True).groupby(df["session"]).sum()
        features[feature_name] = keyword_counts

    if sessions is not None:
        features = features.reindex(sorted(pd.Index(sessions).unique()))

    return features.fillna(0.0).sort_index()
