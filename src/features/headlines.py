import numpy as np
import pandas as pd

from src.features.headline_parser import parse_headlines


def _empty_feature_frame(sessions=None) -> pd.DataFrame:
    index = pd.Index(sorted(pd.Index(sessions).unique())) if sessions is not None else pd.Index([])
    return pd.DataFrame(index=index)


def build_headline_event_table(headlines: pd.DataFrame) -> pd.DataFrame:
    if headlines.empty:
        return headlines.copy()

    events = parse_headlines(headlines)
    events["polarity_abs"] = events["polarity"].abs()
    events["positive_flag"] = (events["polarity"] > 0).astype(float)
    events["negative_flag"] = (events["polarity"] < 0).astype(float)
    events["neutral_flag"] = (events["polarity"] == 0).astype(float)
    events["pct_abs"] = events["pct"].abs()
    events["pct_signed"] = events["pct"] * events["polarity"]
    session_max_bar = events.groupby("session")["bar_ix"].transform("max")
    events["is_recent"] = (events["bar_ix"] >= (session_max_bar - 9)).astype(float)
    return events


def _count_by_category(
    df: pd.DataFrame,
    *,
    category_col: str,
    prefix: str,
    index_cols: list[str],
    min_total_count: int = 1,
    drop_values: set[str] | None = None,
    drop_suffixes: tuple[str, ...] = (),
) -> pd.DataFrame:
    values = df[category_col].fillna("none").astype(str)
    if drop_values:
        values = values.where(~values.isin(drop_values), "__drop__")
    if drop_suffixes:
        values = values.where(~values.str.endswith(drop_suffixes), "__drop__")

    category_counts = values.loc[values != "__drop__"].value_counts()
    keep_values = set(category_counts.loc[category_counts >= min_total_count].index)
    values = values.where(values.isin(keep_values), "__drop__")

    count_frame = df.loc[values != "__drop__", index_cols].copy()
    count_frame[category_col] = values.loc[values != "__drop__"].to_numpy()

    if count_frame.empty:
        return pd.DataFrame()

    counts = (
        count_frame.groupby(index_cols + [category_col], sort=True)
        .size()
        .unstack(fill_value=0.0)
    )
    counts.columns = [f"{prefix}{str(column).lower().replace(' ', '_')}" for column in counts.columns]
    return counts.astype(float)


def build_company_session_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    grouped = events.groupby(["session", "company"], sort=True)
    features = grouped.agg(
        headline_count=("headline", "size"),
        recent_headline_count=("is_recent", "sum"),
        bar_coverage=("bar_ix", "nunique"),
        last_bar_ix=("bar_ix", "max"),
        mean_bar_ix=("bar_ix", "mean"),
        positive_count=("positive_flag", "sum"),
        negative_count=("negative_flag", "sum"),
        neutral_count=("neutral_flag", "sum"),
        polarity_sum=("polarity", "sum"),
        polarity_abs_sum=("polarity_abs", "sum"),
        amount_sum_m=("amount_m", "sum"),
        amount_max_m=("amount_m", "max"),
        pct_sum=("pct", "sum"),
        pct_abs_sum=("pct_abs", "sum"),
        pct_abs_max=("pct_abs", "max"),
        pct_signed_sum=("pct_signed", "sum"),
        has_amount_count=("has_amount", "sum"),
        has_pct_count=("has_pct", "sum"),
    ).astype(float)

    polarity_activity = (features["positive_count"] + features["negative_count"]).clip(lower=1.0)
    features["conflict_score"] = (
        np.minimum(features["positive_count"], features["negative_count"]) / polarity_activity
    )
    features["headline_density"] = features["headline_count"] / features["bar_coverage"].clip(lower=1.0)

    categorical_parts = [
        _count_by_category(
            events,
            category_col="event_family",
            prefix="event__",
            index_cols=["session", "company"],
            min_total_count=25,
            drop_suffixes=("_other",),
        ),
        _count_by_category(
            events,
            category_col="topic",
            prefix="topic__",
            index_cols=["session", "company"],
            drop_values={"none"},
        ),
        _count_by_category(
            events,
            category_col="geography",
            prefix="geo__",
            index_cols=["session", "company"],
            drop_values={"none"},
        ),
    ]
    categorical_parts = [part for part in categorical_parts if not part.empty]
    features = pd.concat([features] + categorical_parts, axis=1).fillna(0.0).sort_index().copy()
    return features


def score_company_relevance(company_features: pd.DataFrame) -> pd.DataFrame:
    if company_features.empty:
        return company_features.copy()

    scored = company_features.copy().sort_index().copy()
    session_level = scored.groupby(level=0, sort=True)

    scored["mention_share"] = scored["headline_count"] / session_level["headline_count"].transform("sum").clip(
        lower=1.0
    )
    scored["recent_share"] = scored["recent_headline_count"] / session_level["recent_headline_count"].transform(
        "sum"
    ).clip(lower=1.0)
    scored["bar_coverage_share"] = scored["bar_coverage"] / session_level["bar_coverage"].transform("sum").clip(
        lower=1.0
    )
    scored["last_bar_ix_norm"] = scored["last_bar_ix"] / session_level["last_bar_ix"].transform("max").clip(
        lower=1.0
    )
    scored["relevance_score"] = (
        0.35 * scored["mention_share"]
        + 0.30 * scored["recent_share"]
        + 0.20 * scored["bar_coverage_share"]
        + 0.15 * scored["last_bar_ix_norm"]
    )
    score_shift = scored["relevance_score"] - session_level["relevance_score"].transform("max")
    score_exp = np.exp(score_shift)
    scored["relevance_weight"] = score_exp / score_exp.groupby(level=0).transform("sum").clip(lower=1e-9)
    scored["relevance_rank"] = session_level["relevance_score"].rank(ascending=False, method="first")
    return scored


def _entropy(values: np.ndarray) -> float:
    probabilities = values[values > 0.0]
    if len(probabilities) == 0:
        return 0.0
    return float(-(probabilities * np.log(probabilities)).sum())


def build_session_text_features(
    company_features: pd.DataFrame,
    sessions=None,
    top_k: int = 2,
) -> pd.DataFrame:
    if company_features.empty:
        return _empty_feature_frame(sessions=sessions)

    company_frame = (
        company_features.reset_index()
        .sort_values(
            ["session", "relevance_score", "headline_count", "company"],
            ascending=[True, False, False, True],
        )
        .copy()
    )
    grouped = company_frame.groupby("session", sort=True)

    session_features = grouped[
        [
            "headline_count",
            "recent_headline_count",
            "positive_count",
            "negative_count",
            "neutral_count",
            "polarity_sum",
            "polarity_abs_sum",
            "amount_sum_m",
            "amount_max_m",
            "pct_sum",
            "pct_abs_sum",
            "pct_abs_max",
            "pct_signed_sum",
            "has_amount_count",
            "has_pct_count",
        ]
    ].sum()

    top2_relevance = grouped["relevance_score"].apply(lambda values: values.nlargest(2).min() if len(values) > 1 else 0.0)
    top2_weight = grouped["relevance_weight"].apply(lambda values: values.nlargest(2).min() if len(values) > 1 else 0.0)
    summary_features = pd.DataFrame(
        {
            "n_companies": grouped.size().astype(float),
            "headline_bar_coverage_sum": grouped["bar_coverage"].sum(),
            "headline_density_mean": grouped["headline_density"].mean(),
            "top_company_share": grouped["mention_share"].max(),
            "top_recent_share": grouped["recent_share"].max(),
            "top1_relevance": grouped["relevance_score"].max(),
            "top1_weight": grouped["relevance_weight"].max(),
            "top2_relevance": top2_relevance,
            "top2_weight": top2_weight,
            "company_entropy": grouped["mention_share"].apply(lambda values: _entropy(values.to_numpy(dtype=float))),
            "relevance_entropy": grouped["relevance_weight"].apply(
                lambda values: _entropy(values.to_numpy(dtype=float))
            ),
            "company_concentration": grouped["mention_share"].apply(
                lambda values: float(np.square(values.to_numpy(dtype=float)).sum())
            ),
            "relevance_concentration": grouped["relevance_weight"].apply(
                lambda values: float(np.square(values.to_numpy(dtype=float)).sum())
            ),
            "conflict_score_mean": grouped["conflict_score"].mean(),
            "conflict_score_max": grouped["conflict_score"].max(),
        }
    )
    summary_features["relevance_gap"] = summary_features["top1_relevance"] - summary_features["top2_relevance"]
    summary_features["weight_gap"] = summary_features["top1_weight"] - summary_features["top2_weight"]

    ranked = company_frame.copy()
    ranked["slot"] = ranked.groupby("session", sort=False).cumcount() + 1
    slot_columns = [
        "headline_count",
        "recent_headline_count",
        "bar_coverage",
        "last_bar_ix",
        "mean_bar_ix",
        "positive_count",
        "negative_count",
        "neutral_count",
        "polarity_sum",
        "polarity_abs_sum",
        "conflict_score",
        "amount_sum_m",
        "amount_max_m",
        "pct_sum",
        "pct_abs_sum",
        "pct_abs_max",
        "pct_signed_sum",
        "mention_share",
        "recent_share",
        "bar_coverage_share",
        "last_bar_ix_norm",
        "relevance_score",
        "relevance_weight",
    ]
    slot_frames: list[pd.DataFrame] = []
    for slot in range(1, top_k + 1):
        slot_frame = ranked.loc[ranked["slot"] == slot, ["session"] + slot_columns].set_index("session")
        slot_frames.append(slot_frame.add_prefix(f"top{slot}_"))

    weighted_columns = slot_columns + [
        column
        for column in company_frame.columns
        if column.startswith(("event__", "topic__", "geo__"))
    ]
    weighted_frame = company_frame[weighted_columns].copy()
    weighted_frame = weighted_frame.mul(company_frame["relevance_weight"], axis=0)
    weighted_frame["session"] = company_frame["session"].to_numpy()
    weighted = weighted_frame.groupby("session", sort=True).sum().add_prefix("w_")

    session_features = pd.concat([session_features, summary_features] + slot_frames + [weighted], axis=1)

    max_entropy = np.log(session_features["n_companies"].clip(lower=1.0)).replace(0.0, 1.0)
    entropy_features = pd.DataFrame(
        {
            "company_entropy_norm": np.divide(session_features["company_entropy"], max_entropy),
            "relevance_entropy_norm": np.divide(session_features["relevance_entropy"], max_entropy),
        },
        index=session_features.index,
    )
    session_features = pd.concat([session_features, entropy_features], axis=1)

    if sessions is not None:
        session_features = session_features.reindex(sorted(pd.Index(sessions).unique()))

    return session_features.fillna(0.0).sort_index()


def build_headline_features(headlines: pd.DataFrame, sessions=None) -> pd.DataFrame:
    if headlines.empty:
        return _empty_feature_frame(sessions=sessions)

    events = build_headline_event_table(headlines)
    company_features = build_company_session_features(events)
    company_features = score_company_relevance(company_features)
    return build_session_text_features(company_features, sessions=sessions)
