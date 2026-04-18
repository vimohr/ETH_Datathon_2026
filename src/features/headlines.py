import numpy as np
import pandas as pd

from src.features.headline_parser import parse_headlines


RECENT_BAR_WINDOW = 9
VERY_RECENT_BAR_WINDOW = 4
CATEGORICAL_PREFIXES = ("event__", "topic__", "geo__", "verb__", "counterparty__")


def _empty_feature_frame(sessions=None) -> pd.DataFrame:
    index = pd.Index(sorted(pd.Index(sessions).unique())) if sessions is not None else pd.Index([])
    return pd.DataFrame(index=index)


def _top_share_margin(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    shares = np.sort(values / total)[::-1]
    top1 = float(shares[0]) if len(shares) else 0.0
    top2 = float(shares[1]) if len(shares) > 1 else 0.0
    return top1 - top2


def _polarity_flip_stats(polarities: np.ndarray) -> tuple[float, float]:
    non_zero = np.sign(polarities[polarities != 0.0])
    if len(non_zero) <= 1:
        return 0.0, 0.0
    flip_count = float(np.sum(non_zero[1:] != non_zero[:-1]))
    flip_rate = flip_count / float(len(non_zero) - 1)
    return flip_count, flip_rate


def _gap_stats(bar_index: np.ndarray) -> tuple[float, float, float]:
    if len(bar_index) <= 1:
        return 0.0, 0.0, 0.0
    gaps = np.diff(bar_index.astype(float))
    gap_mean = float(gaps.mean())
    gap_std = float(gaps.std(ddof=0))
    gap_cv = float(gap_std / max(gap_mean, 1.0))
    denominator = gap_std + gap_mean
    burstiness = float((gap_std - gap_mean) / denominator) if denominator > 0.0 else 0.0
    return gap_mean, gap_cv, burstiness


def _sequence_stats_from_events(event_frame: pd.DataFrame) -> dict[str, float]:
    ordered = event_frame.sort_values(["bar_ix", "headline"], kind="stable").copy()
    session_max_bar = float(ordered["session_max_bar"].iloc[0])
    bar_index = ordered["bar_ix"].to_numpy(dtype=float)
    polarities = ordered["polarity"].to_numpy(dtype=float)
    flip_count, flip_rate = _polarity_flip_stats(polarities)
    gap_mean, gap_cv, burstiness = _gap_stats(bar_index)
    strong_bars = ordered.loc[ordered["is_strong_event"] > 0.0, "bar_ix"]
    if strong_bars.empty:
        time_since_last_strong_event = session_max_bar + 1.0
    else:
        time_since_last_strong_event = float(session_max_bar - float(strong_bars.max()))

    bar_counts = ordered.groupby("bar_ix", sort=True).size().to_numpy(dtype=float)
    max_headlines_single_bar = float(bar_counts.max()) if len(bar_counts) else 0.0
    multi_headline_bar_count = float(np.sum(bar_counts >= 2.0))
    multi_headline_bar_share = multi_headline_bar_count / max(session_max_bar + 1.0, 1.0)

    return {
        "polarity_flip_count": flip_count,
        "polarity_flip_rate": flip_rate,
        "time_since_last_strong_event": time_since_last_strong_event,
        "headline_gap_mean": gap_mean,
        "headline_gap_cv": gap_cv,
        "headline_burstiness": burstiness,
        "max_headlines_single_bar": max_headlines_single_bar,
        "multi_headline_bar_count": multi_headline_bar_count,
        "multi_headline_bar_share": multi_headline_bar_share,
    }


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

    session_max_bar = events.groupby("session")["bar_ix"].transform("max").astype(float)
    events["session_max_bar"] = session_max_bar
    events["bars_from_end"] = session_max_bar - events["bar_ix"].astype(float)
    events["bar_fraction"] = np.divide(
        events["bar_ix"].astype(float),
        session_max_bar.clip(lower=1.0),
    )
    events["is_recent"] = (events["bars_from_end"] <= RECENT_BAR_WINDOW).astype(float)
    events["is_very_recent"] = (events["bars_from_end"] <= VERY_RECENT_BAR_WINDOW).astype(float)
    events["is_early_half"] = (events["bar_fraction"] <= 0.5).astype(float)
    events["is_late_half"] = 1.0 - events["is_early_half"]

    strong_event = (
        (events["polarity_abs"] > 0.0)
        & (
            (events["has_pct"] > 0)
            | (events["has_amount"] > 0)
            | (~events["event_family"].astype(str).str.endswith("_other"))
        )
    )
    events["is_strong_event"] = strong_event.astype(float)
    events["late_positive_flag"] = events["positive_flag"] * events["is_recent"]
    events["late_negative_flag"] = events["negative_flag"] * events["is_recent"]
    events["late_neutral_flag"] = events["neutral_flag"] * events["is_recent"]
    events["late_strong_flag"] = events["is_strong_event"] * events["is_recent"]
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
        very_recent_headline_count=("is_very_recent", "sum"),
        early_half_headline_count=("is_early_half", "sum"),
        late_half_headline_count=("is_late_half", "sum"),
        bar_coverage=("bar_ix", "nunique"),
        last_bar_ix=("bar_ix", "max"),
        mean_bar_ix=("bar_ix", "mean"),
        positive_count=("positive_flag", "sum"),
        negative_count=("negative_flag", "sum"),
        neutral_count=("neutral_flag", "sum"),
        late_positive_count=("late_positive_flag", "sum"),
        late_negative_count=("late_negative_flag", "sum"),
        late_neutral_count=("late_neutral_flag", "sum"),
        strong_event_count=("is_strong_event", "sum"),
        late_strong_event_count=("late_strong_flag", "sum"),
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

    sequence_rows: list[dict[str, float | str]] = []
    for (session, company), company_events in grouped:
        sequence_rows.append(
            {
                "session": int(session),
                "company": company,
                **_sequence_stats_from_events(company_events),
            }
        )
    sequence_features = pd.DataFrame(sequence_rows).set_index(["session", "company"]).sort_index()
    features = features.join(sequence_features, how="left")

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
        _count_by_category(
            events,
            category_col="verb_family",
            prefix="verb__",
            index_cols=["session", "company"],
            min_total_count=10,
            drop_values={"unknown"},
        ),
        _count_by_category(
            events,
            category_col="counterparty",
            prefix="counterparty__",
            index_cols=["session", "company"],
            min_total_count=20,
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
    scored["early_share"] = scored["early_half_headline_count"] / session_level[
        "early_half_headline_count"
    ].transform("sum").clip(lower=1.0)
    scored["late_share"] = scored["late_half_headline_count"] / session_level[
        "late_half_headline_count"
    ].transform("sum").clip(lower=1.0)
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


def _weighted_polarity_disagreement(company_frame: pd.DataFrame) -> float:
    active = company_frame.loc[
        (company_frame["positive_count"] + company_frame["negative_count"]) > 0.0,
        ["relevance_weight", "polarity_sum"],
    ]
    if active.empty:
        return 0.0
    weights = active["relevance_weight"].to_numpy(dtype=float)
    directions = np.sign(active["polarity_sum"].to_numpy(dtype=float))
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return 0.0
    return float(1.0 - abs(np.dot(weights, directions)) / total_weight)


def _top_company_switch(company_frame: pd.DataFrame) -> float:
    if float(company_frame["early_half_headline_count"].sum()) <= 0.0:
        return 0.0
    if float(company_frame["late_half_headline_count"].sum()) <= 0.0:
        return 0.0
    early_top = company_frame.sort_values(["early_share", "company"], ascending=[False, True]).iloc[0]["company"]
    late_top = company_frame.sort_values(["late_share", "company"], ascending=[False, True]).iloc[0]["company"]
    return float(early_top != late_top)


def build_session_sequence_features(events: pd.DataFrame, sessions=None) -> pd.DataFrame:
    if events.empty:
        return _empty_feature_frame(sessions=sessions)

    rows: list[dict[str, float]] = []
    for session, session_events in events.groupby("session", sort=True):
        rows.append({"session": int(session), **_sequence_stats_from_events(session_events)})

    features = pd.DataFrame(rows).set_index("session").sort_index()
    if sessions is not None:
        features = features.reindex(sorted(pd.Index(sessions).unique()))
    return features.fillna(0.0)


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

    additive_columns = [
        "headline_count",
        "recent_headline_count",
        "very_recent_headline_count",
        "early_half_headline_count",
        "late_half_headline_count",
        "positive_count",
        "negative_count",
        "neutral_count",
        "late_positive_count",
        "late_negative_count",
        "late_neutral_count",
        "strong_event_count",
        "late_strong_event_count",
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
    session_features = grouped[additive_columns].sum()

    top2_relevance = grouped["relevance_score"].apply(
        lambda values: float(values.nlargest(2).min()) if len(values) > 1 else 0.0
    )
    top2_weight = grouped["relevance_weight"].apply(
        lambda values: float(values.nlargest(2).min()) if len(values) > 1 else 0.0
    )
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
            "weighted_conflict_score": grouped.apply(
                lambda frame: float(
                    np.dot(
                        frame["relevance_weight"].to_numpy(dtype=float),
                        frame["conflict_score"].to_numpy(dtype=float),
                    )
                )
            ),
            "weighted_polarity_disagreement": grouped.apply(_weighted_polarity_disagreement),
            "positive_company_margin": grouped.apply(
                lambda frame: _top_share_margin(frame["positive_count"].to_numpy(dtype=float))
            ),
            "negative_company_margin": grouped.apply(
                lambda frame: _top_share_margin(frame["negative_count"].to_numpy(dtype=float))
            ),
            "strong_event_company_margin": grouped.apply(
                lambda frame: _top_share_margin(frame["strong_event_count"].to_numpy(dtype=float))
            ),
            "relevance_instability": grouped.apply(
                lambda frame: float(
                    0.5
                    * np.abs(
                        frame["early_share"].to_numpy(dtype=float) - frame["late_share"].to_numpy(dtype=float)
                    ).sum()
                )
            ),
            "top_company_switch_flag": grouped.apply(_top_company_switch),
        }
    )
    summary_features["relevance_gap"] = summary_features["top1_relevance"] - summary_features["top2_relevance"]
    summary_features["weight_gap"] = summary_features["top1_weight"] - summary_features["top2_weight"]

    ranked = company_frame.copy()
    ranked["slot"] = ranked.groupby("session", sort=False).cumcount() + 1
    slot_columns = [
        "headline_count",
        "recent_headline_count",
        "very_recent_headline_count",
        "early_half_headline_count",
        "late_half_headline_count",
        "bar_coverage",
        "last_bar_ix",
        "mean_bar_ix",
        "positive_count",
        "negative_count",
        "neutral_count",
        "late_positive_count",
        "late_negative_count",
        "late_neutral_count",
        "strong_event_count",
        "late_strong_event_count",
        "polarity_sum",
        "polarity_abs_sum",
        "conflict_score",
        "polarity_flip_count",
        "polarity_flip_rate",
        "time_since_last_strong_event",
        "headline_gap_mean",
        "headline_gap_cv",
        "headline_burstiness",
        "max_headlines_single_bar",
        "amount_sum_m",
        "amount_max_m",
        "pct_sum",
        "pct_abs_sum",
        "pct_abs_max",
        "pct_signed_sum",
        "mention_share",
        "recent_share",
        "early_share",
        "late_share",
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
        if column.startswith(CATEGORICAL_PREFIXES)
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


def build_headline_price_interaction_features(
    events: pd.DataFrame,
    company_features: pd.DataFrame,
    bars: pd.DataFrame,
    sessions=None,
) -> pd.DataFrame:
    if events.empty or bars.empty:
        return _empty_feature_frame(sessions=sessions)

    bars_sorted = bars.sort_values(["session", "bar_ix"]).copy()
    bar_grouped = bars_sorted.groupby("session", sort=True)
    bars_sorted["next_close"] = bar_grouped["close"].shift(-1)
    bars_sorted["close_to_seen_end"] = bar_grouped["close"].transform("last")
    bars_sorted["bar_return"] = bar_grouped["close"].pct_change().fillna(0.0)
    bars_sorted["intrabar_range_ratio"] = (
        (bars_sorted["high"] - bars_sorted["low"]) / bars_sorted["open"].clip(lower=1e-6)
    )
    bars_sorted["next_close_return"] = np.divide(
        bars_sorted["next_close"] - bars_sorted["close"],
        bars_sorted["close"].clip(lower=1e-6),
    )
    bars_sorted["next_close_return"] = bars_sorted["next_close_return"].fillna(0.0)
    bars_sorted["return_to_seen_end"] = np.divide(
        bars_sorted["close_to_seen_end"] - bars_sorted["close"],
        bars_sorted["close"].clip(lower=1e-6),
    )

    event_frame = events.merge(
        bars_sorted[
            [
                "session",
                "bar_ix",
                "bar_return",
                "intrabar_range_ratio",
                "next_close_return",
                "return_to_seen_end",
            ]
        ],
        on=["session", "bar_ix"],
        how="left",
    )
    event_frame = event_frame.merge(
        company_features.reset_index()[["session", "company", "relevance_weight", "relevance_rank"]],
        on=["session", "company"],
        how="left",
    ).fillna({"relevance_weight": 0.0, "relevance_rank": 0.0})

    event_frame["polarity_alignment_to_seen_end"] = (
        np.sign(event_frame["return_to_seen_end"]).astype(float) * event_frame["polarity"].astype(float)
    )
    event_frame["polarity_alignment_next_bar"] = (
        np.sign(event_frame["next_close_return"]).astype(float) * event_frame["polarity"].astype(float)
    )
    event_frame["signed_seen_reaction"] = (
        event_frame["return_to_seen_end"].astype(float) * event_frame["polarity"].astype(float)
    )
    event_frame["signed_next_bar_reaction"] = (
        event_frame["next_close_return"].astype(float) * event_frame["polarity"].astype(float)
    )
    event_frame["high_relevance_flag"] = (event_frame["relevance_rank"] == 1.0).astype(float)

    rows: list[dict[str, float]] = []
    for session, session_events in event_frame.groupby("session", sort=True):
        session_bars = bars_sorted.loc[bars_sorted["session"] == session].copy()
        active_events = session_events.loc[session_events["polarity"] != 0].copy()
        high_relevance_events = active_events.loc[active_events["high_relevance_flag"] > 0.0].copy()

        if active_events.empty:
            agreement_to_seen_end = 0.0
            agreement_next_bar = 0.0
            weighted_agreement_to_seen_end = 0.0
            weighted_signed_seen_reaction = 0.0
            weighted_signed_next_bar_reaction = 0.0
            positive_headline_seen_return_mean = 0.0
            negative_headline_seen_return_mean = 0.0
        else:
            active_weights = active_events["relevance_weight"].to_numpy(dtype=float)
            weight_sum = float(active_weights.sum())
            agreement_to_seen_end = float(active_events["polarity_alignment_to_seen_end"].mean())
            agreement_next_bar = float(active_events["polarity_alignment_next_bar"].mean())
            positive_headline_seen_return_mean = float(
                active_events.loc[active_events["polarity"] > 0.0, "return_to_seen_end"].mean()
            )
            negative_headline_seen_return_mean = float(
                active_events.loc[active_events["polarity"] < 0.0, "return_to_seen_end"].mean()
            )
            if weight_sum > 0.0:
                weighted_agreement_to_seen_end = float(
                    np.dot(active_events["polarity_alignment_to_seen_end"].to_numpy(dtype=float), active_weights)
                    / weight_sum
                )
                weighted_signed_seen_reaction = float(
                    np.dot(active_events["signed_seen_reaction"].to_numpy(dtype=float), active_weights) / weight_sum
                )
                weighted_signed_next_bar_reaction = float(
                    np.dot(active_events["signed_next_bar_reaction"].to_numpy(dtype=float), active_weights)
                    / weight_sum
                )
            else:
                weighted_agreement_to_seen_end = agreement_to_seen_end
                weighted_signed_seen_reaction = float(active_events["signed_seen_reaction"].mean())
                weighted_signed_next_bar_reaction = float(active_events["signed_next_bar_reaction"].mean())

        headline_counts = (
            session_events.groupby("bar_ix", sort=True)
            .size()
            .reindex(session_bars["bar_ix"], fill_value=0.0)
            .astype(float)
        )
        burst_strength = headline_counts.rolling(window=3, center=True, min_periods=1).sum()
        burst_mask = (burst_strength >= 2.0).to_numpy(dtype=bool)
        burst_bar_share = float(burst_mask.mean()) if len(burst_mask) else 0.0
        overall_volatility = float(session_bars["bar_return"].std(ddof=0))
        burst_returns = session_bars.loc[burst_mask, "bar_return"]
        burst_volatility = float(burst_returns.std(ddof=0)) if len(burst_returns) else 0.0
        burst_volatility_ratio = burst_volatility / max(overall_volatility, 1e-6) if len(burst_returns) else 0.0
        burst_range_mean = float(session_bars.loc[burst_mask, "intrabar_range_ratio"].mean()) if len(burst_returns) else 0.0

        rows.append(
            {
                "session": int(session),
                "headline_price_agreement_mean": agreement_to_seen_end,
                "headline_price_next_bar_agreement_mean": agreement_next_bar,
                "headline_price_agreement_weighted": weighted_agreement_to_seen_end,
                "headline_signed_seen_reaction_weighted": weighted_signed_seen_reaction,
                "headline_signed_next_bar_reaction_weighted": weighted_signed_next_bar_reaction,
                "high_relevance_signed_seen_reaction_mean": float(
                    high_relevance_events["signed_seen_reaction"].mean()
                )
                if not high_relevance_events.empty
                else 0.0,
                "high_relevance_agreement_mean": float(high_relevance_events["polarity_alignment_to_seen_end"].mean())
                if not high_relevance_events.empty
                else 0.0,
                "positive_headline_seen_return_mean": positive_headline_seen_return_mean,
                "negative_headline_seen_return_mean": negative_headline_seen_return_mean,
                "burst_bar_share": burst_bar_share,
                "burst_window_volatility": burst_volatility,
                "burst_window_volatility_ratio": burst_volatility_ratio,
                "burst_window_range_mean": burst_range_mean,
            }
        )

    features = pd.DataFrame(rows).set_index("session").sort_index()
    if sessions is not None:
        features = features.reindex(sorted(pd.Index(sessions).unique()))
    return features.fillna(0.0)


def build_headline_features(headlines: pd.DataFrame, sessions=None, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    if headlines.empty:
        return _empty_feature_frame(sessions=sessions)

    events = build_headline_event_table(headlines)
    company_features = build_company_session_features(events)
    company_features = score_company_relevance(company_features)

    feature_parts = [
        build_session_text_features(company_features, sessions=sessions),
        build_session_sequence_features(events, sessions=sessions),
    ]
    if bars is not None:
        feature_parts.append(
            build_headline_price_interaction_features(events, company_features, bars, sessions=sessions)
        )

    features = pd.concat(feature_parts, axis=1)
    if sessions is not None:
        features = features.reindex(sorted(pd.Index(sessions).unique()))
    return features.fillna(0.0).sort_index()
