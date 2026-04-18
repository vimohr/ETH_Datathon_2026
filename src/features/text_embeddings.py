import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.features.headlines import (
    TEXT_SOURCE_COLUMNS,
    build_company_session_features,
    build_session_text_features,
    score_company_relevance,
)
from src.paths import EMBEDDINGS_CACHE_DIR, ensure_output_dirs

MODEL_INSTANCES: dict[str, SentenceTransformer] = {}
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _empty_feature_frame(sessions=None) -> pd.DataFrame:
    index = pd.Index(sorted(pd.Index(sessions).unique())) if sessions is not None else pd.Index([])
    return pd.DataFrame(index=index)


def _safe_name(value: str) -> str:
    return SAFE_NAME_PATTERN.sub("_", value)


def _resolve_text_column(text_source: str) -> str:
    if text_source not in TEXT_SOURCE_COLUMNS:
        valid = ", ".join(sorted(TEXT_SOURCE_COLUMNS))
        raise ValueError(f"Unsupported text_source={text_source!r}. Expected one of: {valid}.")
    return TEXT_SOURCE_COLUMNS[text_source]


def _cache_path(model_name: str, text_source: str, normalize_embeddings: bool) -> Path:
    suffix = "norm" if normalize_embeddings else "raw"
    filename = f"{_safe_name(model_name)}__{text_source}__{suffix}.parquet"
    return EMBEDDINGS_CACHE_DIR / filename


def _load_embedding_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    cached = pd.read_parquet(cache_path)
    if "text" not in cached.columns:
        return pd.DataFrame()
    return cached.drop_duplicates(subset=["text"]).set_index("text").sort_index()


def _save_embedding_cache(cache_path: Path, cache_frame: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output = cache_frame.reset_index().rename(columns={"index": "text"})
    output.to_parquet(cache_path, index=False)


def _get_encoder(model_name: str) -> SentenceTransformer:
    if model_name not in MODEL_INSTANCES:
        MODEL_INSTANCES[model_name] = SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=True,
        )
    return MODEL_INSTANCES[model_name]


def embed_texts(
    texts: pd.Series,
    *,
    model_name: str,
    text_source: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    ensure_output_dirs()

    text_series = texts.fillna("").astype(str)
    non_empty_unique = sorted({text for text in text_series if text})
    cache_path = _cache_path(model_name, text_source, normalize_embeddings)
    cache_frame = _load_embedding_cache(cache_path)

    missing_texts = [text for text in non_empty_unique if text not in cache_frame.index]
    if missing_texts:
        encoder = _get_encoder(model_name)
        encoded = encoder.encode(
            missing_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        encoded_frame = pd.DataFrame(
            encoded.astype(np.float32),
            index=pd.Index(missing_texts, name="text"),
            columns=[f"emb_{ix:03d}" for ix in range(encoded.shape[1])],
        )
        cache_frame = pd.concat([cache_frame, encoded_frame], axis=0)
        cache_frame = cache_frame[~cache_frame.index.duplicated(keep="last")].sort_index()
        _save_embedding_cache(cache_path, cache_frame)

    if cache_frame.empty:
        return pd.DataFrame(index=text_series.index)

    embedding_columns = list(cache_frame.columns)
    embedding_values = np.zeros((len(text_series), len(embedding_columns)), dtype=np.float32)
    lookup = cache_frame.to_dict(orient="index")

    for row_ix, text in enumerate(text_series):
        if not text:
            continue
        vector = lookup.get(text)
        if vector is None:
            continue
        embedding_values[row_ix] = np.asarray([vector[column] for column in embedding_columns], dtype=np.float32)

    return pd.DataFrame(embedding_values, index=text_series.index, columns=embedding_columns)


def build_headline_embedding_table(
    events: pd.DataFrame,
    *,
    text_source: str,
    model_name: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    text_column = _resolve_text_column(text_source)
    embedded = events.copy()
    embedding_frame = embed_texts(
        embedded[text_column],
        model_name=model_name,
        text_source=text_source,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    return pd.concat([embedded, embedding_frame], axis=1)


def _weighted_group_average(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    embedding_cols: list[str],
    weight_col: str,
    prefix: str,
) -> pd.DataFrame:
    weighted = frame[embedding_cols].mul(frame[weight_col].to_numpy(dtype=float), axis=0)
    weighted[group_cols] = frame[group_cols]
    numerator = weighted.groupby(group_cols, sort=True)[embedding_cols].sum()
    denominator = frame.groupby(group_cols, sort=True)[weight_col].sum().clip(lower=1e-9)
    averaged = numerator.div(denominator, axis=0)
    averaged.columns = [f"{prefix}{column}" for column in averaged.columns]
    return averaged


def _relevance_summary_features(company_features: pd.DataFrame, *, sessions=None, top_k: int = 2) -> pd.DataFrame:
    session_text = build_session_text_features(company_features, sessions=sessions, top_k=top_k)
    summary_columns = [
        "n_companies",
        "top_company_share",
        "top_recent_share",
        "top1_relevance",
        "top1_weight",
        "top2_relevance",
        "top2_weight",
        "relevance_gap",
        "weight_gap",
        "company_entropy",
        "relevance_entropy",
        "company_concentration",
        "relevance_concentration",
    ]
    existing_columns = [column for column in summary_columns if column in session_text.columns]
    return session_text.reindex(columns=existing_columns).fillna(0.0)


def build_session_embedding_features(
    events: pd.DataFrame,
    *,
    sessions=None,
    text_source: str,
    model_name: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    include_structured: bool = False,
) -> pd.DataFrame:
    if events.empty:
        return _empty_feature_frame(sessions=sessions)

    embedded = build_headline_embedding_table(
        events,
        text_source=text_source,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    embedding_cols = [column for column in embedded.columns if column.startswith("emb_")]
    if not embedding_cols:
        return _empty_feature_frame(sessions=sessions)

    session_mean = embedded.groupby("session", sort=True)[embedding_cols].mean().add_prefix("session_")
    recent_weighted = _weighted_group_average(
        embedded,
        group_cols=["session"],
        embedding_cols=embedding_cols,
        weight_col="is_recent",
        prefix="recent_",
    )
    session_features = pd.concat([session_mean, recent_weighted], axis=1)

    if include_structured:
        company_features = score_company_relevance(build_company_session_features(events))
        session_features = session_features.join(
            build_session_text_features(company_features, sessions=sessions),
            how="left",
        )

    if sessions is not None:
        session_features = session_features.reindex(sorted(pd.Index(sessions).unique()))

    return session_features.fillna(0.0).sort_index()


def build_company_embedding_features(
    events: pd.DataFrame,
    *,
    sessions=None,
    text_source: str,
    model_name: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    top_k: int = 2,
    include_structured: bool = False,
    include_top_slots: bool = True,
) -> pd.DataFrame:
    if events.empty:
        return _empty_feature_frame(sessions=sessions)

    embedded = build_headline_embedding_table(
        events,
        text_source=text_source,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    embedding_cols = [column for column in embedded.columns if column.startswith("emb_")]
    if not embedding_cols:
        return _empty_feature_frame(sessions=sessions)

    company_features = score_company_relevance(build_company_session_features(events))
    company_mean = embedded.groupby(["session", "company"], sort=True)[embedding_cols].mean()
    company_recent = _weighted_group_average(
        embedded,
        group_cols=["session", "company"],
        embedding_cols=embedding_cols,
        weight_col="is_recent",
        prefix="recent_",
    )
    company_frame = company_features.join(company_mean, how="left").join(company_recent, how="left").fillna(0.0)

    ranked = (
        company_frame.reset_index()
        .sort_values(
            ["session", "relevance_score", "headline_count", "company"],
            ascending=[True, False, False, True],
        )
        .copy()
    )
    ranked["slot"] = ranked.groupby("session", sort=False).cumcount() + 1

    slot_columns = embedding_cols + [f"recent_{column}" for column in embedding_cols]
    slot_frames: list[pd.DataFrame] = []
    if include_top_slots:
        for slot in range(1, top_k + 1):
            slot_frame = ranked.loc[ranked["slot"] == slot, ["session"] + slot_columns].set_index("session")
            slot_frame.columns = [f"top{slot}_{column}" for column in slot_frame.columns]
            slot_frames.append(slot_frame)

    weighted_columns = embedding_cols + [f"recent_{column}" for column in embedding_cols]
    weighted_frame = ranked[weighted_columns].mul(ranked["relevance_weight"], axis=0)
    weighted_frame["session"] = ranked["session"].to_numpy()
    weighted_embedding = weighted_frame.groupby("session", sort=True).sum().add_prefix("weighted_")

    session_parts: list[pd.DataFrame] = []
    if slot_frames:
        session_parts.extend(slot_frames)
    session_parts.append(weighted_embedding)
    if not include_top_slots:
        session_parts.append(_relevance_summary_features(company_features, sessions=sessions, top_k=top_k))

    session_features = pd.concat(session_parts, axis=1)

    if include_structured:
        session_features = session_features.join(
            build_session_text_features(company_features, sessions=sessions, top_k=top_k),
            how="left",
        )

    if sessions is not None:
        session_features = session_features.reindex(sorted(pd.Index(sessions).unique()))

    return session_features.fillna(0.0).sort_index()


def build_embedding_feature_frame(
    events: pd.DataFrame,
    *,
    sessions=None,
    text_source: str = "event_normalized",
    aggregation: str = "session",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    top_k: int = 2,
    include_structured: bool = False,
) -> pd.DataFrame:
    if aggregation == "session":
        return build_session_embedding_features(
            events,
            sessions=sessions,
            text_source=text_source,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            include_structured=include_structured,
        )

    if aggregation in {"company_top2", "company_topk", "company_weighted"}:
        company_top_k = 2 if aggregation == "company_top2" else top_k
        return build_company_embedding_features(
            events,
            sessions=sessions,
            text_source=text_source,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            top_k=company_top_k,
            include_structured=include_structured,
            include_top_slots=aggregation != "company_weighted",
        )

    raise ValueError(
        f"Unsupported aggregation={aggregation!r}. Expected one of: session, company_top2, company_topk, company_weighted."
    )
