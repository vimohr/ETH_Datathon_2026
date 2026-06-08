import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.headlines import (
    build_company_session_features,
    build_headline_event_table,
    build_session_text_features,
    score_company_relevance,
)
from src.paths import EMBEDDINGS_CACHE_DIR, ensure_output_dirs

TEXT_SOURCE_COLUMNS = {
    "headline": "headline",
    "event": "event_text",
    "headline_normalized": "headline_normalized",
    "event_normalized": "event_text_normalized",
}

MODEL_INSTANCES: dict[str, object] = {}
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
# Default multi-block feature map used when the caller wants one combined
# session-level text matrix without managing prefixes manually.
DEFAULT_EMBEDDING_BLOCKS = (
    {
        "name": "text_cw_event_norm",
        "text_source": "event_normalized",
        "aggregation": "company_weighted",
    },
    {
        "name": "text_session_event_norm",
        "text_source": "event_normalized",
        "aggregation": "session",
    },
)


def _empty_feature_frame(sessions=None) -> pd.DataFrame:
    index = pd.Index(sorted(pd.Index(sessions).unique())) if sessions is not None else pd.Index([])
    return pd.DataFrame(index=index.rename("session"))


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


def _get_encoder(model_name: str):
    if model_name not in MODEL_INSTANCES:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for text embedding features. "
                "Install project requirements first."
            ) from exc

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


def _build_headline_embedding_table(
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


def _relevance_summary_features(company_features: pd.DataFrame, *, sessions=None) -> pd.DataFrame:
    session_text = build_session_text_features(company_features, sessions=sessions, top_k=2)
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


def _build_session_embedding_features(
    events: pd.DataFrame,
    *,
    sessions=None,
    text_source: str,
    model_name: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    embedded = _build_headline_embedding_table(
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
    features = pd.concat([session_mean, recent_weighted], axis=1)

    if sessions is not None:
        features = features.reindex(sorted(pd.Index(sessions).unique()))

    return features.fillna(0.0).sort_index()


def _build_company_weighted_embedding_features(
    events: pd.DataFrame,
    *,
    sessions=None,
    text_source: str,
    model_name: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    include_relevance_summary: bool = True,
) -> pd.DataFrame:
    embedded = _build_headline_embedding_table(
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
    weighted_columns = embedding_cols + [f"recent_{column}" for column in embedding_cols]
    weighted_frame = ranked[weighted_columns].mul(ranked["relevance_weight"], axis=0)
    weighted_frame["session"] = ranked["session"].to_numpy()
    features = weighted_frame.groupby("session", sort=True).sum().add_prefix("weighted_")

    if include_relevance_summary:
        features = features.join(_relevance_summary_features(company_features, sessions=sessions), how="left")

    if sessions is not None:
        features = features.reindex(sorted(pd.Index(sessions).unique()))

    return features.fillna(0.0).sort_index()


def build_text_embedding_features(
    headlines: pd.DataFrame,
    *,
    sessions=None,
    text_source: str = "event_normalized",
    aggregation: str = "company_weighted",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    include_relevance_summary: bool = True,
) -> pd.DataFrame:
    """Build one session-level text embedding feature block from raw headlines.

    This is the low-level entrypoint used when the caller wants one specific
    embedding family, for example `company_weighted` on `event_normalized`.
    The returned frame is numeric-only and indexed by `session`.
    """
    if headlines.empty:
        return _empty_feature_frame(sessions=sessions)

    events = build_headline_event_table(headlines)
    if aggregation == "session":
        return _build_session_embedding_features(
            events,
            sessions=sessions,
            text_source=text_source,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
    if aggregation == "company_weighted":
        return _build_company_weighted_embedding_features(
            events,
            sessions=sessions,
            text_source=text_source,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            include_relevance_summary=include_relevance_summary,
        )

    raise ValueError(
        f"Unsupported aggregation={aggregation!r}. Expected one of: company_weighted, session."
    )


def build_text_embedding_feature_map(
    headlines: pd.DataFrame,
    *,
    sessions=None,
    blocks: list[dict] | tuple[dict, ...] | None = None,
) -> pd.DataFrame:
    """Build and join multiple prefixed text embedding blocks.

    Each block is a dict with at least:
    - `name`: prefix used for all columns in that block
    - `text_source`: one of TEXT_SOURCE_COLUMNS
    - `aggregation`: `company_weighted` or `session`

    The result is a single numeric feature map that can be handed directly to
    the existing model pipeline on `main`.
    """
    block_specs = list(blocks or DEFAULT_EMBEDDING_BLOCKS)
    if not block_specs:
        return _empty_feature_frame(sessions=sessions)

    feature_blocks: list[pd.DataFrame] = []
    seen_names: set[str] = set()
    for block in block_specs:
        block_name = str(block.get("name") or "").strip()
        if not block_name:
            raise ValueError("Each embedding block must include a non-empty 'name'.")
        if block_name in seen_names:
            raise ValueError(f"Duplicate embedding block name: {block_name!r}")
        seen_names.add(block_name)

        feature_frame = build_text_embedding_features(
            headlines,
            sessions=sessions,
            text_source=block.get("text_source", "event_normalized"),
            aggregation=block.get("aggregation", "company_weighted"),
            model_name=block.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=int(block.get("batch_size", 128)),
            normalize_embeddings=bool(block.get("normalize_embeddings", True)),
            include_relevance_summary=bool(block.get("include_relevance_summary", True)),
        ).add_prefix(f"{block_name}__")
        feature_blocks.append(feature_frame)

    combined = pd.concat(feature_blocks, axis=1)
    if sessions is not None:
        combined = combined.reindex(sorted(pd.Index(sessions).unique()))
    return combined.fillna(0.0).sort_index()
