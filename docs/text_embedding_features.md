# Text Embedding Features

This repo now exposes text embeddings as a pure feature family under
`src/features/text_embeddings.py`.

The goal is to match the existing pipeline convention on `main`:

- raw split data in
- session-indexed numeric feature frame out
- model / CV / submission logic handled elsewhere

## Public API

Low-level single-block builder:

```python
from src.features.text_embeddings import build_text_embedding_features
```

High-level multi-block helper:

```python
from src.features.text_embeddings import build_text_embedding_feature_map
```

## Single Block Usage

Use this when you want one embedding feature family only.

```python
from src.data.load import load_bars, load_headlines
from src.features.text_embeddings import build_text_embedding_features

bars = load_bars("train", "seen")
headlines = load_headlines("train", "seen")
sessions = bars["session"].unique()

X_text = build_text_embedding_features(
    headlines,
    sessions=sessions,
    bars=bars,
    text_source="event_normalized",
    aggregation="company_weighted",
    include_structured_features=True,
    include_sequence_features=True,
    include_price_interactions=True,
)
```

Key parameters:

- `bars`
  - optional for plain text embeddings
  - required when `include_price_interactions=True`
- `text_source`
  - `headline`
  - `event`
  - `headline_normalized`
  - `event_normalized`
- `aggregation`
  - `company_weighted`
  - `session`
- `model_name`
  - defaults to `sentence-transformers/all-MiniLM-L6-v2`
- `include_relevance_summary`
  - append the lightweight relevance summary block
  - defaults to `True`
- `include_structured_features`
  - append the full structured parser feature stack
  - includes company/session summaries, categorical parser counts, ambiguity metrics, and top-company slots
  - defaults to `False`
- `include_sequence_features`
  - append session-level sequence and burst features
  - examples: polarity flips, time since last strong event, headline gap CV, burstiness
  - defaults to `False`
- `include_price_interactions`
  - append text-price interaction features derived from seen bars
  - examples: polarity/price agreement, signed reactions, volatility around news bursts
  - defaults to `False`

Notes on combinations:

- Existing callers do not need to pass these flags; defaults preserve the old embedding-only behavior.
- `include_structured_features=True` subsumes the smaller relevance summary block.

## Multi-Block Usage

Use this when you want several text embedding blocks joined into one feature map.
Each block is automatically prefixed to avoid column collisions.

```python
from src.data.load import load_bars, load_headlines
from src.features.text_embeddings import build_text_embedding_feature_map

bars = load_bars("train", "seen")
headlines = load_headlines("train", "seen")
sessions = bars["session"].unique()

X_text = build_text_embedding_feature_map(
    headlines,
    sessions=sessions,
    bars=bars,
    blocks=[
        {
            "name": "cw_event_norm",
            "text_source": "event_normalized",
            "aggregation": "company_weighted",
            "include_structured_features": True,
            "include_sequence_features": True,
            "include_price_interactions": True,
        },
        {
            "name": "sess_event_norm",
            "text_source": "event_normalized",
            "aggregation": "session",
            "include_structured_features": True,
            "include_sequence_features": True,
        },
    ],
)
```

Example output columns:

- `cw_event_norm__weighted_emb_000`
- `cw_event_norm__top1_weight`
- `cw_event_norm__weighted_polarity_disagreement`
- `cw_event_norm__headline_price_agreement_mean`
- `sess_event_norm__session_emb_000`
- `sess_event_norm__recent_emb_000`
- `sess_event_norm__polarity_flip_rate`

## Default Feature Map

If `blocks` is omitted, `build_text_embedding_feature_map(...)` currently builds:

1. `text_cw_event_norm`
   `company_weighted` on `event_normalized`
2. `text_session_event_norm`
   `session` on `event_normalized`

## Output Contract

Both public functions return a frame with:

- index: `session`
- columns: numeric only
- no missing values
- deterministic column names for a fixed config

This is meant to plug directly into the existing training pipeline on `main`.

## Notes

- Embeddings are cached under `outputs/models/embeddings_cache/`.
- The encoder is loaded with `local_files_only=True`.
  The model weights must already exist on the machine.
- The default embedding setup still does not depend on price data.
- If you enable `include_price_interactions=True`, the feature builder also uses seen bars.
- The structured add-ons reuse the same parser/relevance code path as `src/features/headlines.py`.
