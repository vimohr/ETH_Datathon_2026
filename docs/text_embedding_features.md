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

headlines = load_headlines("train", "seen")
sessions = load_bars("train", "seen")["session"].unique()

X_text = build_text_embedding_features(
    headlines,
    sessions=sessions,
    text_source="event_normalized",
    aggregation="company_weighted",
)
```

Key parameters:

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

## Multi-Block Usage

Use this when you want several text embedding blocks joined into one feature map.
Each block is automatically prefixed to avoid column collisions.

```python
from src.data.load import load_bars, load_headlines
from src.features.text_embeddings import build_text_embedding_feature_map

headlines = load_headlines("train", "seen")
sessions = load_bars("train", "seen")["session"].unique()

X_text = build_text_embedding_feature_map(
    headlines,
    sessions=sessions,
    blocks=[
        {
            "name": "cw_event_norm",
            "text_source": "event_normalized",
            "aggregation": "company_weighted",
        },
        {
            "name": "sess_event_norm",
            "text_source": "event_normalized",
            "aggregation": "session",
        },
    ],
)
```

Example output columns:

- `cw_event_norm__weighted_emb_000`
- `cw_event_norm__top1_weight`
- `sess_event_norm__session_emb_000`
- `sess_event_norm__recent_emb_000`

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
- The current implementation does not depend on price data.
  It only uses seen headlines and parser-derived company relevance.
