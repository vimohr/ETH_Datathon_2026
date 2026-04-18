# Text Embeddings Plan

## Scope

- Worktree: `/scratch/tzerweck/datathon/ETH_Datathon_2026_embeddings`
- Branch: `exp/text-embeddings`
- Goal: evaluate whether text embeddings or transformer-style models add signal beyond the deterministic parser baseline without destabilizing the stable submission workflow on `main`.

## Working assumptions

- The dataset has only about `1000` labeled train sessions, so fully fine-tuning a transformer is high risk.
- Headlines are synthetic and templated, so cheap text methods may already be strong.
- The hard problem is relevance: each session usually contains `3-4` companies, and only some headlines are likely tied to the traded stock.
- Company names repeat across train, public test, and private test, so company identity is a valid feature.

## Decision rule

- Keep `main` submission-ready.
- Merge reusable parser or feature-table improvements early.
- Keep dependency-heavy embedding or transformer experiments isolated on this branch until they show clear CV value.
- Do not add deep-learning dependencies until a lighter text baseline is competitive.

## Experiment ladder

### E0. Parser baseline sanity

Purpose:
- Reproduce the current text-only parser baseline and make sure all future experiments compare against the same session-level CV.

Deliverables:
- `docs/experiment_log.md` entry for current text-only baseline
- saved OOF under `outputs/oof/`

Stop/go:
- If the parser baseline is already weak and unstable across folds, treat text as an auxiliary feature source, not the main alpha path.

### E1. TF-IDF baseline

Purpose:
- Establish the strongest cheap text benchmark before adding embeddings.

Input variants:
- full headline text
- event-only text with company name stripped
- session-concatenated text
- company-grouped text within session

Aggregation variants:
- one concatenated document per session
- one concatenated document per company, then keep `top1`, `top2`, and weighted pooled summaries

Model family:
- linear regression or ridge-style model
- optional sign model if regression is unstable

Files to touch:
- `src/features/headline_parser.py`
- `src/features/headlines.py`
- `src/pipelines/train_text.py`
- add `src/models/text_linear.py` if the current baseline model becomes too overloaded

Stop/go:
- If TF-IDF does not beat the parser baseline or at least match it with lower variance, do not rush into transformer work.

### E2. Frozen sentence embeddings

Purpose:
- Test whether semantic smoothing helps beyond exact template matching.

Encoder options:
- MiniLM / E5 / BGE class models
- start with frozen encoders only

Input variants:
- full headline embeddings
- event-only embeddings with company removed

Aggregation variants:
- mean-pooled session embedding
- recency-weighted mean
- company-level pooled embeddings

Extra metadata to join:
- mention share
- recency share
- number of bars covered
- parser polarity counts
- relevance entropy across companies

Files to touch:
- add `src/features/text_embeddings.py`
- extend `src/pipelines/train_text.py`
- add cached matrices under `outputs/` or another gitignored cache path if needed

Stop/go:
- If mean-pooled session embeddings are weak, do not conclude embeddings fail. Move to company-aware pooling before dropping the idea.

### E3. Company-aware pooled embeddings

Purpose:
- Attack the real problem: noisy multi-company sessions.

Representation:
- parse headlines into `(company, body, event metadata, bar_ix)`
- embed each headline body
- pool by company within session
- build session features from:
  - `top1_company_embedding`
  - `top2_company_embedding`
  - relevance-weighted pooled embedding
  - company-level mention and recency features

Relevance weights:
- mention share
- recency-weighted mention share
- bar coverage
- parser polarity concentration
- optional short-horizon price reaction after each headline

Why this matters:
- naive session-level embedding pooling likely washes out the signal
- company-aware pooling gives the model a way to represent candidate stocks explicitly

Stop/go:
- If this does not improve over E1 and E2, transformer attention is unlikely to rescue the text-only path quickly enough for the hackathon.

### E4. Attention / MIL model

Purpose:
- Try one serious transformer-style relevance model only if earlier steps justify it.

Recommended shape:
- frozen headline encoder
- train a small attention head over headlines or companies
- predict session return and uncertainty

Avoid first:
- end-to-end fine-tuning of BERT-style encoders on concatenated session text
- large multimodal transformer stacks

Stop/go:
- Only continue if E3 is promising and the environment can support the dependencies without derailing the team.

## Team split on this branch

### Headlines owner

Owns:
- `src/features/headline_parser.py`
- `src/features/headlines.py`
- `src/features/text_embeddings.py`
- text preprocessing, parser cleanup, TF-IDF inputs, embedding caches

First tasks:
- add company-stripped event text output
- add per-session and per-company text aggregation helpers
- keep parser metadata available to downstream models

### Modeling and uncertainty owner

Owns:
- `src/models/text_linear.py` if added
- `src/evaluation/`
- `src/pipelines/train_text.py`

First tasks:
- run consistent CV for all text variants
- compare regression vs sign-style outputs
- design uncertainty from relevance ambiguity and model dispersion
- control position sizing and OOF tracking

### Pricing owners

Owns:
- price work on separate branches or worktrees

Interaction with this branch:
- only later for hybrid experiments
- do not block price iteration on text dependencies

## First 2-hour execution plan

1. Log the current parser baseline in `docs/experiment_log.md`.
2. Add event-only text extraction in `src/features/headline_parser.py`.
3. Add TF-IDF-ready aggregation helpers in `src/features/headlines.py`.
4. Run E1 on:
   - session concatenation, full text
   - session concatenation, stripped event text
   - company-aware concatenation with `top1` and `top2`
5. Compare CV Sharpe and fold variance against E0.
6. Only if E1 is competitive, prepare frozen embedding infrastructure.

## Success criteria

- Text-only beats or matches the parser baseline with better stability.
- Company-aware aggregation beats naive session pooling.
- Any embedding path must show value before new heavy dependencies are added.
- Hybrid integration with price should happen only after the text-only stack is internally validated.
