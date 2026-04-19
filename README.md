# ETH Datathon 2026: Simulated Market Close Prediction

![Public Leaderboard](https://img.shields.io/badge/Public%20Leaderboard-2.709-0f766e?style=for-the-badge)
![Best Local CV](https://img.shields.io/badge/Best%20Local%20CV%20Sharpe-3.9637-1d4ed8?style=for-the-badge)

This repository contains our ETH Zurich Datathon 2026 hackathon solution for the simulated market close prediction challenge. The task was to choose a halfway-through position for each synthetic stock session using the seen price path and associated fictional news headlines.

The final submission reached a public leaderboard score of **2.709**. Our strongest local validation run reached a mean cross-validation Sharpe of **3.9637** with a compact forward-selected ridge model.

## Results Snapshot

| Metric | Value |
|--------|-------|
| Public leaderboard score | **2.709** |
| Best local CV Sharpe | **3.9637** |
| Best text-only public score | **0.71617** |
| Submission format | `20000` rows across `public_test` and `private_test` |

Supporting artifacts:

- [Experiment log](docs/experiment_log.md)
- [Experiment sweep leaderboard](outputs/reports/latest_experiment_sweep.csv)
- [Feature-selection subsets](outputs/reports/latest_feature_selection_subset_features.csv)
- [Submission registry](outputs/submissions/registry.jsonl)

## Final Approach

The most robust path was a compact price-first model rather than a large text-heavy stack.

- Build session-level features from the first half of the price path.
- Rank candidate features with out-of-fold Sharpe under time-consistent cross-validation.
- Keep a very small forward-selected subset to avoid unstable, redundant signals.
- Fit a regularized ridge model and convert predicted returns into target positions.
- Use headline features as an auxiliary research track rather than the final core model.

The strongest compact subset was:

- `price_technical__ret_last_20`
- `price__max_drawdown`
- `price__tail_return_10`
- `price__tail_return_5`
- `price__intrabar_range_mean`
- `price__trend_slope`
- `price__close_std`
- `price__seen_low_return`

## What We Learned

- Compact price features generalized better than broader parser and TF-IDF feature sets on the public board.
- Headline-derived signals were real in local validation, but they were less reliable than the best compact price-only models.
- Strong bookkeeping mattered: every submission is versioned, logged, and reproducible from the repo.

## Repository Guide

- `src/data/`: parquet loading, target construction, and CV folds
- `src/features/`: price, technical, parser, and text feature builders
- `src/models/`: linear, ridge, weighted models, and uncertainty logic
- `src/evaluation/`: Sharpe computation and validation helpers
- `src/pipelines/`: end-to-end training, feature selection, and submission entry points
- `configs/`: experiment configs, sweeps, and subset definitions
- `outputs/reports/`: sweep outputs and feature-selection reports
- `outputs/submissions/`: versioned CSV submissions and metadata

## Reproduce

From the repo root:

```bash
source .venv/bin/activate
make cv-baseline
make sweep-experiments SWEEP_CONFIG_GLOB="configs/experiments/*.json"
```

To rerun the compact forward-selection workflow:

```bash
.venv/bin/python -m src.pipelines.select_features \
  --no-core \
  --model ridge \
  --alpha 10 \
  --subset-strategy forward \
  --forward-score oof_sharpe \
  --candidate-block price \
  --candidate-block price_technical \
  --candidate-block headline_parser \
  --total-size 3 \
  --total-size 4 \
  --total-size 5 \
  --total-size 6 \
  --total-size 8
```

Full challenge details are available in [challenge_guide.md](challenge_guide.md).

## Slides

Add the final deck under `docs/slides/` and link it from this section before submission if you want the repo landing page to include the presentation directly.
