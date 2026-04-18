# ETH Datathon 2026 Hackathon Notes

## Goal

Predict a halfway `target_position` per session to maximize out-of-sample Sharpe.

Verified split:

- seen bars: `bar_ix 0..49`
- unseen train bars: `bar_ix 50..99`

Positions are effectively taken at the close of bar `49` and held to the close of bar `99`.

## Team Split

- Pricing 1
  Own `src/features/price.py` and the main price-only feature set.
- Pricing 2
  Own alternative price hypotheses, quick notebook experiments, and candidate features to upstream.
- Modeling and uncertainty
  Own `src/models/`, `src/evaluation/`, position sizing, calibration, and CV interpretation.
- Headlines
  Own `src/features/headlines.py`, headline parsing, relevance filtering, and text features.
- Person 4 later
  Own blending, submission ops, deck/demo support, and whichever workstream is the current bottleneck.

## Repo Workflow

- Shared code lives in `src/`
- Fast experiments live in `notebooks/`
- Generated submissions go to `outputs/submissions/`
- OOF predictions go to `outputs/oof/`
- Experiment tracking lives in `docs/experiment_log.md`

Root-level legacy files like `baseline_model.py` and `analyze_data.ipynb` can be used as references, but the active workflow should move through `src/`.

## Commands

From repo root:

```bash
source .venv/bin/activate
make cv-baseline
make baseline-public
make baseline-private
```

Include simple headline features:

```bash
.venv/bin/python -m src.pipelines.train_baseline --test-split public_test --include-headlines
```

## Submission Handoff

Current submission flow writes:

- versioned CSV files in `outputs/submissions/`
- matching metadata JSON files
- `latest_public_test.csv` or `latest_private_test.csv` as the current handoff artifact
- `registry.jsonl` as the submission log
