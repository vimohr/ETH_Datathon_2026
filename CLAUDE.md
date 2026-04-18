# CLAUDE.md

Working notes for Claude when assisting on the **ETH Datathon 2026 — Simulated Market Close Prediction** project.
For challenge framing read `README.md`, `Instructions.md`, and `challenge_guide.md`. This file captures *project-specific conventions* that are easy to miss.

---

## What we are predicting

- One row per `session` in the test set: `target_position` (float, fictional shares).
- Position is opened at the close of bar `49` and closed at the close of bar `99` (exact halfway split).
- Score = `sharpe = mean(pnl) / std(pnl) * 16`, where `pnl_i = target_position_i * (close_end_i / close_halfway_i - 1)`.
- Sharpe is scale-invariant, so the absolute size of `target_position` does not matter — only its sign and *relative* magnitude across sessions.

## Data layout

All parquet files live in `data/` with naming `{kind}_{visibility}_{split}.parquet`:

| kind | visibility | splits available |
|------|------------|------------------|
| `bars` | `seen` | `train`, `public_test`, `private_test` |
| `bars` | `unseen` | `train` only |
| `headlines` | `seen` | `train`, `public_test`, `private_test` |
| `headlines` | `unseen` | `train` only |

- `seen` covers `bar_ix 0..49`, `unseen` covers `bar_ix 50..99`.
- ~6,000 sessions total. Headlines are noisy — many are irrelevant to their session; company names (Relbos, Alvis, Yorvof, …) are consistent across sessions.
- **Always** load via `src.data.load.load_bars` / `load_headlines`. Do not hardcode parquet paths.

## Codebase map (the canonical workflow lives in `src/`)

```
src/
  paths.py           # ROOT, DATA_DIR, OUTPUTS_DIR, ensure_output_dirs()
  settings.py        # RANDOM_SEED=42, CV_FOLDS=5, ANNUALIZATION_FACTOR=16, POSITION_CLIP=100, ...
  submission.py      # build/validate/save submissions + JSON metadata + registry.jsonl
  data/
    load.py          # parquet loaders
    splits.py        # session-level k-fold (shuffled, seeded)
    targets.py       # build_train_targets -> close_halfway, close_end, target_return
  features/
    price.py         # build_price_features (returns, vol, drawdown, trend slope, ...)
    headlines.py     # build_headline_features (counts + simple keyword regex buckets)
  models/
    baseline.py      # LinearBaselineModel (np.linalg.lstsq, residual-std uncertainty)
    uncertainty.py   # size_positions: score = mu/sigma, percentile-normalized, clipped to ±POSITION_CLIP
  evaluation/
    metrics.py       # pnl, sharpe_from_positions
    validation.py    # run_cross_validation -> (fold_summary, oof_predictions)
  pipelines/
    train_baseline.py  # the entry point invoked by the Makefile
```

Legacy / scratch (do not extend, treat as references): `baseline_model.py`, `examine_data.py`, `generate_notebook.py`, `analyze_data.ipynb`, `eda.ipynb`, root-level `submission.csv`.

## Outputs (generated, gitignored except registry/log)

```
outputs/
  submissions/
    {timestamp}_{model_name}_{split}.csv     # versioned, with .json sidecar
    latest_{split}.csv                       # current handoff artifact
    registry.jsonl                           # append-only audit trail
  oof/{model_name}_oof.csv                   # for blending + diagnostics
  models/                                    # serialized models if/when we save them
  figures/                                   # plots
docs/
  experiment_log.md                          # log every CV/LB result here
  team_notes.md                              # ownership + priorities
```

When creating a submission, **always** go through `src/submission.py::save_submission` — it writes the CSV, the JSON metadata sidecar, the `latest_<split>.csv` alias, and appends to `registry.jsonl`.

## Running things

The repo runs on Windows but the Makefile uses Unix paths (`.venv/bin/python`). Two options:

```bash
# If using Git Bash / WSL with a Unix-style venv:
make cv-baseline
make baseline-public
make baseline-private

# Native Windows (PowerShell or cmd) — call the module directly:
.venv\Scripts\python.exe -m src.pipelines.train_baseline --cv-only
.venv\Scripts\python.exe -m src.pipelines.train_baseline --test-split public_test
.venv\Scripts\python.exe -m src.pipelines.train_baseline --test-split private_test
```

CLI flags on `train_baseline`:
- `--cv-only` — run 5-fold CV and write OOF only, no submission.
- `--test-split {public_test,private_test}` — which split to score.
- `--include-headlines` — join headline features (Make targets do **not** pass this; use the module call).
- `--output PATH` — override submission path; default is `outputs/submissions/{timestamp}_{model_name}_{split}.csv`.

## Conventions

- **Sessions are the unit of analysis.** Group by `session` before computing features; CV folds split on sessions, never on bars.
- **Never leak unseen bars into features.** Test sets only ever ship `seen` data; mirror that in training (use only `bars_seen_train` for features, `bars_unseen_train` only to compute `target_return`).
- **Validation > public LB.** `challenge_guide.md` says the public leaderboard should be *deprioritized*. Trust 5-fold CV Sharpe; only one shot on private LB.
- **Reproducibility:** seed everything from `src.settings.RANDOM_SEED`. CV folds are seeded (`RANDOM_SEED=42`).
- **Position scale doesn't change Sharpe.** Don't tune `POSITION_CLIP` for score; tune it only if a downstream consumer cares.
- **Headlines are noisy.** Many headlines in a session are irrelevant. Filtering / relevance scoring is an open problem — don't assume keyword counts are signal without checking CV.

## Logging experiments

Every meaningful run should add a row to `docs/experiment_log.md` with: timestamp, owner, features, model, CV Sharpe, public LB (if scored), notes. The submission `.json` sidecar + `registry.jsonl` already capture model name, feature count, CV Sharpe, and git commit — use those when filling in the log.

## Team ownership (from `docs/team_notes.md`)

- Infrastructure: `src/data/`, `src/submission.py`, `Makefile`, output paths.
- Pricing: `src/features/price.py`, price EDA, session-level heuristics.
- Modeling & uncertainty: `src/models/`, `src/evaluation/`, position sizing, calibration.
- Headlines: `src/features/headlines.py`, parsing, relevance filtering.

When making changes, stay inside the relevant area unless the task explicitly crosses boundaries.

## Things to avoid

- Editing or rerunning the legacy root-level scripts as if they were the workflow.
- Hardcoding parquet paths or duplicating loader logic — extend `src/data/load.py` instead.
- Writing CSVs directly into `outputs/submissions/` — go through `save_submission`.
- Shuffling at the bar level in CV (must be session-level).
- Tuning hyperparameters against the public leaderboard.
- Using `--no-verify` or destructive git operations without explicit user approval.
