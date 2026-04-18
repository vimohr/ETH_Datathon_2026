# CLAUDE.md

Working notes for Claude when assisting on the **ETH Datathon 2026 — Simulated Market Close Prediction** project.
For challenge framing read `README.md`, `Instructions.md`, and `challenge_guide.md`. This file captures *project-specific conventions* that are easy to miss.

---

## Working style

- **Think before changing modeling logic.** State assumptions explicitly before changing targets, features, validation, or position sizing. If a request is ambiguous (research vs implementation, CV optimization vs LB probing, seen-only features vs leakage), clarify instead of guessing.
- **Prefer the smallest change that can win.** Choose the minimum code needed to improve CV robustness, fix data integrity issues, or make the workflow reproducible. Avoid speculative abstractions, general frameworks, or "future-proofing" that this repo does not need yet.
- **Make surgical edits.** Do not clean up unrelated notebooks, legacy root-level scripts, or adjacent modules while doing a focused task. If you notice unrelated issues or dead code, mention them instead of changing them.
- **Define verification up front.** Before implementing, name the check you will use: session-level CV (`--cv-only`), fold-level Sharpe inspection, no-leakage reasoning, submission validation, artifact path checks, or schema/shape assertions.
- **Prefer evidence over intuition.** For modeling changes, compare against baseline CV Sharpe and inspect fold stability, not just one mean improvement or a public leaderboard bump.
- **Close the loop on artifacts.** If a run creates a result worth keeping, update `docs/experiment_log.md`. If it creates a submission, always route it through `src/submission.py::save_submission`.

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
    latest_competition.csv                   # combined public+private Kaggle upload
    registry.jsonl                           # append-only audit trail
  oof/{model_name}_oof.csv                   # for blending + diagnostics
  models/                                    # serialized models if/when we save them
  figures/                                   # plots
docs/
  experiment_log.md                          # log every CV/LB result here
  team_notes.md                              # ownership + priorities
```

When creating a submission, **always** go through `src/submission.py::save_submission` — it writes the CSV, the JSON metadata sidecar, the `latest_<split>.csv` alias, and appends to `registry.jsonl`.
The Kaggle upload itself should come from a combined `latest_competition.csv`, not from the legacy root `submission.csv`.

## Running things

The repo runs on Windows but the Makefile uses Unix paths (`.venv/bin/python`). Two options:

```bash
# If using Git Bash / WSL with a Unix-style venv:
make cv-baseline
make baseline-public
make baseline-private
make combine-submission
make kaggle-submit SUBMISSION_MESSAGE="your message"
make kaggle-status

# Native Windows (PowerShell or cmd) — call the module directly:
.venv\Scripts\python.exe -m src.pipelines.train_baseline --cv-only
.venv\Scripts\python.exe -m src.pipelines.train_baseline --test-split public_test
.venv\Scripts\python.exe -m src.pipelines.train_baseline --test-split private_test
.venv\Scripts\python.exe -m src.pipelines.competition combine
.venv\Scripts\python.exe -m src.pipelines.competition submit --message "your message"
.venv\Scripts\python.exe -m src.pipelines.competition status
```

CLI flags on `train_baseline`:
- `--cv-only` — run 5-fold CV and write OOF only, no submission.
- `--test-split {public_test,private_test}` — which split to score.
- `--include-headlines` — join headline features (Make targets do **not** pass this; use the module call).
- `--output PATH` — override submission path; default is `outputs/submissions/{timestamp}_{model_name}_{split}.csv`.

Competition submission rules:
- Kaggle expects one uploaded CSV with `20000` rows: `10000` public-test sessions (`1000..10999`) plus `10000` private-test sessions (`11000..20999`).
- Use `src.pipelines.competition combine` to merge split submissions safely and validate the full session coverage.
- Use `src.pipelines.competition status` instead of the raw Kaggle CLI when debugging failures, because it prints `error_description`.

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
- Competition ops: `src/pipelines/competition.py`, `src/kaggle_utils.py`, combined Kaggle uploads, status checks.
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
