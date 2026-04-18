
# Zurich Datathon 2026: Simulated Market Close Prediction

## Challenge

Each data file contains many sessions. Each session simulates a single synthetic
stock trading over a number of time bars. You are given OHLC (Open, High, Low,
Close) bar data and a mix of news headlines for each session.

- Training sessions: you see all bars.
- Test sessions: you see only the first half of the trading session.

Your task: Decide how much stock to buy/sell half-way through each session.

## Submission

A CSV file with two columns:

```
session,target_position
500,42.5
501,-56.92043
...
```

One row per session. The `session` values must match those in the test data.

The `target_position` specifies how many shares of the stock you wish to buy or
sell, in a fictional currency. At the half-way point of each session (the end of
the seen data), we will buy/sell this amount of stock at the close price of that
bar. We will hold it until the end of the session, where we will sell/buy it
back at the close price of the last bar.

## Metric

We will compute the Sharpe ratio of your trading strategy across the test sessions:

pnl_i = target_position_i * (close_price_end_i / close_price_halfway_i - 1)
sharpe = np.mean(pnl_i) / np.std(pnl_i) * 16

Your final score is the Sharpe ratio as defined above.

## Data

### Files

The filenames follow the naming schema:

data/{data_kind}_{seen/unseen}_{train/public_test/private_test}.parquet

You receive the seen & unseen data for train. You receive only the seen data for
the test sets.

### File format

All files contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `session` | int | Session identifier |
| `bar_ix` | int | Bar number - larger = later in the session |

The OHLC files additionally have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `open` | float | Open price of the bar |
| `high` | float | High price of the bar |
| `low` | float | Low price of the bar |
| `close` | float | Close price of the bar |

The headline files additionally have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `headline` | string | Text of the news headline associated with the bar |

### About the headlines

Headlines are fictional and mention various fictional companies. None of the
mentioned companies are real. None of the headlines are real.

### About prices

Prices have been normalized to begin at 1.

## Tips

It may be challenging to attain a large relative improvement over a simple
baseline. You should be mindful about how to you evaluate your model during
development.

Not all headlines in a given session are necessarily relevant to that session.

## Loading the data

```python
import pandas as pd

seen_train_bars = pd.read_parquet("data/bars_seen_train.parquet")
# etc.
```

## Working Structure

For the hackathon, use the modular workflow under `src/` and `outputs/`.
The existing root-level scripts and notebook are still available as legacy references.

- `src/data/`: parquet loading, target construction, CV folds
- `src/features/`: price and headline feature builders
- `src/models/`: return models and uncertainty-based sizing
- `src/evaluation/`: Sharpe and cross-validation helpers
- `src/pipelines/`: runnable training / submission entry points
- `outputs/submissions/`: generated CSV files for upload
- `outputs/oof/`: out-of-fold predictions for blending and diagnostics

The submission flow now also writes:

- a versioned submission filename
- a sidecar JSON metadata file
- `outputs/submissions/latest_<split>.csv` as the current handoff artifact
- `outputs/submissions/registry.jsonl` as a simple audit trail

## Verified Split

The parquet files use an exact halfway split:

- seen bars: `bar_ix 0..49`
- unseen training bars: `bar_ix 50..99`

That means positions are effectively taken at the close of bar `49` and held to the close of bar `99`.

## Quickstart

From the repo root:

```bash
source .venv/bin/activate
make cv-baseline
make baseline-public
make baseline-private
```

To include simple headline features in the baseline:

```bash
.venv/bin/python -m src.pipelines.train_baseline --test-split public_test --include-headlines
```
