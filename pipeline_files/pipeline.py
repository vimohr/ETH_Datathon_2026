"""
pipeline.py — K-fold cross-validation pipeline for the Datathon.

OVERVIEW
--------
This is the central orchestration layer.  It connects:
    all_features.py (feature engineering)  →  all_models.py (model training)

The main function `split_and_train_pipeline` handles:
    1. Loading raw parquet files (OHLC bars)
    2. Calling the feature extractor to produce a feature matrix
    3. Computing the target: return over the unseen half
    4. Running K-fold CV with a fresh model per fold
    5. Reporting per-fold and averaged Sharpe ratios

EVALUATION METRIC
-----------------
The competition metric is the Sharpe ratio:

    pnl_i  = position_i × (close_end_i / close_halfway_i − 1)
    sharpe = mean(pnl) / std(pnl) × 16

We multiply by 16 (≈ sqrt(256)) as a conventional annualisation factor.
The `calc_sharpe` function implements this formula.

WHY K-FOLD CV?
--------------
With only 1000 training sessions, a single 80/20 split is noisy — the
validation Sharpe depends heavily on which 200 sessions land in the val
set.  5-fold CV averages over 5 different splits, giving a more stable
estimate of out-of-sample performance (plus a standard deviation).

USAGE
-----
    from pipeline import split_and_train_pipeline
    from all_models import BestDeep84
    from all_features import extract_features

    model, train_data, val_data = split_and_train_pipeline(
        BestDeep84, extract_features, data_dir='../data', n_folds=5
    )
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
import warnings

warnings.filterwarnings('ignore')


def calc_sharpe(positions, returns):
    """
    Compute the competition Sharpe ratio.

    Parameters
    ----------
    positions : array-like — predicted position sizes (long > 0, short < 0)
    returns   : array-like — actual returns (close_end / close_halfway − 1)

    Returns
    -------
    float — Sharpe ratio × 16 (annualised)
    """
    positions = np.array(positions).flatten()
    returns = np.array(returns).flatten()
    pnl = positions * returns
    return np.mean(pnl) / (np.std(pnl) + 1e-9) * 16


def _train_and_eval(model, X_train, y_train, X_val, y_val):
    """
    Train a single model instance and evaluate it.

    Supports models with either .fit() or .train() for training,
    and either .predict() or __call__() for inference.

    Returns
    -------
    (train_sharpe, val_sharpe) : tuple of floats
    """
    # Train the model (try .fit() first, then .train())
    if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
        model.fit(X_train, y_train)
    elif hasattr(model, 'train') and callable(getattr(model, 'train')):
        model.train(X_train, y_train)
    else:
        raise ValueError("Model has no .fit() or .train() method!")

    # Get predictions (try .predict() first, then direct __call__)
    if hasattr(model, 'predict'):
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
    elif hasattr(model, '__call__'):
        import torch
        train_preds = model(torch.tensor(X_train.values, dtype=torch.float32)).detach().numpy()
        val_preds = model(torch.tensor(X_val.values, dtype=torch.float32)).detach().numpy()
    else:
        raise ValueError("Model has no .predict() or __call__ method!")

    return (
        calc_sharpe(train_preds, y_train.values),
        calc_sharpe(val_preds, y_val.values),
    )


def split_and_train_pipeline(model_class, extract_features_func,
                              data_dir='../data', n_folds=5):
    """
    Full pipeline: load data → extract features → K-fold CV → report.

    Parameters
    ----------
    model_class : callable
        A class (not an instance!) that accepts n_features as its only
        constructor argument.  Must have .fit(X, y) and .predict(X).
        Example: BestDeep84 from all_models.py.

    extract_features_func : callable
        A function that takes a DataFrame of bars and returns a feature
        DataFrame.  Example: extract_features from all_features.py.

    data_dir : str
        Path to the directory containing the parquet files.

    n_folds : int
        Number of cross-validation folds.  Default 5.

    Returns
    -------
    (model, train_data, val_data) : tuple
        - model: the last fold's trained model instance
        - train_data: (X_train, y_train) from the last fold
        - val_data: (X_val, y_val) from the last fold
    """

    # ── Step 1: Load raw OHLC bar data ──
    print("1. Loading raw chunk data...")
    train_seen = pd.read_parquet(f'{data_dir}/bars_seen_train.parquet')
    train_unseen = pd.read_parquet(f'{data_dir}/bars_unseen_train.parquet')

    # ── Step 2: Extract features from the seen half (bars 0-49) ──
    print(f"2. Extracting {train_seen['session'].nunique()} sequences...")
    X = extract_features_func(train_seen)

    # ── Step 3: Compute targets from the unseen half (bars 50-99) ──
    # Target = return if you buy at close of bar 49 and sell at close of bar 99
    print("3. Indexing actual targets...")
    train_halfway_close = train_seen.groupby('session')['close'].last()
    train_end_close = train_unseen.groupby('session')['close'].last()
    y = (train_end_close / train_halfway_close) - 1.0

    # Ensure X and y have the same sessions
    X = X.loc[y.index]

    # ── Step 4: K-fold cross-validation ──
    print(f"4. Running {n_folds}-fold cross-validation on {len(X)} samples, "
          f"{X.shape[1]} features...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_sharpes = []
    val_sharpes = []
    last_model = None
    last_train_data = None
    last_val_data = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fresh model instance for each fold (clean weights)
        model = model_class(X.shape[1])

        tr_sharpe, va_sharpe = _train_and_eval(
            model, X_train, y_train, X_val, y_val
        )
        train_sharpes.append(tr_sharpe)
        val_sharpes.append(va_sharpe)

        print(f"  Fold {fold}/{n_folds}:  "
              f"train={tr_sharpe:+.4f}  val={va_sharpe:+.4f}  "
              f"gap={tr_sharpe - va_sharpe:+.4f}")

        last_model = model
        last_train_data = (X_train, y_train)
        last_val_data = (X_val, y_val)

    # ── Step 5: Report averaged metrics ──
    avg_train = np.mean(train_sharpes)
    avg_val = np.mean(val_sharpes)
    std_val = np.std(val_sharpes)  # variance across folds = estimation noise

    print(f"\n{'='*50}")
    print(f"  AVG TRAIN SHARPE:      {avg_train:+.4f}")
    print(f"  AVG VALIDATION SHARPE: {avg_val:+.4f} ± {std_val:.4f}")
    print(f"  AVG GAP:               {avg_train - avg_val:+.4f}")
    print(f"{'='*50}\n")

    return last_model, last_train_data, last_val_data


if __name__ == "__main__":
    from all_models import BestDeep84
    from all_features import extract_features

    print("Running 5-fold CV with BestDeep84...")
    trained_model, train_data, val_data = split_and_train_pipeline(
        BestDeep84, extract_features
    )
    print("Done!")
