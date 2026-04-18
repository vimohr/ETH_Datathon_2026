import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
import warnings

warnings.filterwarnings('ignore')


### Here the function starts!!!!!

def calc_sharpe(positions, returns):
    positions = np.array(positions).flatten()
    returns = np.array(returns).flatten()
    pnl = positions * returns
    return np.mean(pnl) / (np.std(pnl) + 1e-9) * 16


def _train_and_eval(model, X_train, y_train, X_val, y_val):
    """Train a single model and return (train_sharpe, val_sharpe)."""
    if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
        model.fit(X_train, y_train)
    elif hasattr(model, 'train') and callable(getattr(model, 'train')):
        model.train(X_train, y_train)
    else:
        raise ValueError("Model has no .fit() or .train() method!")

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
    Modular pipeline with K-fold cross-validation.

    1. Loads the parquet training data
    2. Builds features via extract_features_func
    3. Runs n_folds CV: for each fold, instantiates a fresh model,
       trains it, and records train/val Sharpe
    4. Reports per-fold and averaged Sharpe ratios
    5. Returns the last fold's trained model + its train/val data
    """

    print("1. Loading raw chunk data...")
    train_seen = pd.read_parquet(f'{data_dir}/bars_seen_train.parquet')
    train_unseen = pd.read_parquet(f'{data_dir}/bars_unseen_train.parquet')

    print(f"2. Extracting {train_seen['session'].nunique()} sequences...")
    X = extract_features_func(train_seen)

    print("3. Indexing actual targets...")
    train_halfway_close = train_seen.groupby('session')['close'].last()
    train_end_close = train_unseen.groupby('session')['close'].last()
    y = (train_end_close / train_halfway_close) - 1.0
    X = X.loc[y.index]

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

        # Fresh model instance for each fold
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

    avg_train = np.mean(train_sharpes)
    avg_val = np.mean(val_sharpes)
    std_val = np.std(val_sharpes)

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
