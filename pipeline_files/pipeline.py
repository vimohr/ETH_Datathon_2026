import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

def split_and_train_pipeline(model_class, extract_features_func, data_dir='../data'):
    """
    Modular pipeline that:
    1. Loads the parquet training bounds
    2. Builds exhaustive TS features
    3. Aligns targets and formats scaled features
    4. Automatically binds the external model (.train() or .fit())
    5. Calculates pure Evaluation metrics natively mapped on Train vs Validation blocks!
    """
    
    print("1. Loading raw chunk data...")
    train_seen = pd.read_parquet(f'{data_dir}/bars_seen_train.parquet')
    train_unseen = pd.read_parquet(f'{data_dir}/bars_unseen_train.parquet')
    
    print(f"2. Extracting {train_seen['session'].nunique()} sequences (Takes ~30 seconds)...")
    X = extract_features_func(train_seen)
    model = model_class(X.shape[1])
    
    print("3. Indexing actual targets...")
    train_halfway_close = train_seen.groupby('session')['close'].last()
    train_end_close = train_unseen.groupby('session')['close'].last()
    y = (train_end_close / train_halfway_close) - 1.0
    
    # Align X and y perfectly
    X = X.loc[y.index]
    
    print(f"4. Slicing into Split Folds -> Train (80%) | Val (20%)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"5. Engaging Model.train() across {len(X_train)} samples...")
    # Supports models that expose a .train() or .fit()
    if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
        model.fit(X_train, y_train)
    elif hasattr(model, 'train') and callable(getattr(model, 'train')):
        model.train(X_train, y_train)
    else:
        raise ValueError("Model passed does not have a recognizable .train() or .fit() method!")
        
    print("6. Parsing Evaluation Vectors...")
    if hasattr(model, 'predict'):
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
    elif hasattr(model, 'forward') or hasattr(model, '__call__'): # PyTorch-like fallback
        import torch
        train_preds = model(torch.tensor(X_train.values, dtype=torch.float32)).detach().numpy()
        val_preds = model(torch.tensor(X_val.values, dtype=torch.float32)).detach().numpy()
    else:
        raise ValueError("Model passed does not have a recognizable .predict() method!")
        
    # Evaluate Sharpe Metrics
    train_sharpe = calc_sharpe(train_preds, y_train.values)
    val_sharpe = calc_sharpe(val_preds, y_val.values)
    
    print(f"\\n=====================================")
    print(f" FINAL TRAIN SHARPE:      {train_sharpe:.4f}")
    print(f" FINAL VALIDATION SHARPE: {val_sharpe:.4f}")
    print(f"=====================================\\n")
    
    # Export bounds
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)
    
    return model, train_data, val_data

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    
    print("Executing quick internal pipeline baseline test with Sklearn...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    trained_model, train_data, val_data = split_and_train_pipeline(rf_model)
    print("Pipeline standalone functionality tested. Ready for external import!")
