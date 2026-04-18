#!/usr/bin/env python3
"""Quick final validation of all models in MODEL_REGISTRY."""
import time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from all_features import extract_features
from all_models import MODEL_REGISTRY
from pipeline import calc_sharpe

train_seen = pd.read_parquet('../data/bars_seen_train.parquet')
train_unseen = pd.read_parquet('../data/bars_unseen_train.parquet')
X = extract_features(train_seen)
y = (train_unseen.groupby('session')['close'].last()
     / train_seen.groupby('session')['close'].last()) - 1.0
X = X.loc[y.index]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
n = X_train.shape[1]
print(f'Features: {n} | Train: {len(X_train)} | Val: {len(X_val)}')

results = []
for name, Cls in MODEL_REGISTRY.items():
    print(f'  Training {name}...', end=' ', flush=True)
    t0 = time.time()
    model = Cls(n)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    tr = calc_sharpe(model.predict(X_train), y_train.values)
    va = calc_sharpe(model.predict(X_val), y_val.values)
    results.append((name, tr, va, tr - va, elapsed))
    print(f'done ({elapsed:.1f}s)')

print()
print('=' * 70)
print('FINAL VALIDATED LEADERBOARD')
print('=' * 70)
for name, tr, va, gap, t in sorted(results, key=lambda x: -x[2]):
    print(f'  {va:+7.4f} val | {tr:+7.4f} train | {gap:+7.4f} gap | {name}')
print('=' * 70)
