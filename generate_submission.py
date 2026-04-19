#!/usr/bin/env python3
"""
Generate submission.csv using the winning Deep(8,4) model.

Trains on ALL training data (no validation holdout),
predicts on public_test + private_test, outputs submission.csv.
"""

import numpy as np
import pandas as pd
import sys, os

# Add pipeline_files to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline_files"))

from pipeline_files.all_features import extract_features
from pipeline_files.all_models import WinnerBottlePro

DATA_DIR = "data"

# ── 1. Load training data ──
print("Loading training data...")
train_seen = pd.read_parquet(f"{DATA_DIR}/bars_seen_train.parquet")
train_unseen = pd.read_parquet(f"{DATA_DIR}/bars_unseen_train.parquet")

# ── 2. Extract features & targets for training ──
print("Extracting training features...")
X_train = extract_features(train_seen)

train_halfway_close = train_seen.groupby("session")["close"].last()
train_end_close = train_unseen.groupby("session")["close"].last()
y_train = (train_end_close / train_halfway_close) - 1.0
X_train = X_train.loc[y_train.index]

print(f"  Training samples: {len(X_train)}, features: {X_train.shape[1]}")

# ── 3. Train the winning model on ALL training data ──
print("Training winning model (WinnerBottlePro)...")
model = WinnerBottlePro(X_train.shape[1])
model.fit(X_train, y_train)

# ── 4. Load & predict on test sets ──
print("Loading test data...")
public_test_seen = pd.read_parquet(f"{DATA_DIR}/bars_seen_public_test.parquet")
private_test_seen = pd.read_parquet(f"{DATA_DIR}/bars_seen_private_test.parquet")

print("Extracting test features...")
X_public = extract_features(public_test_seen)
X_private = extract_features(private_test_seen)

print(f"  Public test sessions:  {len(X_public)}")
print(f"  Private test sessions: {len(X_private)}")

public_preds = model.predict(X_public)
private_preds = model.predict(X_private)

# ── 5. Scale positions (Sharpe is scale-invariant, but use reasonable size) ──
# Multiply by 1000 for reasonable position sizes
scale = 1000.0
public_positions = public_preds * scale
private_positions = private_preds * scale

# ── 6. Build submission DataFrame ──
sub_public = pd.DataFrame({
    "session": X_public.index,
    "target_position": public_positions,
})
sub_private = pd.DataFrame({
    "session": X_private.index,
    "target_position": private_positions,
})

submission = pd.concat([sub_public, sub_private], ignore_index=True)
submission = submission.sort_values("session").reset_index(drop=True)

# ── 7. Validate ──
expected_public = set(range(1000, 11000))
expected_private = set(range(11000, 21000))
actual_sessions = set(submission["session"].values)

assert expected_public.issubset(actual_sessions), \
    f"Missing public sessions: {expected_public - actual_sessions}"
assert expected_private.issubset(actual_sessions), \
    f"Missing private sessions: {expected_private - actual_sessions}"
assert len(submission) == 20000, f"Expected 20000 rows, got {len(submission)}"

# ── 8. Save ──
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved to submission.csv")
print(f"  Total rows: {len(submission)}")
print(f"  Sessions: {submission['session'].min()} .. {submission['session'].max()}")
print(f"  Position range: [{submission['target_position'].min():.2f}, {submission['target_position'].max():.2f}]")
print(f"  Mean position: {submission['target_position'].mean():.4f}")
print(f"  Std position:  {submission['target_position'].std():.4f}")
print("\nDone!")
