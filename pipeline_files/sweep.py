#!/usr/bin/env python3
"""
Round 2: Systematic hyperparameter sweep with early stopping.

Key anti-overfitting strategies:
  1. Early stopping on validation Sharpe (hold-out from training set)
  2. Much heavier weight_decay (L2 regularization)
  3. Higher dropout rates
  4. Fewer parameters in hidden layers
  5. Gradient clipping
"""

import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from all_features import extract_features
from pipeline import calc_sharpe


# ──────────────────────────────────────────────────────────────────────
# Base trainer with EARLY STOPPING on a held-out validation slice
# ──────────────────────────────────────────────────────────────────────
class SharpeTrainer:
    """
    Trains any nn.Module by maximising Sharpe via AD.
    Splits training data internally into fit/early-stop sets.
    """

    def __init__(self, net, lr=0.05, weight_decay=0.01, epochs=1000,
                 patience=80, es_fraction=0.25, grad_clip=1.0):
        self.net = net
        self.scaler = StandardScaler()
        self.lr = lr
        self.wd = weight_decay
        self.epochs = epochs
        self.patience = patience        # epochs without improvement before stop
        self.es_fraction = es_fraction  # fraction of training data for early-stop
        self.grad_clip = grad_clip

    def fit(self, X, y):
        X_np = self.scaler.fit_transform(X)

        # Internal early-stopping split
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_np, y.values, test_size=self.es_fraction, random_state=99
        )

        X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
        y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
        X_es_t = torch.tensor(X_es, dtype=torch.float32)
        y_es_t = torch.tensor(y_es, dtype=torch.float32)

        optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.wd
        )

        best_val_sharpe = -1e9
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            # --- train step ---
            self.net.train()
            optimizer.zero_grad()
            pos = self.net(X_fit_t).squeeze(-1)
            pnl = pos * y_fit_t
            loss = -torch.mean(pnl) / (torch.std(pnl) + 1e-6)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            optimizer.step()

            # --- early-stopping check ---
            self.net.eval()
            with torch.no_grad():
                es_pos = self.net(X_es_t).squeeze(-1)
                es_pnl = es_pos * y_es_t
                es_sharpe = (torch.mean(es_pnl) / (torch.std(es_pnl) + 1e-6)).item()

            if es_sharpe > best_val_sharpe:
                best_val_sharpe = es_sharpe
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                break

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.stopped_epoch = epoch + 1
        self.best_es_sharpe = best_val_sharpe

    def predict(self, X):
        X_np = self.scaler.transform(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(X_t).squeeze(-1).numpy()


# ──────────────────────────────────────────────────────────────────────
# Architecture factories
# ──────────────────────────────────────────────────────────────────────
def make_linear(n):
    return nn.Linear(n, 1)

def make_shallow(n, hidden=16, dropout=0.3):
    return nn.Sequential(
        nn.Linear(n, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )

def make_deep(n, h1=16, h2=8, dropout=0.4):
    return nn.Sequential(
        nn.Linear(n, h1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h2, 1),
    )

def make_bottleneck(n, neck=4, expand=16, dropout=0.3):
    return nn.Sequential(
        nn.Linear(n, neck),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(neck, expand),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(expand, neck),
        nn.ReLU(),
        nn.Linear(neck, 1),
    )

def make_tanh_bounded(n, hidden=8, dropout=0.3):
    return nn.Sequential(
        nn.Linear(n, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
        nn.Tanh(),
    )


# ──────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# ──────────────────────────────────────────────────────────────────────
CONFIGS = []

# --- Linear: sweep weight_decay and lr ---
for lr in [0.05, 0.1, 0.2]:
    for wd in [0.001, 0.01, 0.05, 0.1]:
        CONFIGS.append({
            "name": f"Linear lr={lr} wd={wd}",
            "factory": lambda n, _lr=lr: make_linear(n),
            "lr": lr, "wd": wd, "epochs": 800, "patience": 100,
        })

# --- Shallow MLP: sweep hidden size, dropout, weight_decay ---
for hidden in [8, 16]:
    for dropout in [0.3, 0.5, 0.7]:
        for wd in [0.05, 0.1, 0.2]:
            for lr in [0.01, 0.02]:
                CONFIGS.append({
                    "name": f"Shallow h={hidden} do={dropout} wd={wd} lr={lr}",
                    "factory": lambda n, _h=hidden, _d=dropout: make_shallow(n, _h, _d),
                    "lr": lr, "wd": wd, "epochs": 800, "patience": 80,
                })

# --- Deep MLP with heavy regularisation ---
for h1, h2 in [(16, 8), (8, 4)]:
    for dropout in [0.4, 0.6]:
        for wd in [0.1, 0.2]:
            CONFIGS.append({
                "name": f"Deep h={h1},{h2} do={dropout} wd={wd}",
                "factory": lambda n, _h1=h1, _h2=h2, _d=dropout: make_deep(n, _h1, _h2, _d),
                "lr": 0.01, "wd": wd, "epochs": 1000, "patience": 80,
            })

# --- Bottleneck with narrow neck ---
for neck in [3, 4, 6]:
    for dropout in [0.3, 0.5]:
        for wd in [0.05, 0.1]:
            CONFIGS.append({
                "name": f"Bottleneck neck={neck} do={dropout} wd={wd}",
                "factory": lambda n, _nk=neck, _d=dropout: make_bottleneck(n, _nk, 16, _d),
                "lr": 0.01, "wd": wd, "epochs": 800, "patience": 80,
            })

# --- Tanh bounded (forced [-1,1] output) ---
for hidden in [4, 8]:
    for dropout in [0.3, 0.5]:
        for wd in [0.05, 0.1]:
            CONFIGS.append({
                "name": f"Tanh h={hidden} do={dropout} wd={wd}",
                "factory": lambda n, _h=hidden, _d=dropout: make_tanh_bounded(n, _h, _d),
                "lr": 0.02, "wd": wd, "epochs": 800, "patience": 80,
            })


print(f"Total configurations to test: {len(CONFIGS)}")


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────
def run_sweep(data_dir="../data"):
    # Load data once
    print("=" * 60)
    print("  LOADING DATA & FEATURES")
    print("=" * 60)

    train_seen = pd.read_parquet(f"{data_dir}/bars_seen_train.parquet")
    train_unseen = pd.read_parquet(f"{data_dir}/bars_unseen_train.parquet")

    X = extract_features(train_seen)
    train_halfway = train_seen.groupby("session")["close"].last()
    train_end = train_unseen.groupby("session")["close"].last()
    y = (train_end / train_halfway) - 1.0
    X = X.loc[y.index]

    # Outer train/val split (this val set is NEVER seen during training)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    n_features = X_train.shape[1]
    print(f"Features: {n_features}  |  Train: {len(X_train)}  |  Val: {len(X_val)}\n")

    rows = []
    for i, cfg in enumerate(CONFIGS):
        net = cfg["factory"](n_features)
        model = SharpeTrainer(
            net, lr=cfg["lr"], weight_decay=cfg["wd"],
            epochs=cfg["epochs"], patience=cfg["patience"],
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_sharpe = calc_sharpe(model.predict(X_train), y_train.values)
        val_sharpe = calc_sharpe(model.predict(X_val), y_val.values)

        rows.append({
            "config": cfg["name"],
            "train_sharpe": round(train_sharpe, 4),
            "val_sharpe": round(val_sharpe, 4),
            "gap": round(train_sharpe - val_sharpe, 4),
            "stopped_epoch": model.stopped_epoch,
            "time_s": round(elapsed, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(CONFIGS)}] latest: {cfg['name']}  "
                  f"val={val_sharpe:.4f}  gap={train_sharpe - val_sharpe:.4f}")

    # Sort and print
    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    df.index += 1

    print("\n")
    print("=" * 90)
    print("                HYPERPARAMETER SWEEP LEADERBOARD (top 25)")
    print("=" * 90)
    print(df.head(25).to_string())
    print("=" * 90)

    # Bottom 5 for reference
    print("\n--- Worst 5 ---")
    print(df.tail(5).to_string())

    df.to_csv("sweep_results.csv", index_label="rank")
    print(f"\nFull results ({len(df)} configs) saved to sweep_results.csv")

    return df


if __name__ == "__main__":
    run_sweep()
