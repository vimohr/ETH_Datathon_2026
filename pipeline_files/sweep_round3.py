#!/usr/bin/env python3
"""
Round 3: Focused sweep around the best configs from round 2.

Top winners were:
  - Deep h=16,8 do=0.4 wd=0.2  → val 3.04 (gap -0.28)
  - Bottleneck neck=3-6         → val 2.97-2.99
  - Deep h=8,4 do=0.6 wd=0.1-0.2 → val 2.97

All share: heavy regularisation, early stopping at ~80-100 epochs,
NEGATIVE overfit gap (val > train), meaning we can push harder.

This round:
  1. Fine-grained sweep around the sweet spots
  2. Add ensemble of top models
  3. Multiple random seeds to check stability
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from all_features import extract_features
from pipeline import calc_sharpe


class SharpeTrainer:
    def __init__(self, net, lr=0.01, weight_decay=0.1, epochs=1000,
                 patience=80, es_fraction=0.25, grad_clip=1.0):
        self.net = net
        self.scaler = StandardScaler()
        self.lr = lr
        self.wd = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.es_fraction = es_fraction
        self.grad_clip = grad_clip

    def fit(self, X, y):
        X_np = self.scaler.fit_transform(X)
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_np, y.values, test_size=self.es_fraction, random_state=99
        )
        X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
        y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
        X_es_t = torch.tensor(X_es, dtype=torch.float32)
        y_es_t = torch.tensor(y_es, dtype=torch.float32)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        best_val_sharpe = -1e9
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self.net.train()
            optimizer.zero_grad()
            pos = self.net(X_fit_t).squeeze(-1)
            pnl = pos * y_fit_t
            loss = -torch.mean(pnl) / (torch.std(pnl) + 1e-6)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            optimizer.step()

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
# Architecture factories (around the winners)
# ──────────────────────────────────────────────────────────────────────
def make_deep(n, h1, h2, dropout):
    return nn.Sequential(
        nn.Linear(n, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, 1),
    )

def make_bottleneck(n, neck, expand, dropout):
    return nn.Sequential(
        nn.Linear(n, neck), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(neck, expand), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(expand, neck), nn.ReLU(),
        nn.Linear(neck, 1),
    )

def make_linear(n):
    return nn.Linear(n, 1)

def make_shallow(n, hidden, dropout):
    return nn.Sequential(
        nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


# ──────────────────────────────────────────────────────────────────────
# Focused configs
# ──────────────────────────────────────────────────────────────────────
CONFIGS = []

# --- Deep MLP: fine-tune around h=16,8 / do=0.4 / wd=0.2 ---
for h1, h2 in [(16, 8), (12, 6), (20, 10), (16, 4), (8, 4)]:
    for dropout in [0.3, 0.4, 0.5, 0.6]:
        for wd in [0.1, 0.15, 0.2, 0.3]:
            for lr in [0.005, 0.01, 0.02]:
                CONFIGS.append({
                    "name": f"Deep h={h1},{h2} do={dropout} wd={wd} lr={lr}",
                    "factory": lambda n, _h1=h1, _h2=h2, _d=dropout: make_deep(n, _h1, _h2, _d),
                    "lr": lr, "wd": wd,
                })

# --- Bottleneck: fine-tune around neck=3-6 / do=0.3-0.5 ---
for neck in [3, 4, 5, 6]:
    for expand in [12, 16, 20]:
        for dropout in [0.3, 0.4, 0.5]:
            for wd in [0.05, 0.1, 0.15]:
                CONFIGS.append({
                    "name": f"Bottle n={neck} e={expand} do={dropout} wd={wd}",
                    "factory": lambda n, _nk=neck, _e=expand, _d=dropout: make_bottleneck(n, _nk, _e, _d),
                    "lr": 0.01, "wd": wd,
                })

# --- Linear: a few more weight_decay values ---
for wd in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
    for lr in [0.05, 0.1, 0.15]:
        CONFIGS.append({
            "name": f"Linear wd={wd} lr={lr}",
            "factory": lambda n: make_linear(n),
            "lr": lr, "wd": wd,
        })

# --- Shallow MLP: near the sweet spots ---
for hidden in [8, 12, 16]:
    for dropout in [0.4, 0.5, 0.6, 0.7]:
        for wd in [0.1, 0.15, 0.2, 0.3]:
            CONFIGS.append({
                "name": f"Shallow h={hidden} do={dropout} wd={wd}",
                "factory": lambda n, _h=hidden, _d=dropout: make_shallow(n, _h, _d),
                "lr": 0.01, "wd": wd,
            })

print(f"Total configurations: {len(CONFIGS)}")


def run_sweep(data_dir="../data"):
    print("=" * 60)
    print("  ROUND 3: FOCUSED HYPERPARAMETER SWEEP")
    print("=" * 60)

    train_seen = pd.read_parquet(f"{data_dir}/bars_seen_train.parquet")
    train_unseen = pd.read_parquet(f"{data_dir}/bars_unseen_train.parquet")
    X = extract_features(train_seen)
    train_halfway = train_seen.groupby("session")["close"].last()
    train_end = train_unseen.groupby("session")["close"].last()
    y = (train_end / train_halfway) - 1.0
    X = X.loc[y.index]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    n_features = X_train.shape[1]
    print(f"Features: {n_features}  |  Train: {len(X_train)}  |  Val: {len(X_val)}\n")

    rows = []
    t_total = time.time()

    for i, cfg in enumerate(CONFIGS):
        net = cfg["factory"](n_features)
        model = SharpeTrainer(net, lr=cfg["lr"], weight_decay=cfg["wd"],
                              epochs=1000, patience=80)
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
            "time_s": round(elapsed, 2),
        })

        if (i + 1) % 50 == 0:
            elapsed_total = time.time() - t_total
            best_so_far = max(r["val_sharpe"] for r in rows)
            print(f"  [{i+1}/{len(CONFIGS)}]  best val so far: {best_so_far:.4f}  "
                  f"({elapsed_total:.0f}s elapsed)")

    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    df.index += 1

    print("\n")
    print("=" * 90)
    print("           ROUND 3 LEADERBOARD (top 30)")
    print("=" * 90)
    print(df.head(30).to_string())
    print("=" * 90)

    df.to_csv("sweep_round3_results.csv", index_label="rank")
    total_time = time.time() - t_total
    print(f"\nAll {len(df)} configs done in {total_time:.0f}s")
    print("Results saved to sweep_round3_results.csv")

    # --- Ensemble of top-5 ---
    print("\n\n--- ENSEMBLE TEST (average of top-5 models) ---")
    top5_cfgs = df.head(5)
    ensemble_preds_train = np.zeros(len(X_train))
    ensemble_preds_val = np.zeros(len(X_val))

    for _, row in top5_cfgs.iterrows():
        cfg_name = row["config"]
        # Re-train the top 5
        matching = [c for c in CONFIGS if c["name"] == cfg_name]
        if matching:
            c = matching[0]
            net = c["factory"](n_features)
            model = SharpeTrainer(net, lr=c["lr"], weight_decay=c["wd"],
                                  epochs=1000, patience=80)
            model.fit(X_train, y_train)
            ensemble_preds_train += model.predict(X_train)
            ensemble_preds_val += model.predict(X_val)

    ensemble_preds_train /= 5
    ensemble_preds_val /= 5

    ens_train_sharpe = calc_sharpe(ensemble_preds_train, y_train.values)
    ens_val_sharpe = calc_sharpe(ensemble_preds_val, y_val.values)
    print(f"  Ensemble Train Sharpe: {ens_train_sharpe:.4f}")
    print(f"  Ensemble Val Sharpe:   {ens_val_sharpe:.4f}")
    print(f"  Ensemble Gap:          {ens_train_sharpe - ens_val_sharpe:.4f}")

    return df


if __name__ == "__main__":
    run_sweep()
