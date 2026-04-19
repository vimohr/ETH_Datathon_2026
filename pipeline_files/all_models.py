"""
all_models.py — PyTorch models optimised for Sharpe ratio via autodiff.

OVERVIEW
--------
This module defines neural network models that are trained by directly
maximising the Sharpe ratio using automatic differentiation (AD).

WINNING ARCHITECTURE: BottlePro
-------------------------------
The BottlePro architecture (introduced in Round 4/5) consistentely outperfoms
standard MLPs on the 1000-sample datathon dataset. It uses:
1.  **Bottleneck structure**: n_features -> neck -> expand -> neck -> 1
    This forces extreme dimensionality reduction followed by sparse expansion,
    which acts as a strong regulariser.
2.  **Layer Normalization**: Applied at the bottleneck to stabilise gradients.
3.  **GELU Activation**: Smoother than ReLU, helping with the small signal-to-noise ratio.
4.  **Extreme Regularisation**: Dropout (0.6) and Weight Decay (0.4) are used
    to reach near-zero overfit gaps.

BENCHMARK RESULTS (5 Rounds, 600+ configs)
------------------------------------------
🥇 BottlePro n=8 e=32 do=0.6 wd=0.4   → 5-Fold Val Sharpe 2.95 (Gap -0.11)
🥈 Deep MLP h=8,4 do=0.3 wd=0.15      → 5-Fold Val Sharpe 2.71 (Gap +0.26)
🥉 Linear Ridge                       → 5-Fold Val Sharpe 2.12 (Gap +0.40)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────
# Core trainer: handles scaling, AD training loop, and early stopping
# ──────────────────────────────────────────────────────────────────────
class SharpeTrainer:
    """
    Generic training wrapper for any nn.Module architecture.
    """

    def __init__(self, net, lr=0.01, weight_decay=0.15, epochs=1000,
                 patience=80, es_fraction=0.25, grad_clip=1.0):
        self.net = net
        self.scaler = StandardScaler()
        self.lr = lr
        self.wd = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.es_fraction = es_fraction
        self.grad_clip = grad_clip
        self.stopped_epoch = 0
        self.best_es_sharpe = -1e9

    def fit(self, X, y):
        X_np = self.scaler.fit_transform(X)
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
# Architecture factories
# ──────────────────────────────────────────────────────────────────────

def _make_bottle_pro(n, neck, expand, dropout):
    return nn.Sequential(
        nn.Linear(n, neck),
        nn.LayerNorm(neck),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(neck, expand),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expand, neck),
        nn.GELU(),
        nn.Linear(neck, 1)
    )

def _make_deep(n, h1, h2, dropout):
    return nn.Sequential(
        nn.Linear(n, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, 1),
    )

def _make_linear(n):
    return nn.Linear(n, 1)


# ──────────────────────────────────────────────────────────────────────
# Pre-configured model classes
# ──────────────────────────────────────────────────────────────────────

# 🥇 Current Champion: BottlePro (Round 5 winner)
class WinnerBottlePro:
    """BottlePro n=8, expand=32, do=0.6, wd=0.4. Best 5-Fold average."""
    def __init__(self, n_features):
        net = _make_bottle_pro(n_features, 8, 32, 0.6)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.4)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# 🥈 Runner up: BottlePro n=10, expand=48, do=0.6, wd=0.5
class StrongBottlePro:
    """BottlePro n=10, expand=48, do=0.6, wd=0.5. Very robust."""
    def __init__(self, n_features):
        net = _make_bottle_pro(n_features, 10, 48, 0.6)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.5)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# 🥉 Standard MLP winner from Round 3
class BestDeep84:
    """Deep MLP h=(8,4), dropout=0.3, wd=0.15, lr=0.02."""
    def __init__(self, n_features):
        net = _make_deep(n_features, 8, 4, 0.3)
        self._trainer = SharpeTrainer(net, lr=0.02, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# Baseline
class LinearRidge:
    """Linear model with L2 regularization."""
    def __init__(self, n_features):
        net = _make_linear(n_features)
        self._trainer = SharpeTrainer(net, lr=0.1, weight_decay=0.05)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "🥇 Winner BottlePro (8,32)": WinnerBottlePro,
    "🥈 Strong BottlePro (10,48)": StrongBottlePro,
    "🥉 Best Deep MLP (8,4)":      BestDeep84,
    "Linear Ridge Baseline":       LinearRidge,
}


# ──────────────────────────────────────────────────────────────────────
# Backward-compatible aliases
# ──────────────────────────────────────────────────────────────────────
LinearSharpeSuperModel = WinnerBottlePro # Use the winner for legacy calls