"""
All models for the Sharpe-ratio AD benchmark.

Every model class follows the same contract:
    __init__(self, n_features)   – called by pipeline with feature count
    .fit(X_df, y_series)         – train on pandas data
    .predict(X_df) -> np.array   – return position predictions

The Sharpe AD loss is:  loss = -mean(pos * ret) / (std(pos * ret) + eps)

BENCHMARK RESULTS (3 rounds, 500+ configs tested):
  Best: Deep h=8,4 do=0.3 wd=0.15 lr=0.02  →  val Sharpe 3.36
  Key: early stopping + heavy L2 + dropout solved overfitting
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
    Wraps any nn.Module and trains it by backpropagating through the
    Sharpe ratio.  Uses an internal early-stopping split to prevent
    overfitting.
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
def _make_linear(n):
    return nn.Linear(n, 1)

def _make_shallow(n, hidden, dropout):
    return nn.Sequential(
        nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )

def _make_deep(n, h1, h2, dropout):
    return nn.Sequential(
        nn.Linear(n, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, 1),
    )

def _make_bottleneck(n, neck, expand, dropout):
    return nn.Sequential(
        nn.Linear(n, neck), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(neck, expand), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(expand, neck), nn.ReLU(),
        nn.Linear(neck, 1),
    )


# ──────────────────────────────────────────────────────────────────────
# Pre-configured model classes (best hyperparams from 500+ config sweep)
# ──────────────────────────────────────────────────────────────────────

# 🥇 #1  Val Sharpe 3.36, gap 0.12
class BestDeep84:
    """Deep MLP h=(8,4), dropout=0.3, wd=0.15, lr=0.02"""
    def __init__(self, n_features):
        net = _make_deep(n_features, 8, 4, 0.3)
        self._trainer = SharpeTrainer(net, lr=0.02, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# 🥈 #2  Val Sharpe 3.24, gap -0.58
class BestBottleneck320:
    """Bottleneck neck=3, expand=20, dropout=0.3, wd=0.15"""
    def __init__(self, n_features):
        net = _make_bottleneck(n_features, 3, 20, 0.3)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# 🥉 #3  Val Sharpe 3.23, gap 1.38
class BestShallow12:
    """Shallow MLP h=12, dropout=0.4, wd=0.15"""
    def __init__(self, n_features):
        net = _make_shallow(n_features, 12, 0.4)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# #4  Val Sharpe 3.16, gap -0.24
class BestDeep164:
    """Deep MLP h=(16,4), dropout=0.4, wd=0.2, lr=0.005"""
    def __init__(self, n_features):
        net = _make_deep(n_features, 16, 4, 0.4)
        self._trainer = SharpeTrainer(net, lr=0.005, weight_decay=0.2)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# #5  Val Sharpe 3.15, gap -0.53
class BestBottleneck516:
    """Bottleneck neck=5, expand=16, dropout=0.4, wd=0.15"""
    def __init__(self, n_features):
        net = _make_bottleneck(n_features, 5, 16, 0.4)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# Linear Ridge baseline  Val Sharpe ~2.72
class LinearRidge:
    """Simple linear model with L2 regularization."""
    def __init__(self, n_features):
        net = _make_linear(n_features)
        self._trainer = SharpeTrainer(net, lr=0.1, weight_decay=0.01)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)


# ──────────────────────────────────────────────────────────────────────
# Ensemble: averages predictions from the top-5 models
# ──────────────────────────────────────────────────────────────────────
class EnsembleTop5:
    """Trains top-5 models independently and averages their predictions."""
    def __init__(self, n_features):
        self.models = [
            BestDeep84(n_features),
            BestBottleneck320(n_features),
            BestShallow12(n_features),
            BestDeep164(n_features),
            BestBottleneck516(n_features),
        ]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "🥇 Deep(8,4) do=0.3 wd=0.15":    BestDeep84,
    "🥈 Bottle(3,20) do=0.3 wd=0.15": BestBottleneck320,
    "🥉 Shallow(12) do=0.4 wd=0.15":  BestShallow12,
    "#4 Deep(16,4) do=0.4 wd=0.2":    BestDeep164,
    "#5 Bottle(5,16) do=0.4 wd=0.15": BestBottleneck516,
    "Linear Ridge":                    LinearRidge,
    "Ensemble Top-5":                  EnsembleTop5,
}


# ──────────────────────────────────────────────────────────────────────
# Backward-compatible aliases
# ──────────────────────────────────────────────────────────────────────
LinearSharpeModel = LinearRidge

class LinearSharpeSuperModel:
    """Legacy wrapper for run_pipeline.ipynb compatibility."""
    def __init__(self, init_data):
        self.model = LinearRidge(init_data)
    def __call__(self, X):
        return self.model.predict(X)
    def train(self, X, y):
        self.model.fit(X, y)