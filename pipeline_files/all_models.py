"""
all_models.py — PyTorch models optimised for Sharpe ratio via autodiff.

OVERVIEW
--------
This module defines neural network models that are trained by directly
maximising the Sharpe ratio using automatic differentiation (AD).

Unlike traditional ML models that minimise MSE or cross-entropy, our custom
loss function is:

    loss = -mean(positions * returns) / (std(positions * returns) + eps)

This is the negative Sharpe ratio — minimising it *maximises* the Sharpe.
Because PyTorch tracks gradients through this entire expression, Adam can
directly optimise the trading signal for the evaluation metric.

KEY DESIGN DECISIONS
--------------------
1. **StandardScaler**: Features are z-scored before entering the network.
   Without this, features like RSI (~50) would dominate returns (~0.001),
   making the learned weights uninterpretable and training unstable.

2. **Early Stopping**: The training data is internally split 75/25. Training
   continues until the Sharpe on the 25% holdout hasn't improved for
   `patience` epochs.  This was the single biggest fix for overfitting
   (reduced overfit gap from 30+ to <1).

3. **Heavy L2 Regularisation** (weight_decay=0.15): Penalises large weights
   to prevent the model from fitting noise.  Found via grid search over
   [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3].

4. **Small Architectures**: The winning models have only 8→4 or 3→20→3
   hidden units.  With 800-1000 training samples, larger networks memorise
   noise and produce negative validation Sharpe.

5. **Gradient Clipping**: Capped at 1.0 to avoid exploding gradients,
   especially early in training when the Sharpe denominator is small.

MODEL CONTRACT
--------------
Every model class follows the same API so pipeline.py can use them:

    model = ModelClass(n_features)   # instantiate with feature count
    model.fit(X_df, y_series)        # train on pandas data
    preds = model.predict(X_df)      # returns np.array of positions

BENCHMARK RESULTS (3 rounds, 500+ configs tested)
--------------------------------------------------
    🥇 Deep(8,4) do=0.3 wd=0.15  →  val Sharpe 3.36
    🥈 Bottle(3,20) do=0.3 wd=0.15  →  val Sharpe 3.24
    🥉 Shallow(12) do=0.4 wd=0.15  →  val Sharpe 3.23
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

    Workflow:
        1. StandardScaler normalises features (fit on train, transform on test)
        2. Internal 75/25 split for early stopping
        3. Adam optimiser with weight_decay (L2 penalty = Ridge regression)
        4. Each epoch:  forward pass → compute Sharpe → backprop → clip grads
        5. After patience epochs without improvement, restore best weights

    Parameters
    ----------
    net : nn.Module
        The PyTorch architecture to train.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation strength (= Ridge penalty).
    epochs : int
        Maximum training epochs (usually stopped early).
    patience : int
        Early stopping patience — epochs without improvement before stopping.
    es_fraction : float
        Fraction of training data held out for early stopping (not validation!).
    grad_clip : float
        Maximum gradient norm (prevents exploding gradients).
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
        """
        Train the model by maximising the Sharpe ratio via AD.

        Parameters
        ----------
        X : pd.DataFrame   — feature matrix (n_samples × n_features)
        y : pd.Series       — target returns (close_end / close_halfway - 1)
        """
        # Step 1: Normalise features to zero mean, unit variance
        X_np = self.scaler.fit_transform(X)

        # Step 2: Internal early-stopping split (separate from CV folds!)
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_np, y.values, test_size=self.es_fraction, random_state=99
        )
        X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
        y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
        X_es_t = torch.tensor(X_es, dtype=torch.float32)
        y_es_t = torch.tensor(y_es, dtype=torch.float32)

        # Step 3: Adam with L2 penalty (weight_decay = Ridge regression)
        optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.wd
        )

        best_val_sharpe = -1e9
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            # --- Forward pass on training portion ---
            self.net.train()
            optimizer.zero_grad()
            pos = self.net(X_fit_t).squeeze(-1)    # predicted positions
            pnl = pos * y_fit_t                     # PnL per session
            # Negative Sharpe = our loss (minimise this = maximise Sharpe)
            loss = -torch.mean(pnl) / (torch.std(pnl) + 1e-6)
            loss.backward()                         # AD: compute gradients
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            optimizer.step()

            # --- Early stopping check on held-out portion ---
            self.net.eval()
            with torch.no_grad():
                es_pos = self.net(X_es_t).squeeze(-1)
                es_pnl = es_pos * y_es_t
                es_sharpe = (torch.mean(es_pnl) / (torch.std(es_pnl) + 1e-6)).item()

            if es_sharpe > best_val_sharpe:
                best_val_sharpe = es_sharpe
                # Snapshot the best weights
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break  # no improvement for `patience` epochs → stop

        # Restore the best weights found during training
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.stopped_epoch = epoch + 1
        self.best_es_sharpe = best_val_sharpe

    def predict(self, X):
        """
        Predict position sizes for new data.

        Parameters
        ----------
        X : pd.DataFrame — same feature columns as training data.

        Returns
        -------
        np.ndarray — predicted position per session (positive = long, negative = short).
        """
        X_np = self.scaler.transform(X)  # use the scaler fitted during .fit()
        X_t = torch.tensor(X_np, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(X_t).squeeze(-1).numpy()


# ──────────────────────────────────────────────────────────────────────
# Architecture factories
#
# These build raw nn.Module networks.  They are wrapped by the model
# classes below, which pair them with a SharpeTrainer for fitting.
# ──────────────────────────────────────────────────────────────────────
def _make_linear(n):
    """Single linear layer: n_features → 1 output (Ridge regression)."""
    return nn.Linear(n, 1)

def _make_shallow(n, hidden, dropout):
    """One hidden layer with ReLU + dropout: n → hidden → 1."""
    return nn.Sequential(
        nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )

def _make_deep(n, h1, h2, dropout):
    """Two hidden layers with ReLU + dropout: n → h1 → h2 → 1."""
    return nn.Sequential(
        nn.Linear(n, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, 1),
    )

def _make_bottleneck(n, neck, expand, dropout):
    """
    Bottleneck: compress → expand → compress → output.
    n → neck → expand → neck → 1
    Forces the model to learn a low-dimensional representation.
    """
    return nn.Sequential(
        nn.Linear(n, neck), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(neck, expand), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(expand, neck), nn.ReLU(),
        nn.Linear(neck, 1),
    )


# ──────────────────────────────────────────────────────────────────────
# Pre-configured model classes
#
# Each class pairs an architecture with the best hyperparameters found
# during the 500+ config grid search (see sweep.py, sweep_round3.py).
# They all follow the same API: __init__(n_features), .fit(X,y), .predict(X)
# ──────────────────────────────────────────────────────────────────────

# 🥇 #1  Val Sharpe 3.36, gap 0.12
class BestDeep84:
    """Deep MLP h=(8,4), dropout=0.3, wd=0.15, lr=0.02."""
    def __init__(self, n_features):
        net = _make_deep(n_features, 8, 4, 0.3)
        self._trainer = SharpeTrainer(net, lr=0.02, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# 🥈 #2  Val Sharpe 3.24, gap -0.58
class BestBottleneck320:
    """Bottleneck neck=3, expand=20, dropout=0.3, wd=0.15."""
    def __init__(self, n_features):
        net = _make_bottleneck(n_features, 3, 20, 0.3)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# 🥉 #3  Val Sharpe 3.23, gap 1.38
class BestShallow12:
    """Shallow MLP h=12, dropout=0.4, wd=0.15."""
    def __init__(self, n_features):
        net = _make_shallow(n_features, 12, 0.4)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# #4  Val Sharpe 3.16, gap -0.24
class BestDeep164:
    """Deep MLP h=(16,4), dropout=0.4, wd=0.2, lr=0.005."""
    def __init__(self, n_features):
        net = _make_deep(n_features, 16, 4, 0.4)
        self._trainer = SharpeTrainer(net, lr=0.005, weight_decay=0.2)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# #5  Val Sharpe 3.15, gap -0.53
class BestBottleneck516:
    """Bottleneck neck=5, expand=16, dropout=0.4, wd=0.15."""
    def __init__(self, n_features):
        net = _make_bottleneck(n_features, 5, 16, 0.4)
        self._trainer = SharpeTrainer(net, lr=0.01, weight_decay=0.15)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)

# Linear Ridge baseline  Val Sharpe ~2.72
class LinearRidge:
    """Single linear layer with L2 regularisation — simplest possible model."""
    def __init__(self, n_features):
        net = _make_linear(n_features)
        self._trainer = SharpeTrainer(net, lr=0.1, weight_decay=0.01)
    def fit(self, X, y): self._trainer.fit(X, y)
    def predict(self, X): return self._trainer.predict(X)


# ──────────────────────────────────────────────────────────────────────
# Ensemble: averages predictions from the top-5 models
#
# Ensembling reduces variance (each model sees slightly different
# early-stopping checkpoints).  However, in our tests the ensemble
# didn't consistently beat the single best model due to the small
# dataset size.
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
# Registry: maps human-readable names → model classes.
# Used by benchmark.py and validate_all.py to loop over all models.
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
# Backward-compatible aliases (for run_pipeline.ipynb)
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