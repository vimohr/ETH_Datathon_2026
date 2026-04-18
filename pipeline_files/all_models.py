import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple linear model
class LinearSharpeModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        # Initialize with slight positive weight on the 'seen_return' feature
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, 0] = 1.0  # (0 is now seen_return if sorted)
            self.linear.bias.zero_()
        
    def forward(self, x):
        return self.linear(x).squeeze()

    def fit(self, x, y):
        optimizer = optim.Adam(self.parameters(), lr=0.1, weight_decay=0.01)
        print("Training PyTorch AD Model to directly maximize Sharpe...")
        # Training loop
        epochs = 500
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass: predict target positions
            positions = self(x)
            
            # Calculate PnL per session
            pnl = positions * y
            
            # AD-compatible pseudo-Sharpe Loss (Minimize negative Sharpe)
            sharpe = torch.mean(pnl) / (torch.std(pnl) + 1e-6)
            loss = -sharpe
            
            # Automatic Differentiation MAGIC! Differentiates the final Sharpe ratio calculation.
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss (Negative Sharpe): {loss.item():.4f}")

class LinearSharpeSuperModel:
    def __init__(self, init_data):
        self.model = LinearSharpeModel(init_data)
    def __call__(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.values, dtype=torch.float32)
        return self.model(X)
    def train(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.values, dtype=torch.float32)
            y = torch.tensor(y.values, dtype=torch.float32)
        self.model.fit(X, y)