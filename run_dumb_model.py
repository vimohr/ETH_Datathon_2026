import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 1. Load Data
# We use the raw OHLC bars from the seen and unseen splits.
train_seen = pd.read_parquet('data/bars_seen_train.parquet')
train_unseen = pd.read_parquet('data/bars_unseen_train.parquet')
test_seen_public = pd.read_parquet('data/bars_seen_public_test.parquet')
test_seen_private = pd.read_parquet('data/bars_seen_private_test.parquet')
test_seen = pd.concat([test_seen_public, test_seen_private], ignore_index=True)

# 2. Extract Features: Log of Last Closing Price
# This captures the price level at the halfway mark.
def extract_features(df):
    grouped = df.groupby('session')
    last_close = grouped['close'].last()
    feat_df = pd.DataFrame(index=last_close.index)
    feat_df['log_last_close'] = np.log(last_close)
    return feat_df

X_train_raw = extract_features(train_seen)
X_test_raw = extract_features(test_seen)

# 3. Target Calculation
# The target is the return strictly over the UNSEEN part of the session.
train_halfway_close = train_seen.groupby('session')['close'].last()
train_end_close = train_unseen.groupby('session')['close'].last()
y = (train_end_close / train_halfway_close) - 1.0

# 4. Normalization
# Scaling is critical for linear models/neural nets to ensure importance isn't scale-dependent.
X_train = (X_train_raw - X_train_raw.mean()) / X_train_raw.std()
X_test = (X_test_raw - X_train_raw.mean()) / X_train_raw.std()
X_train = X_train.loc[y.index]

# 5. Train Model: Ridge Regression with AD
# We use PyTorch to directly optimize the Sharpe Ratio.
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

X_pt = torch.tensor(X_tr.values, dtype=torch.float32)
y_pt = torch.tensor(y_tr.values, dtype=torch.float32)

class LinearSharpeModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze()

ad_model = LinearSharpeModel(X_pt.shape[1])
# weight_decay implements the 'Ridge' L2 penalty.
optimizer = optim.Adam(ad_model.parameters(), lr=0.1, weight_decay=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    pos = ad_model(X_pt)
    pnl = pos * y_pt
    sharpe = torch.mean(pnl) / (torch.std(pnl) + 1e-9)
    # We minimize negative Sharpe to maximize positive Sharpe.
    (-sharpe).backward()
    optimizer.step()

# 6. Predict and Generate Submission
X_test_pt = torch.tensor(X_test.values, dtype=torch.float32)
with torch.no_grad():
    preds = ad_model(X_test_pt).numpy()

# Scale up for reasonable position sizes
submission = pd.DataFrame({
    'session': X_test.index,
    'target_position': preds * 1000
})
submission.to_csv('submission_dumb.csv', index=False)

# Final stats on Validation Set
with torch.no_grad():
    v_pos = ad_model(torch.tensor(X_val.values, dtype=torch.float32))
    v_pnl = v_pos.numpy() * y_val.values
    v_sharpe = np.mean(v_pnl) / np.std(v_pnl) * 16
    print(f"--- Dumb Model Results ---")
    print(f"Validation Sharpe: {v_sharpe:.4f}")
    print(f"Learned Weight:    {ad_model.linear.weight.item():.4f}")
    print(f"Learned Bias:      {ad_model.linear.bias.item():.4f}")
    print(f"\nGenerated submission_dumb.csv")
