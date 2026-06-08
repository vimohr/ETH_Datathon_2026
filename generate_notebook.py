import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# Dumb Benchmark Model: Log-Price Only
This notebook uses a minimalist approach: only the log of the final closing price from the seen data.
It optimizes the Sharpe ratio directly using Ridge Regression with Automatic Differentiation (PyTorch)."""

nb['cells'].append(nbf.v4.new_markdown_cell(text))

code1 = """import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
"""
nb['cells'].append(nbf.v4.new_code_cell(code1))

text2 = """## 1. Load Data"""
nb['cells'].append(nbf.v4.new_markdown_cell(text2))

code2 = """train_seen = pd.read_parquet('data/bars_seen_train.parquet')
train_unseen = pd.read_parquet('data/bars_unseen_train.parquet')
test_seen_public = pd.read_parquet('data/bars_seen_public_test.parquet')
test_seen_private = pd.read_parquet('data/bars_seen_private_test.parquet')
test_seen = pd.concat([test_seen_public, test_seen_private], ignore_index=True)

print(f"Train Sessions: {train_seen['session'].nunique()}")
"""
nb['cells'].append(nbf.v4.new_code_cell(code2))

text3 = """## 2. Feature Engineering: Log of Last Closing Price"""
nb['cells'].append(nbf.v4.new_markdown_cell(text3))

code3 = """def extract_features(df):
    # Only one feature: log(close_49)
    # Since prices are normalized to 1.0 at bar 0, this is the session's log-return so far.
    grouped = df.groupby('session')
    last_close = grouped['close'].last()
    feat_df = pd.DataFrame(index=last_close.index)
    feat_df['log_last_close'] = np.log(last_close)
    return feat_df

X_train_raw = extract_features(train_seen)
X_test_raw = extract_features(test_seen)

print(f"Extracted feature: {X_train_raw.columns.tolist()}")
"""
nb['cells'].append(nbf.v4.new_code_cell(code3))

text4 = """## 3. Target Calculation"""
nb['cells'].append(nbf.v4.new_markdown_cell(text4))

code4 = """train_halfway_close = train_seen.groupby('session')['close'].last()
train_end_close = train_unseen.groupby('session')['close'].last()
y_train = (train_end_close / train_halfway_close) - 1.0

# Normalize features
X_train = (X_train_raw - X_train_raw.mean()) / X_train_raw.std()
X_test = (X_test_raw - X_train_raw.mean()) / X_train_raw.std()
X_train = X_train.loc[y_train.index]
"""
nb['cells'].append(nbf.v4.new_code_cell(code4))

text5 = """## 4. Ridge Regression with AD (PyTorch)"""
nb['cells'].append(nbf.v4.new_markdown_cell(text5))

code5 = """X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_pt = torch.tensor(X_tr.values, dtype=torch.float32)
y_pt = torch.tensor(y_tr.values, dtype=torch.float32)

class LinearSharpeModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze()

ad_model = LinearSharpeModel(X_pt.shape[1])
# weight_decay adds L2 penalty (Ridge)
optimizer = optim.Adam(ad_model.parameters(), lr=0.1, weight_decay=0.01)

for epoch in range(1001):
    optimizer.zero_grad()
    positions = ad_model(X_pt)
    pnl = positions * y_pt
    sharpe = torch.mean(pnl) / (torch.std(pnl) + 1e-9)
    loss = -sharpe
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Train Sharpe: {sharpe.item()*16:.4f}")

# Validation
with torch.no_grad():
    v_pos = ad_model(torch.tensor(X_val.values, dtype=torch.float32))
    v_pnl = v_pos.numpy() * y_val.values
    v_sharpe = np.mean(v_pnl) / np.std(v_pnl) * 16
    print(f"\\nValidation Sharpe: {v_sharpe:.4f}")
"""
nb['cells'].append(nbf.v4.new_code_cell(code5))

text6 = """## 5. Submission"""
nb['cells'].append(nbf.v4.new_markdown_cell(text6))

code6 = """X_test_pt = torch.tensor(X_test.values, dtype=torch.float32)
with torch.no_grad():
    test_preds = ad_model(X_test_pt).numpy()

submission = pd.DataFrame({
    'session': X_test.index,
    'target_position': test_preds * 1000
})

submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")
"""
nb['cells'].append(nbf.v4.new_code_cell(code6))

with open('generate_notebook.py', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
