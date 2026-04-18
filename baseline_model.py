import pandas as pd
import numpy as np

def train_and_predict(train_seen_path, train_unseen_path, test_seen_path):
    # Load data
    train_seen = pd.read_parquet(train_seen_path)
    train_unseen = pd.read_parquet(train_unseen_path)
    test_seen = pd.read_parquet(test_seen_path)
    
    # We want to predict if the return in the second half of the session is positive or negative
    # Calculate the return for the second half of the session in the training data
    # The return is close_end / close_halfway - 1
    
    # Get close halfway (last seen bar)
    train_halfway = train_seen.groupby('session').last()[['close']].rename(columns={'close': 'close_halfway'})
    
    # Get close end (last unseen bar)
    train_end = train_unseen.groupby('session').last()[['close']].rename(columns={'close': 'close_end'})
    
    # Calculate target (the return)
    train_targets = train_halfway.join(train_end)
    train_targets['target_return'] = train_targets['close_end'] / train_targets['close_halfway'] - 1
    
    # Let's build a very simple momentum baseline: 
    # Return of the first half predicts return of the second half
    
    # Get start price
    train_start = train_seen.groupby('session').first()[['close']].rename(columns={'close': 'close_start'})
    
    # Calculate first half return as feature
    train_features = train_start.join(train_halfway)
    train_features['momentum'] = train_features['close_halfway'] / train_features['close_start'] - 1
    
    # Prepare training data
    train_df = train_features.join(train_targets[['target_return']])
    
    # Very simple correlation check
    corr = train_df['momentum'].corr(train_df['target_return'])
    print(f"Correlation between first half return and second half return: {corr:.4f}")
    
    # Mean reversion or momentum?
    # Based on the sign of correlation, we can decide.
    # But let's build a simpler model: predict a position proportional to the first half return
    # or inversely proportional (mean reversion)
    
    # Process test data
    test_start = test_seen.groupby('session').first()[['close']].rename(columns={'close': 'close_start'})
    test_halfway = test_seen.groupby('session').last()[['close']].rename(columns={'close': 'close_halfway'})
    test_features = test_start.join(test_halfway)
    test_features['momentum'] = test_features['close_halfway'] / test_features['close_start'] - 1
    
    # Generate predictions
    # If correlation is positive, momentum. If negative, mean reversion.
    # To maximize sharpe, our position should be proportional to expected return / variance
    # Let's just predict position = alpha * momentum
    # Sign of alpha depends on correlation
    
    alpha = np.sign(corr) # Simple heuristic
    test_features['target_position'] = test_features['momentum'] * alpha * 100 # Scaling factor
    
    # Save predictions
    submission = test_features[['target_position']].reset_index()
    submission.to_csv('submission.csv', index=False)
    print(f"Created submission.csv with {len(submission)} rows.")
    return submission

if __name__ == "__main__":
    train_and_predict(
        'data/bars_seen_train.parquet',
        'data/bars_unseen_train.parquet',
        'data/bars_seen_public_test.parquet' # Example test set
    )
