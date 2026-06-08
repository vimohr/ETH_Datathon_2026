import pandas as pd

bars_seen = pd.read_parquet('data/bars_seen_train.parquet')
print("Bars Seen Info:")
print(bars_seen.info())
print("\nBars Seen Head:")
print(bars_seen.head())

bars_unseen = pd.read_parquet('data/bars_unseen_train.parquet')
print("\nBars Unseen Info:")
print(bars_unseen.info())
print("\nBars Unseen Head:")
print(bars_unseen.head())

try:
    headlines_seen = pd.read_parquet('data/headlines_seen_train.parquet')
    print("\nHeadlines Seen Info:")
    print(headlines_seen.info())
    print("\nHeadlines Seen Head:")
    print(headlines_seen.head())
except Exception as e:
    print(e)
