# Challenge Guide

## Data Setup & Overview

- Prepared notebook to help team members start working with the data easily
- Downloaded zip file from Kaggle Challenge and organized in data directory
- Data stored in Parquet format
- Public leaderboard should be deprioritized in evaluation strategy

## Data Structure

- Dataset contains approximately 6,000 trading sessions
- Each trading session represents a single stock traded over 100 days, with each day being one bar
- Every trading session consists of exactly 100 bars
- Each session has 5 bars with the following features:
    - Bar index
    - Last week's price
    - Open price
    - Close price
    - High price
    - Low price
- All bars end at bar number 45 in the visible portion

## Headlines Feature

- Generated headlines feature for each trading session
- Some headlines are relevant for a given session, others are not - participants must decide which ones matter
- Each bar/trading session has multiple headlines associated with it
- Headlines have consistent structure across sessions
- Headlines include fake company names (e.g., Relbos, Alvis, Yorvof)
- Company names remain the same due to data anonymization/obfuscation

## Prediction Task

- Must predict what happens in the second half of the session based on the first half
- Red line in visualization indicates split between training and test portions of each session
- Participants decide how many stocks/shares to buy for each position
- Can choose whether to incorporate headline/news data into predictions

## Scoring Methodology

- Evaluation uses Sharpe ratio, not just prediction accuracy
- Need to consider both correct predictions (which stocks go up/down) AND confidence levels
- Target allocations should be based on confidence in predictions
- Consistency of strategy matters alongside profitability

## Training Strategy Considerations

- Suggested approach: train on first 50 bars, test on remaining 50 bars
- For final model, consider training on all 100 bars
- Important to think about seen vs. unseen data splits in evaluation

## Possible Approaches

- Intuitive analysis of price data patterns
- Time series modeling approaches
- Sentiment extraction from headlines
- Parsing techniques to extract structured information from text
- Multiple valid methodologies can be explored
