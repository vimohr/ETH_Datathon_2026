"""
all_features.py — Feature engineering for the Zurich Datathon 2026.

OVERVIEW
--------
This module transforms raw OHLC bar data (50 bars per session, bars 0-49)
into a fixed-width feature vector per session.  The features are designed
to capture the *state* of the price process at the halfway point, so the
model can decide whether to go long or short for the second half.

The main entry point is `extract_features(df)`, which takes a DataFrame
of seen bars and returns a DataFrame indexed by session with ~20 features.

FEATURE CATEGORIES
------------------
1. Price-level features:  overall return, log-return, volatility
2. Momentum features:     trailing 5/10/20-bar returns
3. Technical indicators:  RSI-14, EMA crossovers, price-vs-EMA
4. Statistical moments:   skewness, kurtosis of bar-to-bar returns
5. Autocorrelation:       lag-1 and lag-5 serial correlation
6. Fourier features:      dominant frequencies in the price signal
7. Microstructure:        green-bar ratio, average bar spread

WHY THESE FEATURES?
-------------------
- All features are *stationary* (ratios, returns, bounded indicators)
  rather than raw prices, which avoids scale dependence.
- They compress the 50-bar time series into a compact summary that
  a simple model can learn from with only ~1000 training samples.
- If we had more data, we could feed raw bars into an LSTM/Transformer
  instead, but with 1000 samples these hand-crafted features work better.

USAGE
-----
    from all_features import extract_features
    X = extract_features(bars_seen_train)  # returns DataFrame (n_sessions × 20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
import warnings

warnings.filterwarnings('ignore')


def calculate_rsi(prices, window=14):
    """
    Relative Strength Index (RSI) — a momentum oscillator in [0, 100].

    RSI > 70 → overbought (price may reverse down)
    RSI < 30 → oversold  (price may reverse up)

    Uses the standard exponential-smoothing variant where up/down moves
    are averaged with a rolling window.  Returns the final RSI value
    for the given price array.
    """
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    if down == 0:
        rs = np.inf
    else:
        rs = up/down
    res = [100. - 100./(1. + rs)]
    for i in range(window, len(prices)-1):
        delta = deltas[i]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        if down == 0:
            rs = np.inf
        else:
            rs = up/down
        res.append(100. - 100./(1. + rs))
    if len(res) == 0:
        return 50.0  # neutral default
    return res[-1]


def extract_features(df):
    """
    Convert raw OHLC bar data → one feature row per session.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: session, open, high, low, close.
        Typically the "bars_seen_*.parquet" data (bars 0-49).

    Returns
    -------
    pd.DataFrame
        Indexed by session, with ~20 numeric feature columns.
    """
    grouped = df.groupby('session')

    # --- Vectorised price-level features (fast, no per-session loop) ---
    first_open = grouped['open'].first()   # price at bar 0
    last_close = grouped['close'].last()   # price at bar 49

    feat_df = pd.DataFrame(index=first_open.index)

    # Total return over the seen half: (close_49 / open_0) - 1
    feat_df['seen_return'] = (last_close / first_open) - 1.0

    # Log-return: mathematically symmetric for up/down moves
    feat_df['log_return'] = np.log(last_close) - np.log(first_open)

    # Realised volatility: std of close prices across the 50 bars
    feat_df['seen_volatility'] = grouped['close'].std()

    # How far the high/low extremes are from the final close
    feat_df['max_high_ratio'] = grouped['high'].max() / last_close
    feat_df['min_low_ratio'] = grouped['low'].min() / last_close

    # Total price range normalised by close
    feat_df['high_low_spread'] = (grouped['high'].max() - grouped['low'].min()) / last_close

    # --- Per-session advanced features (requires looping) ---
    adv_features = []
    for session, data in grouped:
        close_prices = data['close'].values
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        returns = np.diff(close_prices) / close_prices[:-1]  # bar-to-bar returns

        feat = {'session': session}
        if len(close_prices) < 20:
            adv_features.append(feat)
            continue

        # --- Trailing momentum at different horizons ---
        feat['ret_last_5'] = (close_prices[-1] / close_prices[-6]) - 1.0
        feat['ret_last_10'] = (close_prices[-1] / close_prices[-11]) - 1.0
        feat['ret_last_20'] = (close_prices[-1] / close_prices[-21]) - 1.0

        # --- Exponential Moving Average crossover ---
        # EMA-10 crossing above EMA-30 is a classic bullish signal
        ema_10 = pd.Series(close_prices).ewm(span=10).mean().iloc[-1]
        ema_30 = pd.Series(close_prices).ewm(span=30).mean().iloc[-1]
        feat['ema_10_30_cross'] = (ema_10 / ema_30) - 1.0  # positive = bullish
        feat['price_vs_ema10'] = (close_prices[-1] / ema_10) - 1.0

        # --- RSI (Relative Strength Index) ---
        feat['rsi_14'] = calculate_rsi(close_prices, 14)

        # --- Higher statistical moments of returns ---
        if len(returns) > 5:
            feat['ret_skew'] = skew(returns)      # asymmetry of return distribution
            feat['ret_kurt'] = kurtosis(returns)   # tail heaviness
        else:
            feat['ret_skew'] = 0
            feat['ret_kurt'] = 0

        # --- Serial correlation (mean reversion vs. momentum) ---
        if len(returns) > 10:
            feat['autocorr_1'] = pd.Series(returns).autocorr(lag=1)  # immediate persistence
            feat['autocorr_5'] = pd.Series(returns).autocorr(lag=5)  # medium-term persistence
        else:
            feat['autocorr_1'] = 0
            feat['autocorr_5'] = 0

        # --- Fourier transform: dominant cyclical frequencies ---
        # Removes the mean (DC component) and finds the two strongest oscillations
        fft_vals = np.abs(rfft(close_prices - np.mean(close_prices)))
        if len(fft_vals) >= 3:
            fft_vals[0] = 0  # zero out DC component
            top_indices = np.argsort(fft_vals)[-2:]
            feat['fourier_max_freq_amp'] = fft_vals[top_indices[1]]
            feat['fourier_2nd_freq_amp'] = fft_vals[top_indices[0]]
        else:
            feat['fourier_max_freq_amp'] = 0
            feat['fourier_2nd_freq_amp'] = 0

        # --- Bar-level microstructure ---
        green_bars = np.sum(close_prices > open_prices)  # bars where close > open
        feat['green_bar_ratio'] = green_bars / len(close_prices)
        feat['avg_bar_spread'] = np.mean((high_prices - low_prices) / open_prices)

        adv_features.append(feat)

    # Merge vectorised + per-session features into one DataFrame
    adv_df = pd.DataFrame(adv_features).set_index('session')
    feat_df = feat_df.join(adv_df).fillna(0)
    return feat_df