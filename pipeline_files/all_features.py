import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
import warnings

warnings.filterwarnings('ignore')

### Here extract features is defined

def calculate_rsi(prices, window=14):
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
        return 50.0
    return res[-1]

def extract_features(df):
    features = []
    grouped = df.groupby('session')
    
    first_open = grouped['open'].first()
    last_close = grouped['close'].last()
    
    feat_df = pd.DataFrame(index=first_open.index)
    feat_df['seen_return'] = (last_close / first_open) - 1.0
    feat_df['log_return'] = np.log(last_close) - np.log(first_open)
    feat_df['seen_volatility'] = grouped['close'].std()
    feat_df['max_high_ratio'] = grouped['high'].max() / last_close
    feat_df['min_low_ratio'] = grouped['low'].min() / last_close
    feat_df['high_low_spread'] = (grouped['high'].max() - grouped['low'].min()) / last_close
    
    adv_features = []
    for session, data in grouped:
        close_prices = data['close'].values
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        feat = {'session': session}
        if len(close_prices) < 20: 
            adv_features.append(feat)
            continue
            
        feat['ret_last_5'] = (close_prices[-1] / close_prices[-6]) - 1.0
        feat['ret_last_10'] = (close_prices[-1] / close_prices[-11]) - 1.0
        feat['ret_last_20'] = (close_prices[-1] / close_prices[-21]) - 1.0
        
        ema_10 = pd.Series(close_prices).ewm(span=10).mean().iloc[-1]
        ema_30 = pd.Series(close_prices).ewm(span=30).mean().iloc[-1]
        feat['ema_10_30_cross'] = (ema_10 / ema_30) - 1.0
        feat['price_vs_ema10'] = (close_prices[-1] / ema_10) - 1.0
        feat['rsi_14'] = calculate_rsi(close_prices, 14)
        
        if len(returns) > 5:
            feat['ret_skew'] = skew(returns)
            feat['ret_kurt'] = kurtosis(returns)
        else:
            feat['ret_skew'] = 0
            feat['ret_kurt'] = 0
            
        if len(returns) > 10:
            feat['autocorr_1'] = pd.Series(returns).autocorr(lag=1)
            feat['autocorr_5'] = pd.Series(returns).autocorr(lag=5)
        else:
            feat['autocorr_1'] = 0
            feat['autocorr_5'] = 0
            
        fft_vals = np.abs(rfft(close_prices - np.mean(close_prices)))
        if len(fft_vals) >= 3:
            fft_vals[0] = 0
            top_indices = np.argsort(fft_vals)[-2:]
            feat['fourier_max_freq_amp'] = fft_vals[top_indices[1]]
            feat['fourier_2nd_freq_amp'] = fft_vals[top_indices[0]]
        else:
            feat['fourier_max_freq_amp'] = 0
            feat['fourier_2nd_freq_amp'] = 0
            
        green_bars = np.sum(close_prices > open_prices)
        feat['green_bar_ratio'] = green_bars / len(close_prices)
        feat['avg_bar_spread'] = np.mean((high_prices - low_prices) / open_prices)
        adv_features.append(feat)
        
    adv_df = pd.DataFrame(adv_features).set_index('session')
    feat_df = feat_df.join(adv_df).fillna(0)
    return feat_df