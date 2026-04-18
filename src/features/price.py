import numpy as np
import pandas as pd


def _tail_return(close: np.ndarray, lookback: int) -> float:
    if len(close) <= lookback:
        return 0.0
    return float(close[-1] / close[-1 - lookback] - 1.0)


def _session_price_features(session_bars: pd.DataFrame) -> dict[str, float]:
    open_values = session_bars["open"].to_numpy(dtype=float)
    high_values = session_bars["high"].to_numpy(dtype=float)
    low_values = session_bars["low"].to_numpy(dtype=float)
    close_values = session_bars["close"].to_numpy(dtype=float)
    bar_index = session_bars["bar_ix"].to_numpy(dtype=float)

    close_changes = np.diff(close_values)
    close_returns = close_changes / np.clip(close_values[:-1], 1e-6, None)
    running_max = np.maximum.accumulate(close_values)
    drawdowns = close_values / np.clip(running_max, 1e-6, None) - 1.0
    slope = float(np.polyfit(bar_index, close_values, 1)[0]) if len(close_values) > 1 else 0.0

    return {
        "seen_return_open_to_close": float(close_values[-1] / open_values[0] - 1.0),
        "seen_return_close_to_close": float(close_values[-1] / close_values[0] - 1.0),
        "seen_high_return": float(high_values.max() / open_values[0] - 1.0),
        "seen_low_return": float(low_values.min() / open_values[0] - 1.0),
        "tail_return_5": _tail_return(close_values, lookback=5),
        "tail_return_10": _tail_return(close_values, lookback=10),
        "close_mean": float(close_values.mean()),
        "close_std": float(close_values.std(ddof=0)),
        "close_volatility": float(close_returns.std(ddof=0)) if len(close_returns) else 0.0,
        "realized_volatility": float(np.sqrt(np.mean(close_returns**2))) if len(close_returns) else 0.0,
        "intrabar_range_mean": float(np.mean((high_values - low_values) / np.clip(open_values, 1e-6, None))),
        "up_bar_ratio": float(np.mean(close_values >= open_values)),
        "positive_close_change_ratio": float(np.mean(close_changes >= 0.0)) if len(close_changes) else 0.0,
        "max_drawdown": float(drawdowns.min()),
        "trend_slope": slope,
        "trend_slope_normalized": float(slope / np.clip(close_values.mean(), 1e-6, None)),
    }


def build_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    feature_rows: list[dict[str, float]] = []
    sorted_bars = bars.sort_values(["session", "bar_ix"])
    for session, session_bars in sorted_bars.groupby("session", sort=True):
        feature_rows.append({"session": int(session), **_session_price_features(session_bars)})

    return pd.DataFrame(feature_rows).set_index("session").sort_index()
