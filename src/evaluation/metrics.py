import numpy as np
import pandas as pd

from src.settings import ANNUALIZATION_FACTOR


def pnl(target_position, realized_return) -> pd.Series:
    positions = pd.Series(target_position, copy=False).astype(float)
    returns = pd.Series(realized_return, index=positions.index, copy=False).astype(float)
    return (positions * returns).rename("pnl")


def sharpe_from_pnl(pnl_values, annualization: float = ANNUALIZATION_FACTOR) -> float:
    pnl_series = pd.Series(pnl_values, copy=False).astype(float)
    pnl_std = float(pnl_series.std(ddof=0))
    if not np.isfinite(pnl_std) or pnl_std == 0.0:
        return 0.0
    return float(pnl_series.mean() / pnl_std * annualization)


def sharpe_from_positions(target_position, realized_return, annualization: float = ANNUALIZATION_FACTOR) -> float:
    return sharpe_from_pnl(pnl(target_position, realized_return), annualization=annualization)
