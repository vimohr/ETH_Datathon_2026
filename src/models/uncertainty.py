import numpy as np
import pandas as pd

from src.settings import (
    MIN_SCORE_DENOMINATOR,
    NEUTRAL_BAND,
    POSITION_CLIP,
    POSITION_SCORE_PERCENTILE,
)


def score_predictions(predicted_return, uncertainty=None) -> pd.Series:
    scores = pd.Series(predicted_return, copy=False).astype(float)
    if uncertainty is None:
        return scores

    uncertainty_series = (
        pd.Series(uncertainty, index=scores.index, copy=False)
        .astype(float)
        .abs()
        .clip(lower=MIN_SCORE_DENOMINATOR)
    )
    return scores / uncertainty_series


def size_positions(
    predicted_return,
    uncertainty=None,
    max_abs_position: float = POSITION_CLIP,
    neutral_band: float = NEUTRAL_BAND,
    percentile: float = POSITION_SCORE_PERCENTILE,
) -> pd.Series:
    scores = score_predictions(predicted_return, uncertainty=uncertainty)
    if neutral_band > 0.0:
        scores = scores.where(scores.abs() >= neutral_band, 0.0)

    denominator = float(np.nanpercentile(np.abs(scores), percentile))
    if not np.isfinite(denominator) or denominator < MIN_SCORE_DENOMINATOR:
        denominator = 1.0

    positions = (scores / denominator).clip(lower=-1.0, upper=1.0) * max_abs_position
    return positions.rename("target_position")
