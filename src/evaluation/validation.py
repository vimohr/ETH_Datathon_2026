import pandas as pd

from src.data.splits import make_session_folds
from src.evaluation.metrics import pnl, sharpe_from_positions
from src.models.baseline import LinearBaselineModel
from src.models.uncertainty import size_positions
from src.settings import CV_FOLDS, RANDOM_SEED


def run_cross_validation(
    features: pd.DataFrame,
    target_return: pd.Series,
    n_folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
):
    feature_frame = features.fillna(0.0).sort_index()
    target_series = pd.Series(target_return, index=feature_frame.index).astype(float).sort_index()

    fold_rows: list[dict[str, float]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id, train_sessions, valid_sessions in make_session_folds(
        feature_frame.index, n_folds=n_folds, seed=seed
    ):
        train_features = feature_frame.loc[train_sessions]
        valid_features = feature_frame.loc[valid_sessions]
        train_target = target_series.loc[train_sessions]
        valid_target = target_series.loc[valid_sessions]

        model = LinearBaselineModel().fit(train_features, train_target)
        predicted_return = model.predict_expected_return(valid_features)
        predicted_uncertainty = model.predict_uncertainty(valid_features)
        target_position = size_positions(predicted_return, predicted_uncertainty)
        fold_pnl = pnl(target_position, valid_target)
        fold_sharpe = sharpe_from_positions(target_position, valid_target)

        fold_rows.append(
            {
                "fold": float(fold_id),
                "n_train": float(len(train_sessions)),
                "n_valid": float(len(valid_sessions)),
                "sharpe": fold_sharpe,
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "predicted_return": predicted_return,
                    "predicted_uncertainty": predicted_uncertainty,
                    "target_position": target_position,
                    "realized_return": valid_target,
                    "pnl": fold_pnl,
                    "fold": fold_id,
                }
            )
        )

    fold_summary = pd.DataFrame(fold_rows)
    oof_predictions = pd.concat(oof_frames).sort_index()
    return fold_summary, oof_predictions
