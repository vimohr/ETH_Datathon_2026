import numpy as np
import pandas as pd


class RidgeRegressionModel:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.feature_names: list[str] = []
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.residual_std_: float = 1.0

    def _prepare_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            raise ValueError("Model has not been fitted yet.")
        frame = features.reindex(columns=self.feature_names).fillna(0.0)
        return frame.sort_index()

    def fit(self, features: pd.DataFrame, target_return: pd.Series) -> "RidgeRegressionModel":
        frame = features.fillna(0.0).sort_index()
        target = pd.Series(target_return, index=frame.index).astype(float)

        self.feature_names = list(frame.columns)
        design_matrix = np.column_stack([np.ones(len(frame)), frame.to_numpy(dtype=float)])
        penalty = np.eye(design_matrix.shape[1], dtype=float)
        penalty[0, 0] = 0.0

        lhs = design_matrix.T @ design_matrix + self.alpha * penalty
        rhs = design_matrix.T @ target.to_numpy(dtype=float)
        coefficients = np.linalg.solve(lhs, rhs)

        self.intercept_ = float(coefficients[0])
        self.coef_ = coefficients[1:]

        residuals = target.to_numpy(dtype=float) - self.predict_expected_return(frame).to_numpy(dtype=float)
        residual_std = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
        self.residual_std_ = float(max(residual_std, 1e-6))
        return self

    def predict_expected_return(self, features: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        frame = self._prepare_frame(features)
        predictions = self.intercept_ + frame.to_numpy(dtype=float) @ self.coef_
        return pd.Series(predictions, index=frame.index, name="predicted_return")

    def predict_uncertainty(self, features: pd.DataFrame) -> pd.Series:
        frame = self._prepare_frame(features)
        return pd.Series(
            np.full(len(frame), self.residual_std_, dtype=float),
            index=frame.index,
            name="predicted_uncertainty",
        )
