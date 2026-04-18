import numpy as np
import pandas as pd


class WeightedRidgeRegressionModel:
    def __init__(
        self,
        *,
        alpha: float = 1.0,
        weight_power: float = 1.0,
        min_weight: float = 0.25,
    ) -> None:
        self.alpha = float(alpha)
        self.weight_power = float(weight_power)
        self.min_weight = float(min_weight)
        self.feature_names: list[str] = []
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.residual_std_: float = 1.0

    def _prepare_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            raise ValueError("Model has not been fitted yet.")
        frame = features.reindex(columns=self.feature_names).fillna(0.0)
        return frame.sort_index()

    def _sample_weights(self, target: pd.Series) -> np.ndarray:
        weights = np.abs(target.to_numpy(dtype=float))
        if self.weight_power != 1.0:
            weights = np.power(weights, self.weight_power)
        weights = np.clip(weights, self.min_weight, None)
        mean_weight = float(weights.mean()) if len(weights) else 1.0
        if mean_weight <= 0.0 or not np.isfinite(mean_weight):
            mean_weight = 1.0
        return weights / mean_weight

    def fit(self, features: pd.DataFrame, target_return: pd.Series) -> "WeightedRidgeRegressionModel":
        frame = features.fillna(0.0).sort_index()
        target = pd.Series(target_return, index=frame.index).astype(float)
        weights = self._sample_weights(target)

        self.feature_names = list(frame.columns)
        design_matrix = np.column_stack([np.ones(len(frame)), frame.to_numpy(dtype=float)])
        sqrt_weights = np.sqrt(weights)
        weighted_design = design_matrix * sqrt_weights[:, None]
        weighted_target = target.to_numpy(dtype=float) * sqrt_weights

        penalty = np.eye(design_matrix.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = weighted_design.T @ weighted_design + self.alpha * penalty
        rhs = weighted_design.T @ weighted_target
        coefficients = np.linalg.solve(lhs, rhs)

        self.intercept_ = float(coefficients[0])
        self.coef_ = coefficients[1:]

        residuals = target.to_numpy(dtype=float) - self.predict_expected_return(frame).to_numpy(dtype=float)
        weighted_mean = float(np.average(residuals, weights=weights)) if len(residuals) else 0.0
        weighted_var = float(np.average((residuals - weighted_mean) ** 2, weights=weights)) if len(residuals) else 0.0
        self.residual_std_ = float(max(np.sqrt(weighted_var), 1e-6))
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
