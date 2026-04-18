import numpy as np
import pandas as pd


class LinearBaselineModel:
    def __init__(self, ridge_alpha: float = 2000.0) -> None:
        self.ridge_alpha = float(ridge_alpha)
        self.feature_names: list[str] = []
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.residual_std_: float = 1.0
        self.uncertainty_intercept_: float = 0.0
        self.uncertainty_coef_: np.ndarray | None = None
        self.uncertainty_floor_: float = 1e-6
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None

    def _prepare_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            raise ValueError("Model has not been fitted yet.")
        frame = features.reindex(columns=self.feature_names).fillna(0.0)
        return frame.sort_index()

    def _scale_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError("Model has not been fitted yet.")
        values = frame.to_numpy(dtype=float)
        return (values - self.feature_mean_) / self.feature_scale_

    @staticmethod
    def _fit_ridge(design_matrix: np.ndarray, target: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
        target_mean = float(target.mean()) if len(target) else 0.0
        centered_target = target - target_mean
        if design_matrix.shape[1] == 0:
            return target_mean, np.zeros(0, dtype=float)

        gram = design_matrix.T @ design_matrix
        ridge_penalty = alpha * np.eye(gram.shape[0], dtype=float)
        coef = np.linalg.solve(gram + ridge_penalty, design_matrix.T @ centered_target)
        return target_mean, coef

    def fit(self, features: pd.DataFrame, target_return: pd.Series) -> "LinearBaselineModel":
        frame = features.fillna(0.0).sort_index()
        target = pd.Series(target_return, index=frame.index).astype(float)

        self.feature_names = list(frame.columns)
        feature_values = frame.to_numpy(dtype=float)
        self.feature_mean_ = feature_values.mean(axis=0)
        self.feature_scale_ = feature_values.std(axis=0, ddof=0)
        self.feature_scale_ = np.where(self.feature_scale_ > 1e-6, self.feature_scale_, 1.0)
        scaled_values = self._scale_frame(frame)

        self.intercept_, self.coef_ = self._fit_ridge(
            scaled_values,
            target.to_numpy(dtype=float),
            alpha=self.ridge_alpha,
        )

        fitted = self.intercept_ + scaled_values @ self.coef_
        residuals = target.to_numpy(dtype=float) - fitted
        abs_residuals = np.abs(residuals)
        residual_std = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
        self.residual_std_ = float(max(residual_std, 1e-6))
        self.uncertainty_floor_ = float(max(np.quantile(abs_residuals, 0.1), 1e-6))

        uncertainty_target = np.log1p(abs_residuals)
        self.uncertainty_intercept_, self.uncertainty_coef_ = self._fit_ridge(
            scaled_values,
            uncertainty_target,
            alpha=self.ridge_alpha,
        )
        return self

    def predict_expected_return(self, features: pd.DataFrame) -> pd.Series:
        frame = self._prepare_frame(features)
        scaled_values = self._scale_frame(frame)
        predictions = self.intercept_ + scaled_values @ self.coef_
        return pd.Series(predictions, index=frame.index, name="predicted_return")

    def predict_uncertainty(self, features: pd.DataFrame) -> pd.Series:
        frame = self._prepare_frame(features)
        scaled_values = self._scale_frame(frame)
        if self.uncertainty_coef_ is None:
            values = np.full(len(frame), self.residual_std_, dtype=float)
        else:
            raw_uncertainty = self.uncertainty_intercept_ + scaled_values @ self.uncertainty_coef_
            values = np.expm1(raw_uncertainty)
            values = np.where(np.isfinite(values), values, self.residual_std_)
            values = np.clip(values, self.uncertainty_floor_, None)
        return pd.Series(values, index=frame.index, name="predicted_uncertainty")
