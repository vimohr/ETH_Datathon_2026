from src.experiments.catalog import MODEL_CATALOG
from src.experiments.config import ModelSpec
from src.models.baseline import LinearBaselineModel
from src.models.ridge import RidgeRegressionModel
from src.models.weighted import WeightedRidgeRegressionModel


def build_model(spec: ModelSpec):
    if spec.name == "linear":
        return LinearBaselineModel()

    if spec.name == "ridge":
        return RidgeRegressionModel(alpha=float(spec.params.get("alpha", 1.0)))

    if spec.name == "weighted_linear":
        return WeightedRidgeRegressionModel(
            alpha=0.0,
            weight_power=float(spec.params.get("weight_power", 1.0)),
            min_weight=float(spec.params.get("min_weight", 0.25)),
        )

    if spec.name == "weighted_ridge":
        return WeightedRidgeRegressionModel(
            alpha=float(spec.params.get("alpha", 1.0)),
            weight_power=float(spec.params.get("weight_power", 1.0)),
            min_weight=float(spec.params.get("min_weight", 0.25)),
        )

    available = ", ".join(sorted(MODEL_CATALOG))
    raise ValueError(f"Unsupported model={spec.name!r}. Available options: {available}.")
