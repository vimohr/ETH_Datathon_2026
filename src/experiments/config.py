import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.settings import CV_FOLDS, RANDOM_SEED


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    alias: str | None = None

    @property
    def resolved_alias(self) -> str:
        return self.alias or self.name


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    feature_blocks: tuple[FeatureSpec, ...]
    model: ModelSpec
    cv_folds: int = CV_FOLDS
    seed: int = RANDOM_SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "feature_blocks": [
                {
                    "name": spec.name,
                    "alias": spec.alias,
                    "params": spec.params,
                }
                for spec in self.feature_blocks
            ],
            "model": {
                "name": self.model.name,
                "params": self.model.params,
            },
            "cv_folds": self.cv_folds,
            "seed": self.seed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _as_feature_spec(raw_spec: dict[str, Any]) -> FeatureSpec:
    if "name" not in raw_spec:
        raise ValueError("Each feature block must define a 'name'.")
    return FeatureSpec(
        name=str(raw_spec["name"]),
        alias=raw_spec.get("alias"),
        params=dict(raw_spec.get("params") or {}),
    )


def _as_model_spec(raw_spec: dict[str, Any]) -> ModelSpec:
    if "name" not in raw_spec:
        raise ValueError("The model config must define a 'name'.")
    return ModelSpec(
        name=str(raw_spec["name"]),
        params=dict(raw_spec.get("params") or {}),
    )


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    if "experiment_name" not in payload:
        raise ValueError("Experiment config must define 'experiment_name'.")
    raw_features = payload.get("feature_blocks") or []
    if not raw_features:
        raise ValueError("Experiment config must define at least one feature block.")
    if "model" not in payload:
        raise ValueError("Experiment config must define a model.")

    feature_specs = tuple(_as_feature_spec(raw_spec) for raw_spec in raw_features)
    aliases = [spec.resolved_alias for spec in feature_specs]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Feature block aliases must be unique within one experiment config.")

    return ExperimentConfig(
        experiment_name=str(payload["experiment_name"]),
        feature_blocks=feature_specs,
        model=_as_model_spec(dict(payload["model"])),
        cv_folds=int(payload.get("cv_folds", CV_FOLDS)),
        seed=int(payload.get("seed", RANDOM_SEED)),
    )


def load_experiment_config(path_like) -> ExperimentConfig:
    path = Path(path_like)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return experiment_config_from_dict(payload)
