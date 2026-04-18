import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.config import experiment_config_from_dict
from src.paths import ROOT


@dataclass(frozen=True)
class SweepChoice:
    axis_name: str
    slug: str
    config: dict[str, Any]


@dataclass(frozen=True)
class SweepAxis:
    name: str
    choices: tuple[SweepChoice, ...]


@dataclass(frozen=True)
class SweepSpec:
    name_prefix: str
    output_dir: Path
    axes: tuple[SweepAxis, ...]
    base_config: dict[str, Any]
    name_separator: str = "_"


def _resolve_path(path_like) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_choice(axis_name: str, raw_choice: dict[str, Any]) -> SweepChoice:
    if "slug" not in raw_choice:
        raise ValueError(f"Each choice in axis {axis_name!r} must define a 'slug'.")
    return SweepChoice(
        axis_name=axis_name,
        slug=str(raw_choice["slug"]),
        config=dict(raw_choice.get("config") or {}),
    )


def _as_axis(raw_axis: dict[str, Any]) -> SweepAxis:
    if "name" not in raw_axis:
        raise ValueError("Each sweep axis must define a 'name'.")
    axis_name = str(raw_axis["name"])
    raw_choices = raw_axis.get("choices") or []
    if not raw_choices:
        raise ValueError(f"Sweep axis {axis_name!r} must define at least one choice.")

    choices = tuple(_as_choice(axis_name, raw_choice) for raw_choice in raw_choices)
    slugs = [choice.slug for choice in choices]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"Sweep axis {axis_name!r} has duplicate choice slugs.")
    return SweepAxis(name=axis_name, choices=choices)


def sweep_spec_from_dict(payload: dict[str, Any], *, source_path: Path | None = None) -> SweepSpec:
    raw_axes = payload.get("axes") or []
    if not raw_axes:
        raise ValueError("Sweep spec must define at least one axis.")

    axes = tuple(_as_axis(raw_axis) for raw_axis in raw_axes)
    axis_names = [axis.name for axis in axes]
    if len(set(axis_names)) != len(axis_names):
        raise ValueError("Sweep spec axis names must be unique.")

    default_name_prefix = source_path.stem if source_path is not None else "experiment"
    default_output_dir = Path("configs/generated") / default_name_prefix
    output_dir = _resolve_path(payload.get("output_dir", default_output_dir))

    return SweepSpec(
        name_prefix=str(payload.get("name_prefix", default_name_prefix)),
        output_dir=output_dir,
        axes=axes,
        base_config=dict(payload.get("base_config") or {}),
        name_separator=str(payload.get("name_separator", "_")),
    )


def load_sweep_spec(path_like) -> SweepSpec:
    path = _resolve_path(path_like)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sweep_spec_from_dict(payload, source_path=path)


def expand_sweep_spec(spec: SweepSpec) -> list[dict[str, Any]]:
    expanded_configs: list[dict[str, Any]] = []

    for choice_combo in itertools.product(*(axis.choices for axis in spec.axes)):
        config_payload = copy.deepcopy(spec.base_config)
        slugs = [choice.slug for choice in choice_combo]
        experiment_name = spec.name_separator.join(
            [part for part in [spec.name_prefix, *slugs] if str(part).strip()]
        )
        config_payload["experiment_name"] = experiment_name

        for choice in choice_combo:
            config_payload = _deep_merge(config_payload, choice.config)

        experiment_config_from_dict(config_payload)
        expanded_configs.append(config_payload)

    return expanded_configs


def write_expanded_configs(
    configs: list[dict[str, Any]],
    *,
    output_dir,
) -> list[Path]:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    for existing_path in resolved_output_dir.glob("*.json"):
        existing_path.unlink()

    written_paths: list[Path] = []
    for config_payload in configs:
        experiment_name = str(config_payload["experiment_name"])
        output_path = resolved_output_dir / f"{experiment_name}.json"
        output_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written_paths.append(output_path)
    return written_paths
