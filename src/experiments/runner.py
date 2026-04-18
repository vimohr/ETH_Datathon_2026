from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.load import load_bars, load_headlines, load_train_bars
from src.data.splits import make_repeated_session_folds
from src.data.targets import build_train_targets
from src.evaluation.metrics import pnl, sharpe_from_positions
from src.experiments.config import ExperimentConfig
from src.experiments.features import build_feature_block, build_feature_matrix
from src.experiments.models import build_model
from src.kaggle_utils import DEFAULT_COMPETITION, submit_competition_file
from src.models.uncertainty import size_positions
from src.paths import OOF_DIR, SUBMISSIONS_DIR, ensure_output_dirs
from src.submission import (
    build_submission,
    build_submission_metadata,
    combine_split_submissions,
    expected_competition_sessions,
    save_submission,
)


@dataclass
class TrainingArtifacts:
    fold_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    train_feature_count: int


@dataclass
class CompetitionArtifacts:
    public_submission_path: Path
    private_submission_path: Path
    competition_submission_path: Path
    kaggle_response: str | None = None


def _aggregate_repeated_oof_predictions(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    if oof_predictions.empty or not oof_predictions.index.has_duplicates:
        return oof_predictions.sort_index()

    grouped = oof_predictions.groupby(level=0, sort=True)
    aggregated = pd.DataFrame(
        {
            "predicted_return": grouped["predicted_return"].mean(),
            "predicted_uncertainty": grouped["predicted_uncertainty"].mean(),
            "target_position": grouped["target_position"].mean(),
            "realized_return": grouped["realized_return"].first(),
            "cv_prediction_count": grouped.size().astype(float),
        }
    )
    aggregated["pnl"] = aggregated["target_position"] * aggregated["realized_return"]
    return aggregated.sort_index()


def _filter_sessions(frame: pd.DataFrame, sessions) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_index = pd.Index(sessions).astype(int)
    return frame.loc[frame["session"].isin(session_index)].copy()


def _load_train_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    train_seen_bars, train_unseen_bars = load_train_bars()
    train_headlines = load_headlines("train", "seen")
    train_targets = build_train_targets(train_seen_bars, train_unseen_bars)["target_return"].sort_index()
    return train_seen_bars, train_headlines, train_unseen_bars, train_targets


def _include_headlines(config: ExperimentConfig) -> bool:
    return any(spec.name.startswith("headline") for spec in config.feature_blocks)


def _timestamp_string() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _config_notes(config: ExperimentConfig) -> str:
    return config.to_json()


def _size_positions_for_config(
    config: ExperimentConfig,
    predicted_return: pd.Series,
    predicted_uncertainty: pd.Series,
) -> pd.Series:
    position_kwargs = dict(config.position_sizing)
    return size_positions(predicted_return, predicted_uncertainty, **position_kwargs)


def _save_training_outputs(config: ExperimentConfig, training: TrainingArtifacts) -> Path:
    oof_path = OOF_DIR / f"{config.experiment_name}_oof.csv"
    training.oof_predictions.to_csv(oof_path, index_label="session")
    print(f"Saved OOF predictions to {oof_path}")
    return oof_path


def _run_training(config: ExperimentConfig) -> TrainingArtifacts:
    training = cross_validate_experiment(config)
    if config.cv_repeats > 1:
        repeat_summary = (
            training.fold_summary.groupby(["repeat", "seed"], sort=True)["sharpe"]
            .mean()
            .reset_index(name="mean_sharpe")
        )
        print(repeat_summary.to_string(index=False))
        print(
            "Repeated CV Sharpe: "
            f"{repeat_summary['mean_sharpe'].mean():.4f} +/- {repeat_summary['mean_sharpe'].std(ddof=0):.4f}"
        )
    else:
        print(training.fold_summary.to_string(index=False))
        print(f"Mean CV Sharpe: {training.fold_summary['sharpe'].mean():.4f}")
    _save_training_outputs(config, training)
    return training


def cross_validate_experiment(config: ExperimentConfig) -> TrainingArtifacts:
    train_seen_bars, train_headlines, _, train_targets = _load_train_inputs()
    all_sessions = train_targets.index

    fold_rows: list[dict[str, float]] = []
    oof_frames: list[pd.DataFrame] = []
    last_feature_count = 0

    for repeat_id, repeat_seed, fold_id, train_sessions, valid_sessions in make_repeated_session_folds(
        all_sessions,
        n_folds=config.cv_folds,
        seed=config.seed,
        n_repeats=config.cv_repeats,
    ):
        train_bars = _filter_sessions(train_seen_bars, train_sessions)
        valid_bars = _filter_sessions(train_seen_bars, valid_sessions)
        train_fold_headlines = _filter_sessions(train_headlines, train_sessions)
        valid_fold_headlines = _filter_sessions(train_headlines, valid_sessions)

        block_instances = [build_feature_block(spec) for spec in config.feature_blocks]
        for block in block_instances:
            block.fit(bars=train_bars, headlines=train_fold_headlines, sessions=train_sessions)

        train_features = build_feature_matrix(
            config.feature_blocks,
            block_instances,
            bars=train_bars,
            headlines=train_fold_headlines,
            sessions=train_sessions,
        )
        valid_features = build_feature_matrix(
            config.feature_blocks,
            block_instances,
            bars=valid_bars,
            headlines=valid_fold_headlines,
            sessions=valid_sessions,
        )

        train_target = train_targets.loc[train_features.index]
        valid_target = train_targets.loc[valid_features.index]
        model = build_model(config.model).fit(train_features, train_target)
        predicted_return = model.predict_expected_return(valid_features)
        predicted_uncertainty = model.predict_uncertainty(valid_features)
        target_position = _size_positions_for_config(config, predicted_return, predicted_uncertainty)
        fold_pnl = pnl(target_position, valid_target)
        fold_sharpe = sharpe_from_positions(target_position, valid_target)

        last_feature_count = int(train_features.shape[1])
        fold_rows.append(
            {
                "repeat": float(repeat_id),
                "seed": float(repeat_seed),
                "fold": float(fold_id),
                "n_train": float(len(train_sessions)),
                "n_valid": float(len(valid_sessions)),
                "feature_count": float(last_feature_count),
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
                    "repeat": repeat_id,
                    "seed": repeat_seed,
                    "fold": fold_id,
                }
            )
        )

    oof_predictions = _aggregate_repeated_oof_predictions(pd.concat(oof_frames).sort_index())
    return TrainingArtifacts(
        fold_summary=pd.DataFrame(fold_rows),
        oof_predictions=oof_predictions,
        train_feature_count=last_feature_count,
    )


def fit_full_model(config: ExperimentConfig):
    train_seen_bars, train_headlines, _, train_targets = _load_train_inputs()
    all_sessions = train_targets.index
    block_instances = [build_feature_block(spec) for spec in config.feature_blocks]
    for block in block_instances:
        block.fit(bars=train_seen_bars, headlines=train_headlines, sessions=all_sessions)

    train_features = build_feature_matrix(
        config.feature_blocks,
        block_instances,
        bars=train_seen_bars,
        headlines=train_headlines,
        sessions=all_sessions,
    )
    model = build_model(config.model).fit(train_features, train_targets.loc[train_features.index])
    return block_instances, model, int(train_features.shape[1])


def predict_split_with_model(
    config: ExperimentConfig,
    *,
    block_instances,
    model,
    feature_count: int,
    test_split: str,
):
    test_bars = load_bars(test_split, "seen")
    test_headlines = load_headlines(test_split, "seen")
    test_sessions = test_bars["session"].unique()
    test_features = build_feature_matrix(
        config.feature_blocks,
        block_instances,
        bars=test_bars,
        headlines=test_headlines,
        sessions=test_sessions,
    )
    predicted_return = model.predict_expected_return(test_features)
    predicted_uncertainty = model.predict_uncertainty(test_features)
    target_position = _size_positions_for_config(config, predicted_return, predicted_uncertainty)
    submission = build_submission(test_features.index, target_position)
    return submission, feature_count


def predict_split(config: ExperimentConfig, test_split: str):
    block_instances, model, feature_count = fit_full_model(config)
    return predict_split_with_model(
        config,
        block_instances=block_instances,
        model=model,
        feature_count=feature_count,
        test_split=test_split,
    )


def _save_split_submission(
    config: ExperimentConfig,
    *,
    training: TrainingArtifacts,
    test_split: str,
    submission: pd.DataFrame,
    feature_count: int,
    timestamp: str,
    output_path=None,
) -> Path:
    expected_sessions = load_bars(test_split, "seen")["session"].unique()
    resolved_output = output_path
    if resolved_output is None:
        resolved_output = SUBMISSIONS_DIR / f"{timestamp}_{config.experiment_name}_{test_split}.csv"
    resolved_output = Path(resolved_output)
    metadata = build_submission_metadata(
        model_name=config.experiment_name,
        test_split=test_split,
        feature_count=int(feature_count),
        mean_cv_sharpe=float(training.fold_summary["sharpe"].mean()),
        include_headlines=_include_headlines(config),
        notes=_config_notes(config),
    )
    saved_path = save_submission(
        submission,
        resolved_output,
        expected_sessions=expected_sessions,
        metadata=metadata,
        latest_alias=f"latest_{test_split}.csv",
    )
    print(f"Saved submission to {saved_path}")
    return saved_path


def _save_competition_submission(
    config: ExperimentConfig,
    *,
    training: TrainingArtifacts,
    public_submission: pd.DataFrame,
    private_submission: pd.DataFrame,
    public_submission_path: Path,
    private_submission_path: Path,
    timestamp: str,
    output_path=None,
    competition: str = DEFAULT_COMPETITION,
) -> Path:
    combined_submission = combine_split_submissions(public_submission, private_submission)
    resolved_output = output_path
    if resolved_output is None:
        resolved_output = SUBMISSIONS_DIR / f"{timestamp}_{config.experiment_name}_competition.csv"
    resolved_output = Path(resolved_output)
    metadata = build_submission_metadata(
        model_name=config.experiment_name,
        test_split="competition",
        feature_count=int(training.train_feature_count),
        mean_cv_sharpe=float(training.fold_summary["sharpe"].mean()),
        include_headlines=_include_headlines(config),
        notes=_config_notes(config),
    )
    metadata.update(
        {
            "competition_name": competition,
            "source_public_submission": str(public_submission_path),
            "source_private_submission": str(private_submission_path),
            "row_count": int(len(combined_submission)),
        }
    )
    saved_path = save_submission(
        combined_submission,
        resolved_output,
        expected_sessions=expected_competition_sessions(),
        metadata=metadata,
        latest_alias="latest_competition.csv",
    )
    print(f"Saved combined competition submission to {saved_path}")
    return saved_path


def run_experiment_pipeline(
    config: ExperimentConfig,
    *,
    test_split: str,
    output_path=None,
    cv_only: bool = False,
):
    ensure_output_dirs()

    training = _run_training(config)

    if cv_only:
        return None

    submission, feature_count = predict_split(config, test_split)
    saved_path = _save_split_submission(
        config,
        training=training,
        test_split=test_split,
        submission=submission,
        feature_count=feature_count,
        timestamp=_timestamp_string(),
        output_path=output_path,
    )
    return saved_path


def run_experiment_competition_pipeline(
    config: ExperimentConfig,
    *,
    public_output_path=None,
    private_output_path=None,
    competition_output_path=None,
    competition: str = DEFAULT_COMPETITION,
    submit_kaggle: bool = False,
    submission_message: str | None = None,
    env_file=None,
):
    ensure_output_dirs()

    training = _run_training(config)
    block_instances, model, feature_count = fit_full_model(config)
    timestamp = _timestamp_string()

    public_submission, _ = predict_split_with_model(
        config,
        block_instances=block_instances,
        model=model,
        feature_count=feature_count,
        test_split="public_test",
    )
    private_submission, _ = predict_split_with_model(
        config,
        block_instances=block_instances,
        model=model,
        feature_count=feature_count,
        test_split="private_test",
    )

    public_saved_path = _save_split_submission(
        config,
        training=training,
        test_split="public_test",
        submission=public_submission,
        feature_count=feature_count,
        timestamp=timestamp,
        output_path=public_output_path,
    )
    private_saved_path = _save_split_submission(
        config,
        training=training,
        test_split="private_test",
        submission=private_submission,
        feature_count=feature_count,
        timestamp=timestamp,
        output_path=private_output_path,
    )
    competition_saved_path = _save_competition_submission(
        config,
        training=training,
        public_submission=public_submission,
        private_submission=private_submission,
        public_submission_path=public_saved_path,
        private_submission_path=private_saved_path,
        timestamp=timestamp,
        output_path=competition_output_path,
        competition=competition,
    )

    kaggle_response = None
    if submit_kaggle:
        if not submission_message:
            raise ValueError("submission_message is required when submit_kaggle=True.")
        kaggle_response, loaded_files = submit_competition_file(
            competition_saved_path,
            message=submission_message,
            competition=competition,
            env_file=env_file,
        )
        if loaded_files:
            print("Loaded Kaggle credentials from:")
            for path in loaded_files:
                print(f"- {path}")
        print(kaggle_response)

    return CompetitionArtifacts(
        public_submission_path=public_saved_path,
        private_submission_path=private_saved_path,
        competition_submission_path=competition_saved_path,
        kaggle_response=kaggle_response,
    )
