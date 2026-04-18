import numpy as np
import pandas as pd


def make_session_folds(sessions, n_folds: int = 5, seed: int = 42):
    unique_sessions = np.array(sorted(pd.Index(sessions).unique()))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if len(unique_sessions) < n_folds:
        raise ValueError("n_folds cannot exceed the number of sessions.")

    shuffled_sessions = unique_sessions.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled_sessions)
    valid_fold_sessions = np.array_split(shuffled_sessions, n_folds)

    for fold_id, valid_sessions in enumerate(valid_fold_sessions):
        train_sessions = np.concatenate(
            [fold for index, fold in enumerate(valid_fold_sessions) if index != fold_id]
        )
        yield fold_id, pd.Index(train_sessions), pd.Index(valid_sessions)


def make_repeated_session_folds(sessions, n_folds: int = 5, seed: int = 42, n_repeats: int = 1):
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    for repeat_id in range(int(n_repeats)):
        repeat_seed = int(seed) + repeat_id
        for fold_id, train_sessions, valid_sessions in make_session_folds(
            sessions,
            n_folds=n_folds,
            seed=repeat_seed,
        ):
            yield repeat_id, repeat_seed, fold_id, train_sessions, valid_sessions
