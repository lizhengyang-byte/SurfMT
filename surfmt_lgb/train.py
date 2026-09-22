"""LightGBM multi-task training with cross-validated early stopping.

LightGBM is trained as ONE regressor per task. Key problems with the
single-holdout approach: sparse tasks have too few validation samples
in one fold, causing overly aggressive early stopping.

Solution: use the predefined 10-fold column to run cross-validation per
task to determine the best number of boosting rounds (best_iteration),
then refit on ALL labeled samples for that task with that round count.
This uses all data and gives a robust early-stop point.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd


# LightGBM hyperparameters tuned for small, tabular molecular data
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 10,
    "lambda_l1": 0.1,
    "lambda_l2": 0.3,
    "verbose": -1,
    "seed": 42,
    "num_threads": 4,
}

# Heterogeneous ensemble: diverse hyperparameter sets decorrelate predictions,
# giving a larger averaging gain than same-param multi-seed ensembles.
DIVERSE_CONFIGS = [
    {  # A: default
        "num_leaves": 31, "learning_rate": 0.03,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "min_data_in_leaf": 10, "lambda_l1": 0.1, "lambda_l2": 0.3,
    },
    {  # B: deeper, slower, less feature sampling
        "num_leaves": 63, "learning_rate": 0.02,
        "feature_fraction": 0.7, "bagging_fraction": 0.9,
        "min_data_in_leaf": 20, "lambda_l1": 0.1, "lambda_l2": 0.5,
    },
    {  # C: shallow, faster, more aggressive
        "num_leaves": 15, "learning_rate": 0.05,
        "feature_fraction": 0.9, "bagging_fraction": 0.7,
        "min_data_in_leaf": 5, "lambda_l1": 0.0, "lambda_l2": 0.1,
    },
    {  # D: very deep, very slow, strong sampling
        "num_leaves": 127, "learning_rate": 0.01,
        "feature_fraction": 0.6, "bagging_fraction": 0.9,
        "min_data_in_leaf": 40, "lambda_l1": 0.5, "lambda_l2": 1.0,
    },
    {  # E: default but stronger regularization (additive)
        "num_leaves": 31, "learning_rate": 0.025,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "min_data_in_leaf": 30, "lambda_l1": 0.3, "lambda_l2": 0.3,
    },
]

DEFAULT_NUM_ROUNDS = 5000
DEFAULT_EARLY_STOPPING = 100


class PredefinedFolds:
    """Iterable of (train_idx, valid_idx) built from the fold column."""

    def __init__(self, fold_array, num_folds=10):
        self.fold_array = fold_array
        self.num_folds = num_folds
        self._folds = []
        for f in range(num_folds):
            val_idx = np.where(fold_array == f)[0]
            tr_idx = np.where(fold_array != f)[0]
            self._folds.append((tr_idx, val_idx))
        # Drop folds that have no validation samples for this task
        self._folds = [(tr, va) for tr, va in self._folds if len(va) >= 2]

    def __iter__(self):
        return iter(self._folds)

    def __len__(self):
        return len(self._folds)


def _best_iter_from_cv(X, y, fold_array, params, num_rounds, early_stopping, seed):
    """Run CV to find the best number of boosting rounds."""
    params = dict(params)
    params["seed"] = seed
    dset = lgb.Dataset(X, label=y)
    folds = PredefinedFolds(fold_array)

    # Too few useable folds -> fall back to manual (no early stop)
    if len(folds) < 2:
        return None

    cv_results = lgb.cv(
        params,
        dset,
        num_boost_round=num_rounds,
        folds=folds,
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    # cv_results: dict of metric histories. Find the rmse-mean history.
    mean_key = None
    for k in cv_results.keys():
        if k.endswith("rmse-mean") or (k.endswith("-mean") and "rmse" in k):
            mean_key = k
            break
    if mean_key is None:
        return None
    rmse_hist = cv_results[mean_key]
    best_iter = int(np.argmin(rmse_hist)) + 1
    # Early stopping from the callback may have truncated; use min directly.
    return best_iter


def _standardize(y, epsilon=1e-12):
    """Return (y_std, mean, std). Standardize a target to unit variance.

    Targets with wildly different magnitudes (Gamma_max ~1e-6 vs pCMC ~3)
    make the absolute RMSE metric meaningless for early stopping. Standardization
    puts all tasks on a comparable scale so the CV metric tracks real improvement.
    """
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std < epsilon:
        std = 1.0
    y_std = (y - mean) / std
    return y_std, mean, std


def decide_transform(y):
    """Choose 'log' or 'none' target transform based on skewness.

    Highly right-skewed, strictly positive targets (Gamma_max, A_min span
    orders of magnitude) are better modeled in log space. Rule: if the target
    is strictly positive and log-transforming makes the skewness magnitude
    smaller (more symmetric), use log.
    """
    if not (y > 0).all():
        return "none"
    if len(y) < 10:
        return "none"
    try:
        from scipy import stats
        skew_raw = float(stats.skew(y))
        skew_log = float(stats.skew(np.log(y)))
    except Exception:
        return "none"
    if abs(skew_log) < abs(skew_raw):
        return "log"
    return "none"


def _apply_transform(y, transform):
    """Return target in the modeling space (log-transformed if requested)."""
    if transform == "log":
        return np.log(y)
    return y


def train_single_task(
    X,
    y,
    fold_array,
    transform="none",
    params_override=None,
    num_rounds=DEFAULT_NUM_ROUNDS,
    early_stopping=DEFAULT_EARLY_STOPPING,
    seed=42,
):
    """Train one LightGBM regressor using CV-based early stopping, refit full data.

    Targets are log-transformed (optional) then standardized internally;
    predictions are returned in the original scale via (model, best_iter, mean, std).

    Args:
        X: [N, F] labeled feature matrix for this task.
        y: [N] raw labels for this task.
        fold_array: [N] fold index per labeled sample (0-9).
        transform: 'log' or 'none'. If 'log', the model predicts log(y).
        params_override: optional partial override dict, merged atop LGB_PARAMS.

    Returns:
        (lgb.Booster, best_iter, mean, std, transform)
    """
    # Merge base params with (optional) override so a sparse override works.
    merged = dict(LGB_PARAMS)
    if params_override:
        merged.update(params_override)
    merged["seed"] = seed
    merged["metric"] = "rmse"

    # Standardize target (after optional log) so RMSE metric is meaningful
    y_log = _apply_transform(y, transform)
    y_std, mean, std = _standardize(y_log)

    best_iter = _best_iter_from_cv(
        X, y_std, fold_array, merged, num_rounds, early_stopping, seed
    )

    dset = lgb.Dataset(X, label=y_std)
    if best_iter is None or best_iter >= num_rounds:
        best_iter = num_rounds
    model = lgb.train(merged, dset, num_boost_round=best_iter)
    return model, int(best_iter), mean, std, transform


def train_all_tasks(
    X_train, y_train, mask_train, fold_train,
    X_test, y_test, mask_test,
    params=None, seed=42, num_rounds=DEFAULT_NUM_ROUNDS,
    early_stopping=DEFAULT_EARLY_STOPPING,
):
    """Train per-task LightGBM models using CV-selected boosting rounds.

    Args:
        X_train: [N, F] ALL training features (all folds).
        y_train: [N, T] ALL training targets.
        mask_train: [N, T]
        fold_train: [N] fold column.
        X_test, y_test, mask_test: test rows (unused for fitting, kept for symmetry).

    Returns:
        models (list of Booster), logs (dict), scalers (list of (transform, mean, std)).
    """
    num_tasks = y_train.shape[1]
    models = []
    logs = {}
    scalers = []
    for t in range(num_tasks):
        sel = mask_train[:, t].astype(bool)
        X_t = X_train[sel]
        y_t = y_train[sel, t]
        folds_t = fold_train[sel]

        transform = decide_transform(y_t)
        print(f"  Task {t}: samples={len(y_t)}, transform={transform}")
        model, best_iter, mean, std, transform = train_single_task(
            X_t, y_t, folds_t, transform=transform,
            params_override=params, seed=seed,
            num_rounds=num_rounds, early_stopping=early_stopping,
        )
        models.append(model)
        scalers.append((transform, mean, std))
        logs[f"task_{t}"] = {
            "samples": int(len(y_t)),
            "best_iteration": int(best_iter),
            "transform": transform,
            "target_mean": mean,
            "target_std": std,
        }
    return models, logs, scalers


def compute_val_early_stop_logs(models, X_val, y_val, mask_val):
    """Optional: evaluate models on fold-9 val for reporting (log only)."""
    return {}