"""Random Forest multi-task training with CV-selected tree count.

Random Forest is trained as ONE regressor per task, each using only that
task's labeled samples (the same masked-multi-task strategy as the LightGBM
baseline). Unlike gradient boosting, RF has no sequential iterations with an
early-stop point, so the number of trees is chosen per task via the predefined
10-fold column: a small sweep over `n_estimators` is cross-validated and the
best count is retained, then the model refits on ALL labeled samples.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Shared Random Forest hyperparameters tuned for small, tabular molecular data.
RF_PARAMS = {
    "max_depth": None,       # unlimited depth, rely on ensemble averaging
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42,
}

# Candidate tree counts tried per task; CV picks the best.
N_ESTIMATORS_CANDIDATES = [100, 300, 600, 1000]

# Heterogeneous ensemble: diverse hyperparameter sets decorrelate predictions.
DIVERSE_CONFIGS = [
    {  # A: default
        "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2,
    },
    {  # B: deeper, no feature sampling -> max_features=sqrt but more leaves
        "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2,
    },
    {  # C: wider feature sampling
        "max_features": 0.8, "min_samples_leaf": 2, "min_samples_split": 4,
    },
    {  # D: aggressive regularization (larger leaves, log2 features)
        "max_features": "log2", "min_samples_leaf": 5, "min_samples_split": 10,
    },
    {  # E: squared (all) features -> high variance, decorrelates from A-D
        "max_features": 1.0, "min_samples_leaf": 1, "min_samples_split": 2,
    },
]


class PredefinedFolds:
    """Iterable of (train_idx, valid_idx) built from the fold column."""

    def __init__(self, fold_array, num_folds=10):
        self.fold_array = fold_array
        self._folds = []
        for f in range(num_folds):
            val_idx = np.where(fold_array == f)[0]
            tr_idx = np.where(fold_array != f)[0]
            self._folds.append((tr_idx, val_idx))
        self._folds = [(tr, va) for tr, va in self._folds if len(va) >= 2]

    def __iter__(self):
        return iter(self._folds)

    def __len__(self):
        return len(self._folds)


def select_n_estimators(X, y, fold_array, candidates, params, seed):
    """Cross-validate candidate tree counts, return the best n_estimators.

    Falls back to the first candidate if there are too few usable folds.
    """
    params = dict(params)
    params["random_state"] = seed
    folds = PredefinedFolds(fold_array)
    if len(folds) < 2:
        return candidates[0]

    best_n = candidates[0]
    best_rmse = float("inf")
    for n in candidates:
        p = dict(params)
        p["n_estimators"] = n
        rmses = []
        for tr_idx, va_idx in folds:
            model = RandomForestRegressor(**p).fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            rmses.append(np.sqrt(mean_squared_error(y[va_idx], pred)))
        rmse = float(np.mean(rmses))
        print(f"    n_estimators={n}: CV RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_n = rmse, n
    return best_n


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


def train_single_task(
    X,
    y,
    fold_array,
    transform="none",
    params_override=None,
    candidates=N_ESTIMATORS_CANDIDATES,
    seed=42,
):
    """Train one RandomForestRegressor using CV-selected tree count, refit full data.

    Targets may be log-transformed for skewed tasks; predictions are returned
    in the model space and inverse-transformed by the caller.

    Args:
        X: [N, F] labeled feature matrix for this task.
        y: [N] raw labels for this task.
        fold_array: [N] fold index per labeled sample (0-9).
        transform: 'log' or 'none'. If 'log', the model predicts log(y).
        params_override: optional partial override dict, merged atop RF_PARAMS.

    Returns:
        (RandomForestRegressor, n_estimators, transform)
    """
    merged = dict(RF_PARAMS)
    if params_override:
        merged.update(params_override)
    merged["random_state"] = seed

    y_model = np.log(y) if transform == "log" else y

    n_estimators = select_n_estimators(X, y_model, fold_array, candidates, merged, seed)
    merged["n_estimators"] = n_estimators

    model = RandomForestRegressor(**merged).fit(X, y_model)
    return model, n_estimators, transform


def train_all_tasks(
    X_train, y_train, mask_train, fold_train,
    X_test, y_test, mask_test,
    params=None, seed=42, candidates=N_ESTIMATORS_CANDIDATES,
):
    """Train per-task Random Forest models using CV-selected tree counts.

    Args:
        X_train: [N, F] ALL training features (all folds).
        y_train: [N, T] ALL training targets.
        mask_train: [N, T]
        fold_train: [N] fold column.
        X_test, y_test, mask_test: test rows (unused for fitting, kept for symmetry).

    Returns:
        models (list of RandomForestRegressor), logs (dict), transforms (list).
    """
    num_tasks = y_train.shape[1]
    models = []
    logs = {}
    transforms = []
    for t in range(num_tasks):
        sel = mask_train[:, t].astype(bool)
        X_t = X_train[sel]
        y_t = y_train[sel, t]
        folds_t = fold_train[sel]

        transform = decide_transform(y_t)
        print(f"  Task {t}: samples={len(y_t)}, transform={transform}")
        model, n_estimators, transform = train_single_task(
            X_t, y_t, folds_t, transform=transform,
            params_override=params, candidates=candidates, seed=seed,
        )
        models.append(model)
        transforms.append(transform)
        logs[f"task_{t}"] = {
            "samples": int(len(y_t)),
            "n_estimators": int(n_estimators),
            "transform": transform,
        }
    return models, logs, transforms