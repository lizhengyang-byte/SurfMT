"""Masked multi-task metrics: R2, RMSE, MAE per task and average."""
import numpy as np


TASK_COLS = ["pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "Pi_CMC", "pC20"]
TASK_NAMES = ["pCMC", "gamma_CMC", "Gamma_max", "A_min", "pi_CMC", "pC20"]


def compute_metrics(pred, target, mask, task_names=None):
    """Compute per-task and average metrics, ignoring masked entries.

    Args:
        pred: Predictions [N, num_tasks]
        target: Targets [N, num_tasks]
        mask: Binary mask [N, num_tasks]
    """
    num_tasks = pred.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(num_tasks)]

    results = {}
    for t, name in enumerate(task_names):
        m = mask[:, t].astype(bool)
        if m.sum() < 2:
            results[f"{name}_r2"] = float("nan")
            results[f"{name}_rmse"] = float("nan")
            results[f"{name}_mae"] = float("nan")
            continue
        y_true = target[m, t]
        y_pred = pred[m, t]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[f"{name}_r2"] = float(r2)
        results[f"{name}_rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        results[f"{name}_mae"] = float(np.mean(np.abs(y_true - y_pred)))

    for metric in ["r2", "rmse", "mae"]:
        vals = [results[f"{n}_{metric}"] for n in task_names
                if not np.isnan(results[f"{n}_{metric}"])]
        results[f"avg_{metric}"] = float(np.mean(vals)) if vals else float("nan")
    return results