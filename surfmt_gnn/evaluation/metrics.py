"""Evaluation metrics: R2, RMSE, MAE per task and average."""
import numpy as np


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    task_names: list = None,
) -> dict:
    """Compute R2, RMSE, MAE per task and averaged.

    Args:
        pred: Predictions, shape [N, num_tasks]
        target: Targets, shape [N, num_tasks]
        mask: Binary mask, shape [N, num_tasks]
        task_names: List of task name strings (optional).

    Returns:
        Dict with per-task and averaged metrics.
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

        # R2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        results[f"{name}_r2"] = float(r2)
        results[f"{name}_rmse"] = float(rmse)
        results[f"{name}_mae"] = float(mae)

    # Average over tasks with valid metrics
    for metric in ["r2", "rmse", "mae"]:
        vals = [
            results[f"{n}_{metric}"]
            for n in task_names
            if not np.isnan(results[f"{n}_{metric}"])
        ]
        results[f"avg_{metric}"] = float(np.mean(vals)) if vals else float("nan")

    return results
