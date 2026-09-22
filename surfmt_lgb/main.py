"""SurfMT-LightGBM main entry point.

Trains per-task LightGBM regressors (features = ECFP4 + 12 descriptors +
temperature). Correct boosting-round count per task is chosen via 10-fold
cross-validation on the predefined fold column; each task then refits on
ALL its labeled training samples. Finally evaluates on the test set.

Usage:
    python main.py                          # single seed
    python main.py --seed 123               # specify seed
    python main.py --output_dir outputs/lgb --max_rounds 5000
    python main.py --seeds 42,123,456       # multi-seed ensemble (averaged)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as `python surfmt_lgb/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surfmt_lgb.data import load_data
from surfmt_lgb.train import train_all_tasks, LGB_PARAMS, DIVERSE_CONFIGS
from surfmt_lgb.metrics import compute_metrics, TASK_NAMES, TASK_COLS


def run_single_seed(train, test, seed, max_rounds, early_stopping, params_override=None):
    """Train one set of per-task models for a given seed and optional param set."""
    models, logs, scalers = train_all_tasks(
        train.X, train.y, train.mask, train.fold,
        test.X, test.y, test.mask,
        params=params_override, seed=seed,
        num_rounds=max_rounds, early_stopping=early_stopping,
    )

    # Predict, then invert the target transform and standardization per task
    pred_std = np.stack([m.predict(test.X) for m in models], axis=1)
    denorm = np.zeros_like(pred_std)
    for t, (transform, mean, std) in enumerate(scalers):
        y_trans = pred_std[:, t] * std + mean
        if transform == "log":
            denorm[:, t] = np.exp(y_trans)
        else:
            denorm[:, t] = y_trans
    return models, logs, scalers, denorm


def main():
    parser = argparse.ArgumentParser(description="SurfMT-LightGBM training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for ensemble (e.g. '42,123,456'). "
             "Overrides --seed.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/lgb_seed42")
    parser.add_argument("--max_rounds", type=int, default=5000)
    parser.add_argument("--early_stopping", type=int, default=100)
    parser.add_argument(
        "--hetero", action="store_true",
        help="Heterogeneous ensemble: train one model per DIVERSE_CONFIGS "
             "param set and average predictions (decorrelates members).",
    )
    args = parser.parse_args()

    if args.hetero:
        # Heterogeneous ensemble: each member uses a different hyperparam set.
        members = DIVERSE_CONFIGS
        # Give each member a distinct seed too for extra decorrelation.
        member_seeds = [42, 101, 202, 303, 404][:len(members)]
    elif args.seeds:
        try:
            seeds = [int(s) for s in args.seeds.split(",")]
        except ValueError:
            seeds = [args.seed]
        members = [None] * len(seeds)
        member_seeds = seeds
    else:
        members = [None]
        member_seeds = [args.seed]

    print("=" * 60)
    print("SurfMT-LightGBM (per-task regression, CV-selected rounds)")
    if args.hetero:
        print(f"Heterogeneous ensemble: {len(members)} diverse configs")
    else:
        print(f"Seeds: {member_seeds}")
    print("=" * 60)

    # ---- Load data ----
    print("\nLoading data...")
    train, test = load_data()
    print(f"Training features: {train.X.shape[1]}")

    # ---- Train each ensemble member ----
    test_preds = []  # list of (denormalized pred_array per member)
    all_logs = {}
    member_names = []
    for i, (params_override, seed) in enumerate(zip(members, member_seeds)):
        if args.hetero:
            name = f"cfg{i}"
        else:
            name = f"seed_{seed}"
        print(f"\nTrain {name}:")
        models, logs, scalers, pred = run_single_seed(
            train, test, seed, args.max_rounds, args.early_stopping, params_override)
        test_preds.append(pred)
        all_logs[name] = logs
        member_names.append(name)

    # ---- Evaluate test ----
    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)  # average over seeds
    test_metrics = compute_metrics(test_pred, test.y, test.mask, TASK_NAMES)

    print("\nTest metrics:")
    for name in TASK_NAMES:
        r2 = test_metrics.get(f"{name}_r2", float("nan"))
        rmse = test_metrics.get(f"{name}_rmse", float("nan"))
        mae = test_metrics.get(f"{name}_mae", float("nan"))
        n = int(test.mask[:, TASK_NAMES.index(name)].sum())
        print(f"  {name:12s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  (n={n})")
    print(f"\n  Average R2:   {test_metrics['avg_r2']:.4f}")
    print(f"  Average RMSE: {test_metrics['avg_rmse']:.4f}")
    print(f"  Average MAE:  {test_metrics['avg_mae']:.4f}")

    # ---- Per-member metrics for comparison ----
    print("\nPer-member Average R2:")
    for name, pred in zip(member_names, test_preds):
        m = compute_metrics(pred, test.y, test.mask, TASK_NAMES)
        print(f"  {name}: {m['avg_r2']:.4f}")

    # ---- Save results ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "members": member_names,
        "hetero": args.hetero,
        "num_members": len(member_names),
        "num_features": int(train.X.shape[1]),
        "params": LGB_PARAMS,
        "logs": all_logs,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'test_results.json'}")

    return test_metrics


if __name__ == "__main__":
    main()