"""Evaluate ensemble model on test set with uncertainty quantification.

Loads all 60 models (6 seeds x 10 folds), collects predictions,
computes ensemble mean/std, and evaluates uncertainty metrics.

Usage:
    python scripts/evaluate_ensemble.py --ensemble_dir outputs/ensemble --output_dir outputs/eval
"""
import argparse
import json
import sys
from pathlib import Path
import glob

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from surfmt_gnn.config import Config
from surfmt_gnn.data.dataset import SurfProDataset
from surfmt_gnn.models.surfmt_gnn import SurfMTGNN
from surfmt_gnn.evaluation.metrics import compute_metrics


def load_model(model_path: str, config, device: str) -> SurfMTGNN:
    """Load a single model from checkpoint."""
    model = SurfMTGNN(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader, device, target_mean, target_std):
    """Collect predictions for all samples in a loader."""
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            # Denormalize
            pred_raw = pred_norm * target_std + target_mean
            all_pred.append(pred_raw.cpu().numpy())
    return np.concatenate(all_pred, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble on test set")
    parser.add_argument(
        "--ensemble_dir", type=str, default="outputs/ensemble",
        help="Directory containing ensemble models"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ensemble_eval",
        help="Output directory for results"
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    config = Config()
    if args.device:
        config.device = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Ensemble Evaluation on Test Set")
    print("=" * 60)

    # ---- Load test dataset ----
    print("\nLoading data...")
    train_ds = SurfProDataset(
        root=config.data_dir,
        csv_path=str(Path(config.data_dir) / config.train_csv),
        split="train",
    )
    test_ds = SurfProDataset(
        root=config.data_dir,
        csv_path=str(Path(config.data_dir) / config.test_csv),
        split="test",
        desc_mean=train_ds.desc_mean,
        desc_std=train_ds.desc_std,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
        temp_mean=train_ds.temp_mean,
        temp_std=train_ds.temp_std,
    )
    print(f"Test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    target_mean = torch.tensor(train_ds.target_mean, dtype=torch.float32, device=config.device)
    target_std = torch.tensor(train_ds.target_std, dtype=torch.float32, device=config.device)

    # ---- Collect ground truth ----
    all_target = []
    all_mask = []
    for batch in test_loader:
        all_target.append(batch.y_raw.numpy())
        all_mask.append(batch.mask.numpy())
    all_target = np.concatenate(all_target, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)

    # ---- Find all model checkpoints ----
    ckpt_pattern = str(Path(args.ensemble_dir) / "seed_*" / "fold_*" / "best.pt")
    ckpt_paths = sorted(glob.glob(ckpt_pattern))

    if not ckpt_paths:
        # Try alternative: just single model dirs
        ckpt_pattern = str(Path(args.ensemble_dir) / "*" / "best.pt")
        ckpt_paths = sorted(glob.glob(ckpt_pattern))

    print(f"\nFound {len(ckpt_paths)} model checkpoints")

    if len(ckpt_paths) == 0:
        print("ERROR: No model checkpoints found!")
        sys.exit(1)

    # ---- Collect predictions from all models ----
    print("\nCollecting predictions...")
    all_preds = []  # [num_models, num_samples, num_tasks]

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"  Model {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).parent.parent.name}/{Path(ckpt_path).parent.name}")
        model = load_model(ckpt_path, config, config.device)
        preds = collect_predictions(model, test_loader, config.device, target_mean, target_std)
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)  # [M, N, 6]
    print(f"Predictions shape: {all_preds.shape}")

    # ---- Ensemble statistics ----
    ensemble_mean = np.mean(all_preds, axis=0)  # [N, 6]
    ensemble_std = np.std(all_preds, axis=0, ddof=1)  # [N, 6]

    # ---- Ensemble metrics (mean predictions) ----
    print("\n" + "=" * 60)
    print("Ensemble Test Metrics")
    print("=" * 60)

    ensemble_metrics = compute_metrics(ensemble_mean, all_target, all_mask, config.task_names)

    for name in config.task_names:
        r2 = ensemble_metrics.get(f"{name}_r2", float("nan"))
        rmse = ensemble_metrics.get(f"{name}_rmse", float("nan"))
        mae = ensemble_metrics.get(f"{name}_mae", float("nan"))
        n = int(all_mask[:, config.task_names.index(name)].sum())
        print(f"  {name:12s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  (n={n})")

    print(f"\n  Average R2:   {ensemble_metrics['avg_r2']:.4f}")
    print(f"  Average RMSE: {ensemble_metrics['avg_rmse']:.4f}")
    print(f"  Average MAE:  {ensemble_metrics['avg_mae']:.4f}")

    # ---- Uncertainty analysis ----
    print("\n" + "=" * 60)
    print("Uncertainty Analysis")
    print("=" * 60)

    uncertainty_results = {}
    for t, name in enumerate(config.task_names):
        mask = all_mask[:, t].astype(bool)
        if mask.sum() < 10:
            continue

        y_true = all_target[mask, t]
        y_pred = ensemble_mean[mask, t]
        y_std = ensemble_std[mask, t]
        abs_error = np.abs(y_true - y_pred)

        # 95% confidence interval coverage
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))

        # Spearman rank correlation between std and error
        if np.std(y_std) > 1e-10 and np.std(abs_error) > 1e-10:
            spearman_r, _ = stats.spearmanr(y_std, abs_error)
        else:
            spearman_r = float("nan")

        # Error ratio: high uncertainty quartile vs low uncertainty quartile
        q1 = np.percentile(y_std, 25)
        q3 = np.percentile(y_std, 75)
        low_uncert_err = np.mean(abs_error[y_std <= q1])
        high_uncert_err = np.mean(abs_error[y_std >= q3])
        err_ratio = high_uncert_err / max(low_uncert_err, 1e-10)

        # Average ensemble std
        avg_std = np.mean(y_std)

        uncertainty_results[name] = {
            "coverage_95": float(coverage),
            "spearman_r": float(spearman_r),
            "avg_std": float(avg_std),
            "low_uncert_mae": float(low_uncert_err),
            "high_uncert_mae": float(high_uncert_err),
            "error_ratio": float(err_ratio),
        }

        print(f"\n  {name}:")
        print(f"    95% CI coverage: {coverage:.3f} (nominal: 0.950)")
        print(f"    Spearman r(std, |error|): {spearman_r:.3f}")
        print(f"    Avg ensemble std: {avg_std:.6f}")
        print(f"    Low uncert MAE:  {low_uncert_err:.6f}")
        print(f"    High uncert MAE: {high_uncert_err:.6f}")
        print(f"    Error ratio (high/low): {err_ratio:.2f}x")

    # ---- Save results ----
    results = {
        "num_models": len(ckpt_paths),
        "ensemble_metrics": ensemble_metrics,
        "uncertainty": uncertainty_results,
        "model_paths": ckpt_paths,
    }

    results_path = out_dir / "ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save predictions
    np.savez(
        out_dir / "ensemble_predictions.npz",
        predictions=all_preds,
        mean=ensemble_mean,
        std=ensemble_std,
        targets=all_target,
        mask=all_mask,
    )

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
