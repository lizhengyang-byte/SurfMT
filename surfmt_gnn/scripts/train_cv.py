"""10-fold cross-validation training for SurfMT-GNN.

Uses the 'fold' column in surfpro_train.csv for pre-defined splits.
Trains one model per fold and saves results.

Usage:
    python scripts/train_cv.py --seed 42 --output_dir outputs/cv_seed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from surfmt_gnn.config import Config
from surfmt_gnn.data.dataset import SurfProDataset
from surfmt_gnn.models.surfmt_gnn import SurfMTGNN
from surfmt_gnn.training.trainer import Trainer
from surfmt_gnn.evaluation.metrics import compute_metrics
from surfmt_gnn.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="10-fold CV training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/cv_seed42",
        help="Output base directory"
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated fold indices to run")
    args = parser.parse_args()

    config = Config()
    config.seed = args.seed
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.patience is not None:
        config.patience = args.patience

    folds_to_run = [int(f) for f in args.folds.split(",")]

    set_seed(config.seed)

    print("=" * 60)
    print(f"SurfMT-GNN 10-Fold CV (seed={config.seed})")
    print(f"Folds to run: {folds_to_run}")
    print("=" * 60)

    # ---- Load full training dataset ----
    print("\nLoading training dataset...")
    full_train_ds = SurfProDataset(
        root=config.data_dir,
        csv_path=str(Path(config.data_dir) / config.train_csv),
        split="train",
    )
    print(f"Total samples: {len(full_train_ds)}")

    # Split by fold column
    fold_indices = {f: [] for f in range(10)}
    for i, data in enumerate(full_train_ds):
        fold = data.fold.item()
        if 0 <= fold <= 9:
            fold_indices[fold].append(i)

    for f in range(10):
        print(f"  Fold {f}: {len(fold_indices[f])} samples")

    cv_results = {}

    for fold in folds_to_run:
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}")
        print("=" * 60)

        # Split data
        val_indices = fold_indices[fold]
        train_indices = [i for f2 in range(10) if f2 != fold for i in fold_indices[f2]]

        train_subset = full_train_ds[train_indices]
        val_subset = full_train_ds[val_indices]

        print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

        # Data loaders
        train_loader = DataLoader(
            train_subset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_subset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers,
        )

        # Model (reset seed per fold for reproducibility)
        set_seed(config.seed + fold * 100)
        model = SurfMTGNN(config)

        # Train
        fold_dir = Path(args.output_dir) / f"fold_{fold}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=config.device,
            save_dir=str(fold_dir),
        )
        val_metrics = trainer.train()

        cv_results[f"fold_{fold}"] = val_metrics

        print(f"\nFold {fold} best val avg_r2: {val_metrics.get('avg_r2', 'N/A')}")

    # ---- Aggregate CV results ----
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)

    r2_per_fold = []
    for fold in folds_to_run:
        r2 = cv_results[f"fold_{fold}"].get("avg_r2", float("nan"))
        r2_per_fold.append(r2)
        print(f"  Fold {fold}: avg_r2 = {r2:.4f}")

    print(f"\n  Mean avg_r2: {np.nanmean(r2_per_fold):.4f} +/- {np.nanstd(r2_per_fold):.4f}")

    # Per-task mean
    for task in config.task_names:
        vals = [cv_results[f"fold_{f}"].get(f"{task}_r2", float("nan")) for f in folds_to_run]
        print(f"  {task:12s}: mean R2 = {np.nanmean(vals):.4f} +/- {np.nanstd(vals):.4f}")

    # Save summary
    summary = {
        "seed": config.seed,
        "folds": folds_to_run,
        "cv_results": cv_results,
        "mean_avg_r2": float(np.nanmean(r2_per_fold)),
        "std_avg_r2": float(np.nanstd(r2_per_fold)),
    }
    summary_path = Path(args.output_dir) / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
