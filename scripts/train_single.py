"""Train a single SurfMT-GNN model and evaluate on test set.

Usage:
    python scripts/train_single.py --seed 42 --output_dir outputs/single_seed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from surfmt_gnn.config import Config
from surfmt_gnn.data.dataset import SurfProDataset
from surfmt_gnn.models.surfmt_gnn import SurfMTGNN
from surfmt_gnn.training.trainer import Trainer
from surfmt_gnn.evaluation.metrics import compute_metrics
from surfmt_gnn.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train single SurfMT-GNN model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/single_seed42",
        help="Output directory for checkpoints and results"
    )
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    args = parser.parse_args()

    # Config
    config = Config()
    config.seed = args.seed
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.patience is not None:
        config.patience = args.patience

    # Seed
    set_seed(config.seed)

    print("=" * 60)
    print("SurfMT-GNN Single Model Training")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # ---- Load data ----
    print("\nLoading datasets...")
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

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # ---- Split train into train/val (90/10, based on fold column) ----
    # Use fold 9 as validation, rest as training
    train_indices = []
    val_indices = []
    for i, data in enumerate(train_ds):
        fold = data.fold.item()
        if fold == 9:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_subset = train_ds[train_indices]
    val_subset = train_ds[val_indices]

    print(f"Train split: {len(train_subset)} | Val split: {len(val_subset)}")

    # ---- Data loaders ----
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    # ---- Model ----
    print("\nInitializing model...")
    model = SurfMTGNN(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---- Train ----
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device,
        save_dir=args.output_dir,
    )
    val_metrics = trainer.train()

    # ---- Test evaluation ----
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    model.eval()
    all_pred = []
    all_target = []
    all_mask = []

    target_mean = torch.tensor(train_ds.target_mean, dtype=torch.float32, device=config.device)
    target_std = torch.tensor(train_ds.target_std, dtype=torch.float32, device=config.device)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.device)
            pred_norm = model(batch)
            # Denormalize
            pred_raw = pred_norm * target_std + target_mean
            all_pred.append(pred_raw.cpu().numpy())
            all_target.append(batch.y_raw.cpu().numpy())
            all_mask.append(batch.mask.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)

    test_metrics = compute_metrics(all_pred, all_target, all_mask, config.task_names)

    print("\nTest metrics:")
    for name in config.task_names:
        r2 = test_metrics.get(f"{name}_r2", float("nan"))
        rmse = test_metrics.get(f"{name}_rmse", float("nan"))
        mae = test_metrics.get(f"{name}_mae", float("nan"))
        n_samples = int(all_mask[:, config.task_names.index(name)].sum())
        print(f"  {name:12s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  (n={n_samples})")

    print(f"\n  Average R2:   {test_metrics['avg_r2']:.4f}")
    print(f"  Average RMSE: {test_metrics['avg_rmse']:.4f}")
    print(f"  Average MAE:  {test_metrics['avg_mae']:.4f}")

    # Save test results
    test_results = {
        "seed": config.seed,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "num_params": total_params,
    }
    out_path = Path(args.output_dir) / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
