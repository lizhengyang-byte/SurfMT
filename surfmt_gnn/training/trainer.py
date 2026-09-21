"""Trainer class for SurfMT-GNN with early stopping and checkpointing."""
from pathlib import Path
import json

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .loss import masked_mse_loss
from .scheduler import get_scheduler
from ..evaluation.metrics import compute_metrics


class Trainer:
    """Trainer for multi-task GNN.

    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Config object.
        device: 'cuda' or 'cpu'.
        save_dir: Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        save_dir: str = "outputs/exp01",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            warmup_epochs=config.warmup_epochs,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            scheduler_type=config.scheduler_type,
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            total_epochs=config.max_epochs,
        )
        self.scheduler_type = config.scheduler_type

        # Loss
        self.criterion = masked_mse_loss
        self.task_weights = config.task_weights.to(device)

        # Target scaler (for denormalizing predictions)
        self.target_mean = None
        self.target_std = None

        # Try to infer target scaler from train dataset
        # Handle both Dataset and Subset wrappers
        train_ds = train_loader.dataset
        if hasattr(train_ds, 'target_mean') and train_ds.target_mean is not None:
            self.target_mean = torch.tensor(train_ds.target_mean, dtype=torch.float32, device=device)
            self.target_std = torch.tensor(train_ds.target_std, dtype=torch.float32, device=device)
        elif hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'target_mean'):
            self.target_mean = torch.tensor(train_ds.dataset.target_mean, dtype=torch.float32, device=device)
            self.target_std = torch.tensor(train_ds.dataset.target_std, dtype=torch.float32, device=device)

        # Training state
        self.best_val_loss = float("inf")
        self.best_val_r2 = float("-inf")
        self.early_stop_metric = "avg_r2"  # use avg_r2 for early stopping (higher is better)
        self.epochs_no_improve = 0
        self.epoch = 0
        self.history = []

    def train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss per sample.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Model predicts normalized targets; loss uses normalized targets
            pred = self.model(batch)
            loss = self.criterion(pred, batch.y, batch.mask, self.task_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip_max_norm
            )
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation and compute metrics on original (denormalized) scale.

        Returns:
            Dict with 'loss' (normalized), and metrics (r2, rmse, mae) on original scale.
        """
        self.model.eval()

        all_pred_norm = []
        all_target_raw = []
        all_mask = []
        total_loss = 0.0
        total_samples = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            pred_norm = self.model(batch)

            # Loss on normalized targets
            loss = self.criterion(pred_norm, batch.y, batch.mask, self.task_weights)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

            # Denormalize predictions for metric computation
            if self.target_mean is not None and self.target_std is not None:
                pred_raw = pred_norm * self.target_std + self.target_mean
            else:
                pred_raw = pred_norm

            all_pred_norm.append(pred_norm.cpu().numpy())
            all_target_raw.append(batch.y_raw.cpu().numpy())
            all_mask.append(batch.mask.cpu().numpy())

        all_pred_raw = np.concatenate(all_pred_norm, axis=0)
        # If we have scalers, denormalize
        if self.target_mean is not None and self.target_std is not None:
            all_pred_raw = all_pred_raw * self.target_std.cpu().numpy() + self.target_mean.cpu().numpy()
        all_target_raw = np.concatenate(all_target_raw, axis=0)
        all_mask = np.concatenate(all_mask, axis=0)

        avg_loss = total_loss / max(total_samples, 1)

        # Metrics on original scale
        metrics = compute_metrics(all_pred_raw, all_target_raw, all_mask, self.config.task_names)
        metrics["loss"] = avg_loss

        return metrics

    def train(self) -> dict:
        """Full training loop with early stopping.

        Returns:
            Best validation metrics dict.
        """
        print(f"Starting training for max {self.config.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("-" * 60)

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics["loss"]

            # LR scheduler step
            if self.scheduler_type == "reduce_on_plateau":
                # ReduceLROnPlateau uses validation metric
                self.scheduler.step(val_metrics.get("avg_r2", 0.0))
            else:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "avg_r2": val_metrics.get("avg_r2", float("nan")),
            }
            self.history.append(record)

            # Log every 10 epochs or first/last
            if epoch % 10 == 0 or epoch < 5 or self.epochs_no_improve == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"avg_r2={val_metrics.get('avg_r2', 0):.4f} | "
                    f"lr={current_lr:.6f}"
                )

            # Early stopping check (based on avg_r2, higher is better)
            current_r2 = val_metrics.get("avg_r2", float("nan"))
            if not np.isnan(current_r2) and current_r2 > self.best_val_r2:
                self.best_val_r2 = current_r2
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint("best.pt")
                # Save best metrics
                best_info = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_avg_r2": current_r2,
                    "metrics": val_metrics,
                }
                with open(self.save_dir / "best_metrics.json", "w") as f:
                    json.dump(best_info, f, indent=2, default=str)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.config.patience})")
                    break

        # Load best model and return its metrics
        self.load_checkpoint("best.pt")
        final_metrics = self.validate()

        # Save training history
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print("-" * 60)
        print("Training finished. Best validation metrics:")
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        return final_metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
