"""Learning rate schedulers."""
import torch


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    T_0: int = 100,
    T_mult: int = 1,
    eta_min: float = 1e-6,
    scheduler_type: str = "cosine_restart",
    patience: int = 20,
    factor: float = 0.5,
    total_epochs: int = 500,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create LR scheduler.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of warmup epochs (used with cosine_restart).
        T_0: Initial cycle length for CosineAnnealingWarmRestarts.
        T_mult: Cycle length multiplier.
        eta_min: Minimum learning rate.
        scheduler_type: 'cosine_restart' or 'reduce_on_plateau' or 'cosine'.
        patience: Patience for ReduceLROnPlateau.
        factor: Reduction factor for ReduceLROnPlateau.

    Returns:
        LR scheduler.
    """
    if scheduler_type == "cosine_restart":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs,
        )
        cos_restart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cos_restart],
            milestones=[warmup_epochs],
        )
    elif scheduler_type == "cosine":
        # Cosine annealing without restarts (single cycle)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs,
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cos],
            milestones=[warmup_epochs],
        )
    elif scheduler_type == "reduce_on_plateau":
        # Note: this scheduler requires step(metric) instead of step()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # maximize avg_r2
            factor=factor,
            patience=patience,
            min_lr=eta_min,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler
