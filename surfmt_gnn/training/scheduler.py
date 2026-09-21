"""Learning rate scheduler with warmup + cosine annealing with warm restarts."""
import torch


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    T_0: int = 50,
    T_mult: int = 2,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Create LR scheduler: linear warmup -> cosine annealing with warm restarts.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of warmup epochs.
        T_0: Initial cycle length for CosineAnnealingWarmRestarts.
        T_mult: Multiplicative factor for cycle length increase.
        eta_min: Minimum learning rate.

    Returns:
        SequentialLR scheduler.
    """
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
    return scheduler
