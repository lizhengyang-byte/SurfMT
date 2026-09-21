"""Masked MSE loss with per-task weighting for multi-task learning."""
import torch
import torch.nn as nn


# Default task weights from the paper:
#   pCMC=1.0, gamma_CMC=1.3, Gamma_max=1.5, A_min=1.1, pi_CMC=1.3, pC20=1.0
TASK_WEIGHTS = torch.tensor([1.0, 1.3, 1.5, 1.1, 1.3, 1.0], dtype=torch.float32)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    task_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute masked MSE loss with per-task weighting.

    Loss = sum_t( w_t * (1 / N_t) * sum_i( m_it * (y_it - y_hat_it)^2 ) )

    Args:
        pred: Predictions, shape [batch_size, num_tasks]
        target: Targets, shape [batch_size, num_tasks]
        mask: Binary mask (1=present, 0=missing), shape [batch_size, num_tasks]
        task_weights: Per-task weight tensor, shape [num_tasks], or None for uniform.

    Returns:
        Scalar loss value.
    """
    # Per-element squared error
    sq_error = (pred - target) ** 2  # [batch_size, num_tasks]

    # Zero out missing entries
    masked_sq = sq_error * mask  # [batch_size, num_tasks]

    # Per-task average (divide by number of non-missing samples per task)
    task_count = mask.sum(dim=0).clamp(min=1)  # [num_tasks], avoid div by 0
    per_task_loss = masked_sq.sum(dim=0) / task_count  # [num_tasks]

    # Apply task weights
    if task_weights is not None:
        per_task_loss = per_task_loss * task_weights

    return per_task_loss.sum()
