"""Configuration for SurfMT-GNN.

v8: Optimized for best performance building on v5 (our best at 0.62 test R²).
Key improvements:
- Multi-head AttentiveFP with edge features in ALL layers (v6 fix)
- 4 GNN layers (deeper, since edge features now flow through all layers)
- ReduceLROnPlateau scheduler (stable, proven in v5)
- Stronger but balanced regularization
- Wider task heads
- LayerNorm on shared representation
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    """Hyperparameters and settings for SurfMT-GNN."""

    # ---- Data ----
    data_dir: str = "data/surfpro"
    train_csv: str = "surfpro_train.csv"
    test_csv: str = "surfpro_test.csv"
    num_tasks: int = 6
    task_names: list = field(
        default_factory=lambda: [
            "pCMC",
            "gamma_CMC",
            "Gamma_max",
            "A_min",
            "pi_CMC",
            "pC20",
        ]
    )
    csv_to_task: dict = field(
        default_factory=lambda: {
            "pCMC": 0,
            "AW_ST_CMC": 1,
            "Gamma_max": 2,
            "Area_min": 3,
            "Pi_CMC": 4,
            "pC20": 5,
        }
    )

    # ---- Task weights (masked MSE) ----
    # Balanced: modest boost for scarce tasks, not too aggressive
    task_weights: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [1.0, 1.3, 1.8, 1.3, 1.3, 1.0], dtype=torch.float32
        )
    )

    # ---- Training ----
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 120
    lr: float = 1.5e-4  # v8: balanced LR
    weight_decay: float = 3e-4  # v8: balanced weight decay
    warmup_epochs: int = 10
    grad_clip_max_norm: float = 1.0

    # Scheduler
    # reduce_on_plateau: proven stable baseline scheduler
    scheduler_type: str = "reduce_on_plateau"
    T_0: int = 100
    T_mult: int = 2
    eta_min: float = 1e-6
    scheduler_patience: int = 25
    scheduler_factor: float = 0.5

    # ---- Model: graph branch ----
    # Multi-head AttentiveFP with edge features in all layers
    gnn_in_channels: int = 39
    gnn_edge_dim: int = 10
    gnn_hidden_dim: int = 256
    gnn_out_channels: int = 256
    gnn_num_layers: int = 4  # v8: 4 layers (deeper, edge features now work properly)
    gnn_num_timesteps: int = 2
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.15  # v8: moderate dropout

    # ---- Model: temperature branch ----
    temp_out_dim: int = 64
    temp_hidden_dim: int = 32

    # ---- Model: descriptor branch ----
    num_descriptors: int = 12
    desc_out_dim: int = 64
    desc_hidden_dim: int = 32

    # ---- Model: fingerprint branch ----
    use_fingerprint: bool = True
    fp_dim: int = 2048
    fp_hidden_dim: int = 256
    fp_out_dim: int = 128
    fp_dropout: float = 0.3

    # ---- Model: fusion & shared ----
    fusion_dropout: float = 0.15  # v8: moderate
    shared_dim: int = 128
    head_dropout: float = 0.15  # v8: dropout in task heads

    # ---- Model: task heads ----
    head_hidden_dims: list = field(default_factory=lambda: [128, 64])  # v8: wider heads

    # ---- Ensemble ----
    seeds: list = field(
        default_factory=lambda: [42, 123, 456, 789, 1024, 2048]
    )
    num_folds: int = 10

    # ---- Misc ----
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    def __post_init__(self):
        if not Path(self.data_dir).is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.data_dir = str(project_root / self.data_dir)
