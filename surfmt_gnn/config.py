"""Configuration for SurfMT-GNN."""
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
    # CSV column name -> task index mapping
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
    task_weights: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [1.0, 1.3, 1.5, 1.1, 1.3, 1.0], dtype=torch.float32
        )
    )

    # ---- Training ----
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 80
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    grad_clip_max_norm: float = 1.0
    T_0: int = 50  # CosineAnnealingWarmRestarts first cycle
    T_mult: int = 2
    eta_min: float = 1e-6

    # ---- Model: graph branch ----
    gnn_in_channels: int = 39
    gnn_edge_dim: int = 10
    gnn_hidden_dim: int = 256
    gnn_out_channels: int = 256
    gnn_num_layers: int = 3
    gnn_num_timesteps: int = 2
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1

    # ---- Model: temperature branch ----
    temp_out_dim: int = 64
    temp_hidden_dim: int = 32

    # ---- Model: descriptor branch ----
    num_descriptors: int = 12
    desc_out_dim: int = 64
    desc_hidden_dim: int = 32

    # ---- Model: fusion & shared ----
    fusion_dropout: float = 0.2
    shared_dim: int = 128

    # ---- Model: task heads ----
    head_hidden_dims: list = field(default_factory=lambda: [64, 32])

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
        # Ensure project root is set correctly when data_dir is relative
        if not Path(self.data_dir).is_absolute():
            # Assume data_dir is relative to project root (parent of surfmt_gnn/)
            project_root = Path(__file__).resolve().parent.parent
            self.data_dir = str(project_root / self.data_dir)
