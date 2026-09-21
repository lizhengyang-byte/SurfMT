"""SurfMT-GNN: Multi-task graph neural network for surfactant property prediction.

Three-branch architecture:
  1. Graph encoder (AttentiveFP) -> z_graph (256-dim)
  2. Temperature encoder (MLP) -> z_temp (64-dim)
  3. Descriptor encoder (MLP) -> z_desc (64-dim)

Feature fusion -> shared representation -> 6 task heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MLP
from .attentive_fp import MultiHeadAttentiveFP


class SurfMTGNN(nn.Module):
    """SurfMT-GNN model.

    Args:
        config: Config object with model hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- Branch 1: Graph encoder (Multi-Head AttentiveFP) ----
        self.graph_encoder = MultiHeadAttentiveFP(
            in_channels=config.gnn_in_channels,
            hidden_channels=config.gnn_hidden_dim,
            out_channels=config.gnn_out_channels,
            edge_dim=config.gnn_edge_dim,
            num_layers=config.gnn_num_layers,
            num_timesteps=config.gnn_num_timesteps,
            num_heads=config.gnn_num_heads,
            dropout=config.gnn_dropout,
        )

        # ---- Branch 2: Temperature encoder MLP (1 -> 32 -> 64, GELU) ----
        self.temp_encoder = MLP(
            hidden_dims=[1, config.temp_hidden_dim, config.temp_out_dim],
            activation="gelu",
            dropout=0.0,
        )

        # ---- Branch 3: Descriptor encoder MLP (12 -> 32 -> 64, ReLU) ----
        self.desc_encoder = MLP(
            hidden_dims=[config.num_descriptors, config.desc_hidden_dim, config.desc_out_dim],
            activation="relu",
            dropout=0.0,
        )

        # ---- Fusion: 384 -> 256, dropout=0.2 ----
        fusion_in = config.gnn_out_channels + config.temp_out_dim + config.desc_out_dim
        self.fusion_mlp = MLP(
            hidden_dims=[fusion_in, config.gnn_out_channels],
            activation="relu",
            dropout=config.fusion_dropout,
        )

        # ---- Shared layer: 256 -> 128 ----
        self.shared_layer = nn.Linear(config.gnn_out_channels, config.shared_dim)
        self.shared_norm = nn.LayerNorm(config.shared_dim)

        # ---- Task heads: 6 independent MLP heads (128 -> 64 -> 32 -> 1) ----
        head_dims = [config.shared_dim] + config.head_hidden_dims + [1]
        self.task_heads = nn.ModuleList([
            MLP(hidden_dims=head_dims, activation="relu", dropout=config.head_dropout)
            for _ in range(config.num_tasks)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming for ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Batch object with keys:
                x, edge_index, edge_attr, batch,
                temp_norm, temp_mask, descriptors

        Returns:
            predictions: [batch_size, num_tasks]
        """
        # ---- Graph branch ----
        z_graph = self.graph_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        # z_graph: [batch_size, 256]

        # ---- Temperature branch (with mask for missing temps) ----
        temp_norm = data.temp_norm.unsqueeze(-1) if data.temp_norm.dim() == 1 else data.temp_norm
        # Ensure shape [batch_size, 1]
        temp_norm = temp_norm.view(-1, 1)
        z_temp = self.temp_encoder(temp_norm)  # [batch_size, 64]

        # Apply mask: zero out temp embedding when temperature is missing
        temp_mask = data.temp_mask.view(-1, 1)  # [batch_size, 1]
        z_temp = z_temp * temp_mask

        # ---- Descriptor branch ----
        desc = data.descriptors
        if desc.dim() == 1:
            desc = desc.view(-1, self.config.num_descriptors)
        z_desc = self.desc_encoder(desc)  # [batch_size, 64]

        # ---- Fusion ----
        z_concat = torch.cat([z_graph, z_temp, z_desc], dim=-1)  # [batch_size, 384]
        z_fused = self.fusion_mlp(z_concat)  # [batch_size, 256]
        z_shared = F.relu(self.shared_norm(self.shared_layer(z_fused)))  # [batch_size, 128]

        # ---- Task heads ----
        preds = torch.cat([head(z_shared) for head in self.task_heads], dim=-1)
        # preds: [batch_size, 6]

        return preds
