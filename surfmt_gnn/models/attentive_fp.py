"""Multi-head AttentiveFP graph encoder.

Custom implementation based on the AttentiveFP algorithm (Xiong et al., 2020)
with multi-head attention support. Architecture:
  - Input projection: atom features -> hidden dim
  - GATEConv first layer (incorporates edge features)
  - GATConv message passing layers (multi-head, with edge features, with GRU)
  - Super-node (virtual node) readout with multiple timesteps
  - Output projection to graph embedding

v2 fixes:
  - All GATConv layers now use edge features (edge_dim parameter)
  - Consistent activation (ELU for conv layers, ReLU for GRU output)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.nn.models.attentive_fp import GATEConv


class MultiHeadAttentiveFP(nn.Module):
    """Multi-head AttentiveFP graph encoder.

    Args:
        in_channels: Input atom feature dimension.
        hidden_channels: Hidden dimension size.
        out_channels: Output graph embedding dimension.
        edge_dim: Edge feature dimension.
        num_layers: Number of message passing layers.
        num_timesteps: Number of readout timesteps (super-node iterations).
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 39,
        hidden_channels: int = 256,
        out_channels: int = 256,
        edge_dim: int = 10,
        num_layers: int = 3,
        num_timesteps: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.num_heads = num_heads
        self.dropout = dropout

        # ---- Atom embedding ----
        self.lin_in = nn.Linear(in_channels, hidden_channels)

        # ---- Message passing layers ----
        # Layer 0: GATEConv (handles edge features with gating mechanism)
        self.conv0 = GATEConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gru0 = nn.GRUCell(hidden_channels, hidden_channels)

        # Layers 1..num_layers-1: GATConv (multi-head, with edge features)
        # v2: edge_dim=edge_dim so edge features inform attention weights
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=True,  # v2: enable self-loops (standard in GAT)
                edge_dim=edge_dim,  # v2: use edge features for attention
            )
            self.convs.append(conv)
            self.grus.append(nn.GRUCell(hidden_channels, hidden_channels))

        # ---- Readout (super-node) ----
        # Virtual super-node connected to all atoms
        # Readout doesn't have edge features (virtual edges), so no edge_dim
        self.readout_convs = nn.ModuleList()
        self.readout_grus = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readout_convs.append(
                GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
            self.readout_grus.append(nn.GRUCell(hidden_channels, hidden_channels))

        # ---- Output ----
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_in.weight)
        nn.init.zeros_(self.lin_in.bias)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass.

        Args:
            x: Atom features [num_atoms, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch vector [num_atoms]

        Returns:
            Graph embeddings [batch_size, out_channels]
        """
        # Input projection
        h = F.leaky_relu(self.lin_in(x))

        # ---- Message passing: Layer 0 (GATEConv with edge features) ----
        h_conv = F.elu(self.conv0(h, edge_index, edge_attr))
        h_conv = F.dropout(h_conv, p=self.dropout, training=self.training)
        h = self.gru0(h_conv, h)
        h = F.relu(h)

        # ---- Message passing: Layers 1..N-1 (GATConv multi-head with edge features) ----
        # v2: pass edge_attr to all conv layers
        for conv, gru in zip(self.convs, self.grus):
            h_conv = F.elu(conv(h, edge_index, edge_attr))
            h_conv = F.dropout(h_conv, p=self.dropout, training=self.training)
            h = gru(h_conv, h)
            h = F.relu(h)

        # ---- Readout: super-node attention ----
        # Initialize super-node state as sum of all atom states per graph
        super_node = global_add_pool(h, batch)  # [batch_size, hidden]

        for t in range(self.num_timesteps):
            # Build edges: super-node <-> all atoms
            num_atoms = h.size(0)
            batch_size = super_node.size(0)

            # New node indices for super nodes
            super_indices = torch.arange(
                num_atoms, num_atoms + batch_size, device=x.device
            )

            # Map each atom to its super-node index
            super_per_atom = super_indices[batch]  # [num_atoms]

            # Bidirectional edges between atoms and their super-node
            edge_index_atom_to_super = torch.stack(
                [torch.arange(num_atoms, device=x.device), super_per_atom], dim=0
            )
            edge_index_super_to_atom = torch.stack(
                [super_per_atom, torch.arange(num_atoms, device=x.device)], dim=0
            )
            edge_index_readout = torch.cat(
                [edge_index_atom_to_super, edge_index_super_to_atom], dim=1
            )

            # Combine atom states with super-node states
            h_combined = torch.cat([h, super_node], dim=0)  # [N+B, hidden]

            # GATConv message passing (no edge features for virtual edges)
            h_readout = F.elu(
                self.readout_convs[t](h_combined, edge_index_readout)
            )
            h_readout = F.dropout(
                h_readout, p=self.dropout, training=self.training
            )

            # Update super-node states with GRU
            new_super = h_readout[num_atoms:]  # [batch_size, hidden]
            super_node = self.readout_grus[t](new_super, super_node)
            super_node = F.relu(super_node)

        # Output projection
        out = self.lin_out(super_node)  # [batch_size, out_channels]
        return out
