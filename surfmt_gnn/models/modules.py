"""MLP building blocks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and dropout.

    Args:
        hidden_dims: List of dimensions [input_dim, h1, h2, ..., output_dim].
        activation: Activation function name ('relu', 'gelu', 'elu', 'leaky_relu').
        dropout: Dropout probability (applied after each activation except last).
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        hidden_dims: list,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert len(hidden_dims) >= 2, "MLP needs at least input and output dims"

        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        act_fn = act_map[activation.lower()]

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias))
            if i < len(hidden_dims) - 2:  # no activation/dropout after last layer
                layers.append(act_fn)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
