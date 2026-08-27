from __future__ import annotations

import torch
from torch import nn # PyTorch's neural network module, providing layers, activations, and utilities for building deep learning models.


class ConditionProjector(nn.Module):
    """Project VLM features into the action expert embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        hidden_dim = hidden_dim or output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        ) # Construct a 2-layer feedforward neural network with GELU activation.

    def forward(self, vlm_features: torch.Tensor) -> torch.Tensor:
        """Map ``[B, L, input_dim]`` to ``[B, L, output_dim]``."""
        if vlm_features.ndim != 3:
            raise ValueError(
                "vlm_features must have shape [batch, sequence, feature_dim]."
            )
        if vlm_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature_dim={self.input_dim}, "
                f"got {vlm_features.shape[-1]}."
            ) # The last dimension of vlm_features must match the input_dim specified during initialization.

        parameter = next(self.projector.parameters()) # Get the first parameter of the projector.
        vlm_features = vlm_features.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ) # Align the device and data type of vlm_features with the projector's parameters for consistent computation.
        return self.projector(vlm_features) # Forward process.


__all__ = ["ConditionProjector"]