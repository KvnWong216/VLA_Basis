from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ActionSpec:
    """Definition and normalization statistics for a Franka action chunk."""

    action_dim: int = 7
    horizon: int = 16 # The number of time steps (H) in the action sequence.
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None 
    # Calculated from training data, and initialized to 0.0/1.0 if not provided.

    def __post_init__(self) -> None:
        if self.action_dim != 7:
            raise ValueError("This project currently uses a Franka 7DoF action space.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

        self.mean = self._prepare_statistic(self.mean, "mean", default=0.0)
        self.std = self._prepare_statistic(self.std, "std", default=1.0)
        if torch.any(self.std <= 0):
            raise ValueError("Every value in std must be positive.")

    def _prepare_statistic(
        self,
        value: torch.Tensor | None, # The input value for the statistic, which can be a tensor or None.
        name: str,
        default: float,
    ) -> torch.Tensor:
        statistic = torch.full(
            (self.action_dim,), # Create a 1D tensor of shape action_dim.
            default, # Fill the tensor with the default value (0.0 for mean, 1.0 for std).
            dtype=torch.float32, 
        ) if value is None else torch.as_tensor(value, dtype=torch.float32) # Convert provided value into float32 Tensor.
        if statistic.shape != (self.action_dim,):
            raise ValueError(
                f"{name} must have shape [{self.action_dim}], "
                f"got {tuple(statistic.shape)}."
            ) # Align the shape of the statistic tensor with action_dim.
        return statistic

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize an action tensor whose final dimension is 7."""
        self._check_action_shape(action)
        mean = self.mean.to(device=action.device, dtype=action.dtype)
        std = self.std.to(device=action.device, dtype=action.dtype)
        return (action - mean) / std

    def denormalize(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Convert a normalized action tensor back to the original scale."""
        self._check_action_shape(normalized_action)
        mean = self.mean.to(
            device=normalized_action.device,
            dtype=normalized_action.dtype,
        )
        std = self.std.to(
            device=normalized_action.device,
            dtype=normalized_action.dtype,
        )
        return normalized_action * std + mean

    def _check_action_shape(self, action: torch.Tensor) -> None:
        if action.ndim < 1 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must have shape [..., {self.action_dim}], "
                f"got {tuple(action.shape)}."
            )


__all__ = ["ActionSpec"]