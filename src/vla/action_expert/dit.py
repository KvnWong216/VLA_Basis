from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Encode diffusion timesteps as continuous transformer features."""

    def __init__(self, embedding_dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim # The dimensionality of the output embedding vector for each timestep.
        self.max_period = max_period # The maximum period for the sinusoidal functions, controlling the frequency of the sine and cosine waves used in the embedding.

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [batch].")

        half_dim = self.embedding_dim // 2 # Get the (p, 2i) and (p, 2i+1) dimensions for sine and cosine components.
        frequency = torch.exp( # Frequency computation
            -math.log(self.max_period)
            * torch.arange( # Use arange to directly generate a tensor using different i values.
                half_dim,
                device=timestep.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1) 
        ) # Official formula for sinusoidal positional encoding frequency scaling used in DiT.
        angles = timestep.float()[:, None] * frequency[None, :]
        # timestep.float()[:, None] expands the timestep tensor to shape [batch, 1]
        # frequency[None, :] expands the frequency tensor to shape [1, half_dim]
        # Leverage broadcasting to compute the outer product, resulting in a tensor of shape [batch, half_dim].
        # angles_{b, i} = t_b * frenquency_i
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1) # [batch, half_dim] -> [batch, embedding_dim] by concatenating the cosine and sine components along the last dimension.
        if self.embedding_dim % 2: # If the embedding dimension is odd, append a zero vector to maintain the correct dimensionality.
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), 
                # Append a zero vector of shape [batch, 1] to the embedding tensor.
                # embedding[:, :1] sets the shape as [batch(unchanged), 1(sliced)] and zeros_like creates a copy of zero tensor with such shape. 
                dim=-1, # Add at last to maintain the correct dimensionality.
            )
        return embedding


class ActionDiTBlock(nn.Module):
    """Pre-norm self-attention, condition cross-attention, and feed-forward block."""

    def __init__(
        self,
        model_dim: int, # The dimensionality of the input and output feature vectors for the transformer block.
        condition_dim: int, # The dimension of VLM features that will be used as the condition for cross-attention.
        num_heads: int, # The number of attention heads in the multi-head attention mechanism.
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        mlp_dim = int(model_dim * mlp_ratio) # Define the hidden dimension by ratio.
        self.action_norm = nn.LayerNorm(model_dim) # Do LayerNorm first for better training stability, as suggested in the DiT paper.
        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True, # Changes format from [seq_len, batch, embed_dim] to [batch, seq_len, embed_dim].
        )
        self.condition_norm = nn.LayerNorm(model_dim)
        self.condition_key_value = nn.Linear(condition_dim, model_dim) # Convert VLM features to the same dimension as action tokens for cross-attention.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        ) # Cross-attention shares the same structure as self-attention, so only in forward process we will see the difference due to values.
        self.mlp_norm = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, model_dim),
            nn.Dropout(dropout),
        ) # Finally apply a feed-forward MLP to the output of the cross-attention, with GELU activation and dropout for regularization.

    def forward(
        self,
        action_tokens: torch.Tensor, # The input is projected from the noisy action tensor, with shape [batch, horizon, model_dim].
        condition: torch.Tensor, # The condition tensor is projected from VLM features, with shape [batch, length, model_dim].
        condition_mask: torch.Tensor | None = None, # To deal with variable-length conditions with paddings in some tensors.
    ) -> torch.Tensor:
        normalized_actions = self.action_norm(action_tokens)
        self_attention_output, _ = self.self_attention(
            normalized_actions,
            normalized_actions,
            normalized_actions,
        ) # Self-attention uses the same action tokens as Q, K and V...
        action_tokens = action_tokens + self_attention_output # Residual connection.

        normalized_actions = self.condition_norm(action_tokens)
        normalized_condition = self.condition_key_value(condition)
        cross_attention_output, _ = self.cross_attention(
            normalized_actions,
            normalized_condition,
            normalized_condition,
            key_padding_mask=self._padding_mask(condition_mask),
        ) # ... while cross-attention uses the condition as K and V, and action tokens as Q.
        action_tokens = action_tokens + cross_attention_output # Residual connection.
        return action_tokens + self.mlp(self.mlp_norm(action_tokens))
    
    @staticmethod
    def _padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim != 2: # mask should have shape [batch, condition_length], where each element is 0 for padding and 1 for valid tokens.
            raise ValueError("condition_mask must have shape [batch, condition_length].")
        return mask == 0 # Switch to boolean mask with True for padding (elements == 0).


class ActionDiT(nn.Module):
    """Conditional diffusion transformer for continuous Franka action chunks."""

    def __init__(
        self,
        action_dim: int = 7,
        model_dim: int = 512,
        condition_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or model_dim <= 0 or condition_dim <= 0:
            raise ValueError("action_dim, model_dim, and condition_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.action_dim = action_dim
        self.model_dim = model_dim
        self.condition_dim = condition_dim
        self.action_embedding = nn.Linear(action_dim, model_dim)
        self.timestep_embedding = SinusoidalTimestepEmbedding(model_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ActionDiTBlock(
                    model_dim=model_dim,
                    condition_dim=condition_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.action_head = nn.Linear(model_dim, action_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a diffusion target with shape ``[batch, horizon, action_dim]``."""
        self._check_inputs(noisy_action, timestep, condition, condition_mask)
        action_tokens = self.action_embedding(noisy_action)
        time_tokens = self.timestep_mlp(self.timestep_embedding(timestep))
        action_tokens = action_tokens + time_tokens[:, None, :]

        for block in self.blocks:
            action_tokens = block(action_tokens, condition, condition_mask)

        return self.action_head(self.output_norm(action_tokens))

    def _check_inputs(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None,
    ) -> None:
        if noisy_action.ndim != 3 or noisy_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"noisy_action must have shape [batch, horizon, {self.action_dim}]."
            )
        if timestep.ndim != 1 or timestep.shape[0] != noisy_action.shape[0]:
            raise ValueError("timestep must have shape [batch].")
        if condition.ndim != 3 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape [batch, length, {self.condition_dim}]."
            )
        if condition.shape[0] != noisy_action.shape[0]:
            raise ValueError("noisy_action and condition must have the same batch size.")
        if condition_mask is not None:
            if condition_mask.shape != condition.shape[:2]:
                raise ValueError("condition_mask must match condition's first two dimensions.")


__all__ = ["ActionDiT", "ActionDiTBlock", "SinusoidalTimestepEmbedding"]