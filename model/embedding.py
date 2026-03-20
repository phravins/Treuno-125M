"""
Treuno 125M — Token Embedding + RoPE (Rotary Positional Embeddings)

RoPE rotates query and key vectors by position-dependent angles,
giving length generalisation beyond the training window.
This module provides:
  - TreunoEmbedding: standard nn.Embedding (32768 x 768)
  - TreunoRoPE: precomputed sin/cos cache + apply function
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import TreunoConfig


class TreunoEmbedding(nn.Module):
    """
    Token embedding table: vocab_size × hidden_size.
    In Treuno the weight of this module is SHARED with the LM head
    (tied embeddings). That share is wired in TreunoModel.__init__.
    """

    def __init__(self, config: TreunoConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hidden_size = config.hidden_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        return self.embed(token_ids)


class TreunoRoPE(nn.Module):
    """
    Rotary Positional Embedding (Su et al., 2022).

    Precomputes a frequency cache of shape (max_seq_len, head_dim)
    and applies complex-number rotation to Q and K before attention.

    Key properties:
    - Position information is baked into Q @ K dot products directly
    - Relative position preserved; works for seq_len > training length
    - No extra parameters; the sin/cos table is a buffer (not a parameter)
    """

    def __init__(self, config: TreunoConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_seq_len = config.context_length
        self.theta = config.rope_theta

        # Precompute inverse frequency vector — shape: (head_dim // 2,)
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos cache for positions 0..max_seq_len-1
        # Cache shape: (max_seq_len, head_dim)
        self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate x by 90°: splits head_dim in half, negates second half."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE rotation to query and key tensors.

        Args:
            q:            (batch, num_q_heads, seq_len, head_dim)
            k:            (batch, num_kv_heads, seq_len, head_dim)
            position_ids: (batch, seq_len) or None — defaults to 0..seq_len-1

        Returns:
            q_rot, k_rot: same shapes as input
        """
        seq_len = q.shape[2]

        # Extend cache if needed (e.g., during long inference)
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        if position_ids is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(1)  # (B,1,S,D)
            sin = self.sin_cached[position_ids].unsqueeze(1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
