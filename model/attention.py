"""
Treuno 125M — Grouped Query Attention (GQA) with RoPE

Architecture:
  - 12 query heads, 4 KV heads  (3:1 ratio)
  - Each KV head serves 3 query heads
  - RoPE applied to Q and K before softmax
  - Causal mask (upper-triangular) always on
  - No bias anywhere (bias=False)
  - KV cache support for incremental decoding

Memory savings vs MHA:
  - KV projections: 768x(4x64) instead of 768x(12x64) — 3x smaller
  - KV cache grows at 1/3 the rate during long generation
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import TreunoConfig
from .embedding import TreunoRoPE


class TreunoAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    For Treuno-125M:
        num_q_heads  = 12
        num_kv_heads = 4
        head_dim     = 64
        groups       = 12 / 4 = 3  →  each KV head serves 3 Q heads
    """

    def __init__(self, config: TreunoConfig):
        super().__init__()
        self.config = config
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.groups = config.num_q_heads // config.num_kv_heads  # 3

        # ── Projections ──────────────────────────────────────────────────────
        # Q: hidden → (num_q_heads × head_dim)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_q_heads * self.head_dim,
            bias=config.bias,
        )
        # K, V: hidden → (num_kv_heads × head_dim)  ← smaller because GQA
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.bias,
        )
        # Output projection: (num_q_heads × head_dim) → hidden
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            self.hidden_size,
            bias=config.bias,
        )

        self.rope = TreunoRoPE(config)
        self.scale = math.sqrt(self.head_dim)

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Expand KV from (B, num_kv_heads, S, D) to (B, num_q_heads, S, D)
        by repeating each KV head `groups` times.
        """
        B, num_kv, S, D = kv.shape
        # (B, num_kv, 1, S, D) → (B, num_kv, groups, S, D) → (B, num_q, S, D)
        kv = kv.unsqueeze(2).expand(B, num_kv, self.groups, S, D)
        return kv.reshape(B, self.num_q_heads, S, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, S, H)
            attention_mask: (B, 1, S, S) additive mask (0 or -inf), optional
            position_ids:   (B, S), defaults to 0..S-1
            past_key_value: cached (K, V) from previous steps, for generation
            use_cache:      if True, return updated KV cache

        Returns:
            output:          (B, S, H)
            new_key_value:   (K, V) tensors or None
        """
        B, S, H = hidden_states.shape

        # ── Project ──────────────────────────────────────────────────────────
        q = self.q_proj(hidden_states)  # (B, S, num_q×head_dim)
        k = self.k_proj(hidden_states)  # (B, S, num_kv×head_dim)
        v = self.v_proj(hidden_states)  # (B, S, num_kv×head_dim)

        # ── Reshape to (B, heads, S, head_dim) ───────────────────────────────
        q = q.view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── Apply RoPE ───────────────────────────────────────────────────────
        q, k = self.rope(q, k, position_ids=position_ids)

        # ── KV-cache for incremental generation ──────────────────────────────
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_key_value = (k, v) if use_cache else None
        full_seq_len = k.shape[2]

        # ── Expand KV to match Q heads (GQA) ─────────────────────────────────
        k_exp = self._expand_kv(k)   # (B, num_q_heads, full_S, head_dim)
        v_exp = self._expand_kv(v)

        # ── Scaled dot-product attention ─────────────────────────────────────
        # Use PyTorch 2.x fused implementation when available
        if hasattr(F, "scaled_dot_product_attention"):
            # Build causal mask
            attn_output = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None),
                scale=1.0 / self.scale,
            )
        else:
            # Manual fallback
            attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) / self.scale
            # Causal mask
            causal_mask = torch.triu(
                torch.full((S, full_seq_len), float("-inf"), device=q.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v_exp)

        # ── Merge heads and project ───────────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, S, num_q×head_dim)
        attn_output = attn_output.view(B, S, self.num_q_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_key_value
