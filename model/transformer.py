"""
Treuno 125M — Core Transformer

Components:
  - RMSNorm: pre-normalization (applied BEFORE attention and FFN)
  - SwiGLUFFN: gated feed-forward network
  - TreunoBlock: one decoder block (norm → attn → add) (norm → ffn → add)
  - TreunoModel: full decoder stack with tied input/output embeddings

Tied embeddings:
    self.lm_head.weight = self.embed_tokens.embed.weight
    This saves 32768 x 768 = 25.2M parameters.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import TreunoConfig
from .embedding import TreunoEmbedding
from .attention import TreunoAttention


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Simpler than LayerNorm: no mean subtraction, just RMS scaling.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability, then cast back
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_normed = (x_f32 / rms).to(x.dtype)
        return x_normed * self.weight


# ── SwiGLU Feed-Forward Network ──────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).

    FFN(x) = (W_gate_proj(x) * SiLU(W_up_proj(x))) @ W_down_proj

    Uses 3 weight matrices instead of 2 (as in standard FFN), but at
    2/3 the intermediate width to keep the parameter count equivalent.
    SwiGLU consistently outperforms ReLU and GELU on code tasks.

    For Treuno-125M:
        hidden_size = 768
        ffn_size    = 3072  (4 × hidden — the full width is already set here)
    """
    def __init__(self, config: TreunoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=config.bias)
        self.up_proj   = nn.Linear(config.hidden_size, config.ffn_size, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_size, config.hidden_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate activations with SiLU (aka Swish): gate ⊙ SiLU(up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Decoder Block ─────────────────────────────────────────────────────────────

class TreunoBlock(nn.Module):
    """
    One Treuno decoder block:

        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Pre-norm placement (norm before sublayer) is more stable than post-norm.
    """
    def __init__(self, config: TreunoConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn      = TreunoAttention(config)
        self.ffn_norm  = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn       = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # ── Self-attention with pre-norm ────────────────────────────────────
        residual = hidden_states
        hidden_states, kv_cache = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # ── Feed-forward with pre-norm ───────────────────────────────────────
        residual = hidden_states
        hidden_states = residual + self.ffn(self.ffn_norm(hidden_states))

        return hidden_states, kv_cache


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class TreunoOutput:
    logits: torch.Tensor                              # (B, S, vocab_size)
    past_key_values: Optional[List[Tuple]] = None    # one (K,V) per layer
    hidden_states: Optional[torch.Tensor] = None     # final hidden (B, S, H)


# ── Full Model ────────────────────────────────────────────────────────────────

class TreunoModel(nn.Module):
    """
    Treuno 125M: full decoder-only causal language model.

    Key property — TIED EMBEDDINGS:
        The LM head (vocab projection) shares its weight matrix with the
        input token embedding table. After construction:

            assert model.lm_head.weight is model.embed_tokens.embed.weight

        This saves 32,768 × 768 = 25.2M parameters, recovering headroom
        that allows deeper or wider layers within the 125M budget.
    """

    def __init__(self, config: TreunoConfig):
        super().__init__()
        self.config = config

        # ── Embedding table ──────────────────────────────────────────────────
        self.embed_tokens = TreunoEmbedding(config)

        # ── Transformer blocks ───────────────────────────────────────────────
        self.layers = nn.ModuleList(
            [TreunoBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # ── Final normalization ──────────────────────────────────────────────
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

        # ── LM Head ─────────────────────────────────────────────────────────
        # bias=False is mandatory to keep the tie clean
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── Tie embeddings ───────────────────────────────────────────────────
        # THIS IS THE CRITICAL LINE: share weight tensors
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.embed.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens.embed

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens.embed = embeddings
        if self.config.tie_embeddings:
            self.lm_head.weight = embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> TreunoOutput:
        """
        Args:
            input_ids:         (B, S)
            attention_mask:    (B, S) binary or None
            position_ids:      (B, S) or None — auto-generated if None
            past_key_values:   list of (K, V) per layer; for generation
            use_cache:         return updated KV cache
            output_hidden_states: return final hidden states
        Returns:
            TreunoOutput with .logits, .past_key_values, .hidden_states
        """
        B, S = input_ids.shape
        device = input_ids.device

        # ── Position IDs ─────────────────────────────────────────────────────
        if position_ids is None:
            offset = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(offset, offset + S, device=device).unsqueeze(0)

        # ── Attention mask → additive ─────────────────────────────────────────
        # Convert (B, S) → (B, 1, 1, S) additive mask for broadcasting
        if attention_mask is not None and attention_mask.dim() == 2:
            # 0 → 0.0 (attend), 1 → -inf (mask)
            attention_mask = (1.0 - attention_mask.float()) * torch.finfo(torch.float32).min
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # ── Forward pass ──────────────────────────────────────────────────────
        hidden_states = self.embed_tokens(input_ids)   # (B, S, H)

        new_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values else None
            hidden_states, kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            if use_cache and kv is not None:
                new_key_values.append(kv)

        hidden_states = self.norm(hidden_states)       # (B, S, H)
        logits = self.lm_head(hidden_states)           # (B, S, V)

        return TreunoOutput(
            logits=logits,
            past_key_values=new_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy / nucleus sampling generation with KV-cache.

        Returns:
            generated token IDs (B, input_len + new_tokens)
        """
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Only pass new tokens after first step (KV-cache)
            if past_key_values is not None:
                cur_ids = generated[:, -1:]
            else:
                cur_ids = generated

            out = self.forward(
                cur_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]  # (B, V)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok_id in set(generated[0].tolist()):
                    logits[0, tok_id] /= repetition_penalty

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens above top_p cumulative probability
                remove = cumulative_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count trainable parameters. With tied embeddings, LM head is not double-counted."""
        params = {id(p): p for p in self.parameters()}
        total = sum(p.numel() for p in params.values())
        if non_embedding:
            total -= self.embed_tokens.embed.weight.numel()
        return total
