"""
Treuno 125M — Model Utilities
Weight initialization helpers and parameter analysis tools.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .config import TreunoConfig


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total parameter count, deduplicating tied weights."""
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p.data)
        if pid in seen:
            continue
        seen.add(pid)
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def parameter_groups(model: nn.Module) -> Dict[str, int]:
    """
    Break down parameter count by component.

    Returns a dict like:
        {
          "embed_tokens": 25165824,
          "layers.0.attn": 1966080,
          ...
          "norm": 768,
          "lm_head": 0,     # 0 because tied to embed_tokens
        }
    """
    groups: Dict[str, int] = {}
    seen_data_ptrs = set()

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_data_ptrs:
            top = name.split(".")[0]
            groups.setdefault(f"{top} (tied)", 0)
            continue
        seen_data_ptrs.add(ptr)
        top = ".".join(name.split(".")[:2])
        groups[top] = groups.get(top, 0) + param.numel()

    return groups


def print_model_summary(model: nn.Module, config: TreunoConfig) -> None:
    """Print a human-readable model summary."""
    total = count_parameters(model)
    groups = parameter_groups(model)

    print(f"\n{'='*55}")
    print(f"  {config.model_name.upper()}  v{config.model_version}")
    print(f"{'='*55}")
    print(f"  {'Architecture':<30} Decoder-only transformer")
    print(f"  {'Layers':<30} {config.num_layers}")
    print(f"  {'Hidden dim':<30} {config.hidden_size}")
    print(f"  {'FFN dim (SwiGLU)':<30} {config.ffn_size}")
    print(f"  {'Q heads / KV heads':<30} {config.num_q_heads} / {config.num_kv_heads}")
    print(f"  {'Context length':<30} {config.context_length:,} tokens")
    print(f"  {'Vocabulary':<30} {config.vocab_size:,} tokens")
    print(f"  {'Tied embeddings':<30} {'Yes' if config.tie_embeddings else 'No'}")
    print(f"  {'dtype':<30} {config.torch_dtype}")
    print(f"{'─'*55}")
    print(f"  {'Component':<38} {'Params':>12}")
    print(f"{'─'*55}")
    for name, count in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {name:<38} {count:>12,}")
    print(f"{'─'*55}")
    print(f"  {'TOTAL (unique params)':<38} {total:>12,}")
    print(f"{'='*55}\n")


def verify_tied_embeddings(model: nn.Module) -> bool:
    """Assert that lm_head.weight is embed_tokens.embed.weight (same tensor)."""
    embed_w = model.embed_tokens.embed.weight
    lm_w = model.lm_head.weight
    tied = embed_w is lm_w
    if tied:
        print("✅ Tied embeddings verified: lm_head.weight IS embed_tokens.embed.weight")
    else:
        print("❌ Tied embeddings BROKEN: lm_head.weight is a different tensor")
    return tied


def cast_to_bfloat16(model: nn.Module) -> nn.Module:
    """Cast all model parameters to bfloat16 (training/serving dtype)."""
    return model.to(torch.bfloat16)


def get_rope_scaling_info(config: TreunoConfig) -> Tuple[int, float]:
    """Return (head_dim, rope_theta) for RoPE configuration."""
    return config.head_dim, config.rope_theta
