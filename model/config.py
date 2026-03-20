"""
Treuno 125M — Model Configuration
All hyperparameters in a single dataclass. This is the canonical source of truth.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreunoConfig:
    # ── Architecture ────────────────────────────────────────────────────────
    num_layers: int = 12
    hidden_size: int = 768
    ffn_size: int = 3072           # SwiGLU intermediate dim (4 × hidden)
    num_q_heads: int = 12          # Query attention heads
    num_kv_heads: int = 4          # Key-Value heads (GQA: 3:1 ratio)
    head_dim: int = 64             # hidden_size / num_q_heads
    context_length: int = 8192     # Maximum sequence length

    # ── Vocabulary & Tokenizer ───────────────────────────────────────────────
    vocab_size: int = 32768        # BPE vocabulary, power-of-2 for efficiency
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Fill-in-the-Middle (FIM) special tokens
    fim_prefix_token_id: int = 32765
    fim_middle_token_id: int = 32766
    fim_suffix_token_id: int = 32767

    # ── Positional Encoding ──────────────────────────────────────────────────
    rope_theta: float = 10000.0    # RoPE base frequency

    # ── Normalization ────────────────────────────────────────────────────────
    norm_eps: float = 1e-5         # RMSNorm epsilon

    # ── Regularization ──────────────────────────────────────────────────────
    dropout: float = 0.0           # No dropout (modern LLM standard)
    bias: bool = False             # No bias in linear layers

    # ── Tied Embeddings ─────────────────────────────────────────────────────
    # Input embedding weight is shared with LM head output projection.
    # This recovers ~24M parameters (32768 × 768 = 25.2M) worth of headroom.
    tie_embeddings: bool = True

    # ── Precision ───────────────────────────────────────────────────────────
    # Model trains and serves in bfloat16. Quantized to GPTQ int4 for on-disk.
    torch_dtype: str = "bfloat16"

    # ── Training ─────────────────────────────────────────────────────────────
    initializer_range: float = 0.02

    # ── Inference ───────────────────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.2       # Low temp for deterministic code generation
    top_p: float = 0.95
    repetition_penalty: float = 1.1

    # ── Model identity ──────────────────────────────────────────────────────
    model_name: str = "treuno-125m"
    model_version: str = "0.1.0"

    def __post_init__(self):
        assert self.hidden_size % self.num_q_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_q_heads {self.num_q_heads}"
        )
        assert self.num_q_heads % self.num_kv_heads == 0, (
            f"num_q_heads {self.num_q_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
        )
        computed_head_dim = self.hidden_size // self.num_q_heads
        if self.head_dim != computed_head_dim:
            self.head_dim = computed_head_dim

    @classmethod
    def treuno_125m(cls) -> "TreunoConfig":
        """
        Canonical 125M configuration.
        Decoder-only, 12L / 768H / 12Qh / 4KVh / 8192ctx / 32768vocab.
        Tied embeddings, bfloat16.
        """
        return cls(
            num_layers=12,
            hidden_size=768,
            ffn_size=3072,
            num_q_heads=12,
            num_kv_heads=4,
            context_length=8192,
            vocab_size=32768,
            rope_theta=10000.0,
            norm_eps=1e-5,
            dropout=0.0,
            bias=False,
            tie_embeddings=True,
            torch_dtype="bfloat16",
        )

    def param_estimate(self) -> int:
        """
        Rough parameter count estimate:
          - Embedding: vocab_size × hidden (shared with LM head, counted once)
          - Attention: 12L × (Q + K + V + O projections)
          - FFN: 12L × SwiGLU (3 weight matrices at 2/3 width)
          - Norms: 12L × 2 norms + 1 final, each of size hidden
        """
        embed = self.vocab_size * self.hidden_size
        # Q: H×H, K: H×(kv_heads×head_dim), V: same, O: H×H
        kv_dim = self.num_kv_heads * self.head_dim
        attn_per_layer = (
            self.hidden_size * self.hidden_size +   # Q
            self.hidden_size * kv_dim +             # K
            self.hidden_size * kv_dim +             # V
            self.hidden_size * self.hidden_size     # O
        )
        # SwiGLU: gate_proj + up_proj (both H→F) + down_proj (F→H)
        ffn_per_layer = 2 * self.hidden_size * self.ffn_size + self.ffn_size * self.hidden_size
        norms = (2 * self.num_layers + 1) * self.hidden_size
        total = embed + self.num_layers * (attn_per_layer + ffn_per_layer) + norms
        return total
