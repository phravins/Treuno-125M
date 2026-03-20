"""
Treuno 125M — Tokenizer
BPE tokenizer wrapper with FIM (Fill-in-the-Middle) support.
Falls back to GPT-2 tokenizer for out-of-the-box usability;
replace with a code-trained BPE tokenizer for production.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Union

from .config import TreunoConfig


# FIM sentinel strings
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"


class TreunoTokenizer:
    """
    Thin wrapper around a HuggingFace tokenizer that adds FIM utilities.

    Usage:
        tok = TreunoTokenizer.from_pretrained("gpt2")        # dev / testing
        tok = TreunoTokenizer.from_pretrained("d:/MODEL/tokenizer")  # production

    FIM encoding:
        ids = tok.encode_fim(prefix="def foo(", suffix="):\\n    pass", middle="x")
    """

    def __init__(self, hf_tokenizer, config: Optional[TreunoConfig] = None):
        self._tok = hf_tokenizer
        self.config = config or TreunoConfig.treuno_125m()

        # Ensure FIM tokens are registered
        special = {
            "additional_special_tokens": [FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX]
        }
        self._tok.add_special_tokens(special)

        # Cache FIM token IDs
        self.fim_prefix_id = self._tok.convert_tokens_to_ids(FIM_PREFIX)
        self.fim_middle_id = self._tok.convert_tokens_to_ids(FIM_MIDDLE)
        self.fim_suffix_id = self._tok.convert_tokens_to_ids(FIM_SUFFIX)

    # ── Core encode / decode ─────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        kwargs: dict = dict(add_special_tokens=add_special_tokens)
        if max_length:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = truncation
        return self._tok.encode(text, **kwargs)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ):
        ml = max_length or self.config.context_length
        return self._tok(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=ml,
            return_tensors=return_tensors,
        )

    # ── FIM utilities ────────────────────────────────────────────────────────

    def encode_fim(
        self,
        prefix: str,
        suffix: str,
        middle: str = "",
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode a Fill-in-the-Middle (FIM) sequence.

        Format:
            <fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}<EOS>

        Args:
            prefix:  Code before the hole
            suffix:  Code after the hole
            middle:  Ground truth that fills the hole (empty at inference time)
            add_eos: Whether to append EOS token

        Returns:
            List of token IDs
        """
        ids = (
            [self.fim_prefix_id]
            + self._tok.encode(prefix, add_special_tokens=False)
            + [self.fim_suffix_id]
            + self._tok.encode(suffix, add_special_tokens=False)
            + [self.fim_middle_id]
        )
        if middle:
            ids += self._tok.encode(middle, add_special_tokens=False)
        if add_eos and self._tok.eos_token_id is not None:
            ids.append(self._tok.eos_token_id)
        return ids

    def decode_fim(self, token_ids: List[int]) -> dict:
        """
        Decode a FIM sequence back into its constituent parts.

        Returns:
            dict with keys: prefix, suffix, middle, raw
        """
        raw = self._tok.decode(token_ids, skip_special_tokens=False)
        result = {"prefix": "", "suffix": "", "middle": "", "raw": raw}
        try:
            if FIM_PREFIX in raw and FIM_SUFFIX in raw and FIM_MIDDLE in raw:
                after_prefix = raw.split(FIM_PREFIX, 1)[1]
                prefix_part, rest = after_prefix.split(FIM_SUFFIX, 1)
                suffix_part, middle_part = rest.split(FIM_MIDDLE, 1)
                result["prefix"] = prefix_part.strip()
                result["suffix"] = suffix_part.strip()
                result["middle"] = middle_part.strip()
        except ValueError:
            pass
        return result

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._tok.pad_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tok.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tok.bos_token_id

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to directory."""
        self._tok.save_pretrained(str(path))

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        config: Optional[TreunoConfig] = None,
    ) -> "TreunoTokenizer":
        """
        Load a tokenizer from HuggingFace hub or local directory.

        For development: use "gpt2" as a placeholder.
        For production: use a code-trained BPE tokenizer with vocab_size=32768.
        """
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(name_or_path)
        # Ensure pad token exists
        if hf_tok.pad_token is None:
            hf_tok.pad_token = hf_tok.eos_token
        return cls(hf_tok, config)

    def __repr__(self) -> str:
        return (
            f"TreunoTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"fim_prefix_id={self.fim_prefix_id}, "
            f"fim_middle_id={self.fim_middle_id}, "
            f"fim_suffix_id={self.fim_suffix_id})"
        )
