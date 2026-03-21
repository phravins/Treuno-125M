"""
Treuno 125M — Inference Engine
The orchestrator that brings all three systems together:

  1. Antigravity Retrieval  → enrich prompt with live web docs
  2. TreunoModel            → generate code
  3. Code Sandbox           → verify and self-correct

Usage:
    engine = TreunoEngine.from_pretrained("d:/MODEL/weights")
    result = engine.generate("Write a Python function to fetch JSON from a URL")
    print(result.text)
    print(result.execution.stdout)
"""

from __future__ import annotations
import os
import re
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from model.config import TreunoConfig
from model.transformer import TreunoModel
from model.tokenizer import TreunoTokenizer
from Modelworks.retriever import ModelRetriever
from Modelworks.rag import build_rag_prompt
from sandbox.executor import CodeExecutor
from sandbox.verifier import CodeVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    prompt: str                          # Final enriched prompt sent to model
    text: str                            # Raw model output
    code: Optional[str] = None           # Extracted code block
    language: Optional[str] = None       # Detected language
    verification: Optional[VerificationResult] = None
    retrieved_sources: List[str] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def verified(self) -> bool:
        return self.verification is not None and self.verification.passed

    @property
    def stdout(self) -> str:
        if self.verification:
            return self.verification.stdout
        return ""


class TreunoEngine:
    """
    Full Treuno inference pipeline:

        query → [Modelworks] → enriched prompt
              → [TreunoModel]  → raw output
              → [Sandbox]      → verified code
              → GenerationResult
    """

    def __init__(
        self,
        model: TreunoModel,
        tokenizer: TreunoTokenizer,
        config: TreunoConfig,
        use_retrieval: bool = True,
        use_sandbox: bool = True,
        retriever: Optional[ModelRetriever] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.use_retrieval = use_retrieval
        self.use_sandbox = use_sandbox
        self.retriever = retriever or (ModelRetriever() if use_retrieval else None)
        self.executor = CodeExecutor()
        self.verifier = CodeVerifier()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        use_retrieval: Optional[bool] = None,
        use_sandbox: Optional[bool] = None,
        verify_retries: int = 3,
    ) -> GenerationResult:
        """
        Full pipeline inference.

        Args:
            prompt:          User's coding question or instruction
            language:        Expected output language (auto-detected if None)
            max_new_tokens:  Max tokens to generate
            temperature:     Sampling temperature (lower = more deterministic)
            top_p:           Nucleus sampling threshold
            use_retrieval:   Override instance setting
            use_sandbox:     Override instance setting
            verify_retries:  Max self-correction attempts

        Returns:
            GenerationResult with text, code, verification, sources
        """
        do_retrieval = use_retrieval if use_retrieval is not None else self.use_retrieval
        do_sandbox = use_sandbox if use_sandbox is not None else self.use_sandbox

        retrieved_sources = []
        enriched_prompt = prompt

        # ── Step 1: Modelworks Retrieval ────────────────────────────────────
        if do_retrieval and self.retriever:
            try:
                logger.info("Modelworks: searching...")
                results = self.retriever.search_for_code_query(prompt)
                if results:
                    docs = [{"text": r.snippet, "url": r.url, "title": r.title}
                            for r in results]
                    enriched_prompt = build_rag_prompt(prompt, docs, prompt)
                    retrieved_sources = [r.url for r in results if r.url]
                    logger.info(f"Modelworks: injected {len(docs)} sources.")
            except Exception as e:
                logger.warning(f"Modelworks retrieval failed: {e}. Proceeding without.")

        # ── Step 2: Generate ─────────────────────────────────────────────────
        raw_output = self._run_model(enriched_prompt, max_new_tokens, temperature, top_p)

        # ── Step 3: Extract code block ───────────────────────────────────────
        detected_lang = language or self._detect_language(raw_output)
        code = CodeExecutor.extract_code_block(raw_output, detected_lang)

        # ── Step 4: Sandbox verification ─────────────────────────────────────
        verification = None
        if do_sandbox and code and detected_lang:
            def _generate_fn(correction_prompt: str) -> str:
                return self._run_model(
                    correction_prompt, max_new_tokens, temperature, top_p
                )
            verification = self.verifier.verify_with_retries(
                code=code,
                language=detected_lang,
                generate_fn=_generate_fn,
                original_prompt=enriched_prompt,
                max_retries=verify_retries,
            )
            # Update code to final (possibly corrected) code
            if verification.execution.code:
                code = verification.execution.code

        return GenerationResult(
            prompt=enriched_prompt,
            text=raw_output,
            code=code,
            language=detected_lang,
            verification=verification,
            retrieved_sources=retrieved_sources,
        )

    def _run_model(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> str:
        """Tokenize, generate, decode."""
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, max_length=self.config.context_length - max_new_tokens, truncation=True)],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids.tolist())

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        """Detect language from markdown code fence."""
        match = re.search(r"```(\w+)", text)
        if match:
            lang = match.group(1).lower()
            return lang if lang not in ("text", "plain", "output") else None
        return "python"  # Default to Python for code tasks

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        use_retrieval: bool = True,
        use_sandbox: bool = True,
        device: Optional[str] = None,
    ) -> "TreunoEngine":
        """
        Load engine from a directory or a HuggingFace hub ID.
        If model_path is a HuggingFace ID (e.g. 'gpt2'), it loads that model.
        """
        config = TreunoConfig.treuno_125m()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Step 1: Handle model loading ─────────────────────────────────────
        weights_path = f"{model_path}/model.pt"
        if os.path.isfile(weights_path):
            # Case 1: Local Treuno weights
            model = TreunoModel(config)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded Treuno weights from {weights_path}")
        elif not os.path.isdir(model_path) and "/" in model_path or model_path in ("gpt2", "openai-community/gpt2"):
            # Case 2: HuggingFace Hub ID (for Immediate Test Mode)
            from transformers import AutoModelForCausalLM
            logger.info(f"Test Mode: Loading pre-trained Hub model '{model_path}'...")
            model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            # Case 3: Empty / Random init
            logger.warning(f"No weights at {weights_path}. Using random init (dev mode).")
            model = TreunoModel(config)

        # ── Step 2: Handle tokenizer ─────────────────────────────────────────
        tokenizer_path = f"{model_path}/tokenizer" if os.path.isdir(f"{model_path}/tokenizer") else model_path
        tokenizer = TreunoTokenizer.from_pretrained(tokenizer_path, config=config)

        return cls(model=model, tokenizer=tokenizer, config=config,
                   use_retrieval=use_retrieval, use_sandbox=use_sandbox, device=device)
        tokenizer = TreunoTokenizer.from_pretrained(
            f"{model_path}/tokenizer" if __import__('os').path.isdir(f"{model_path}/tokenizer") else "gpt2",
            config=config,
        )
        return cls(model=model, tokenizer=tokenizer, config=config,
                   use_retrieval=use_retrieval, use_sandbox=use_sandbox, device=device)
