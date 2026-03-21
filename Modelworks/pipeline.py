"""
Treuno — Pipeline Orchestrator
===========================================
Full inference call path:
  1. AG-Cache.lookup(query) → if hit (sim >= 0.92) → return cached response
  2. AG-Retrieve(query)     → hybrid search → rerank → top-3 passages
  3. Inject passages into prompt → TreunoModel.generate()
  4. AG-Execute(code)       → Docker+gVisor sandbox → execution result
  5. AG-Verify(response)    → confidence score → intercept if < 0.75
  6. AG-Cache.store(query, verified_response)
  7. AG-Update.collect_training_example(if verified)
  8. Return final VerifiedResponse to user

Usage:
    pipeline = AntigravityPipeline.default()
    result = pipeline.run(
        query="Write a Python function to fetch JSON from a URL",
        model_generate_fn=treuno_engine.generate,
    )
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .ag_retrieve import AGRetrieve, RetrievedPassage, RetrievalResult
from .ag_execute import AGExecute, ExecutionResult
from .ag_verify import AGVerify, VerifiedResponse
from .ag_cache import AGCache, CacheEntry
from .ag_update import AGUpdate, TrainingExample

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    final_text: str                       # What the user sees
    verified: bool
    confidence_score: float
    intercepted: bool                     # True = uncertainty statement shown
    from_cache: bool
    retrieved_passages: List[RetrievedPassage] = field(default_factory=list)
    execution: Optional[ExecutionResult] = None
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    cache_entry: Optional[CacheEntry] = None


class AntigravityPipeline:
    """
    The complete Antigravity inference framework for Treuno.

    Instantiate once at server startup, reuse across requests.
    All components support independent enable/disable flags for dev mode.
    """

    def __init__(
        self,
        retriever: Optional[AGRetrieve] = None,
        executor: Optional[AGExecute] = None,
        verifier: Optional[AGVerify] = None,
        cache: Optional[AGCache] = None,
        updater: Optional[AGUpdate] = None,
        use_retrieval: bool = True,
        use_sandbox: bool = True,
        use_verify: bool = True,
        use_cache: bool = True,
        use_update: bool = True,
        max_retries: int = 3,
    ):
        self.retriever = retriever
        self.executor = executor
        self.verifier = verifier
        self.cache = cache
        self.updater = updater
        self.use_retrieval = use_retrieval
        self.use_sandbox = use_sandbox
        self.use_verify = use_verify
        self.use_cache = use_cache
        self.use_update = use_update
        self.max_retries = max_retries

    @classmethod
    def default(
        cls,
        brave_api_key: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        use_gvisor: bool = True,
    ) -> "AntigravityPipeline":
        """
        Create a fully-configured Antigravity pipeline with all 5 components.
        """
        cache = AGCache(redis_url=redis_url)
        retriever = AGRetrieve(
            brave_api_key=brave_api_key,
            cache=cache,
        )
        executor = AGExecute(use_gvisor=use_gvisor, fallback_to_subprocess=True)
        verifier = AGVerify()
        updater = AGUpdate(retriever=retriever)
        return cls(
            retriever=retriever,
            executor=executor,
            verifier=verifier,
            cache=cache,
            updater=updater,
        )

    @classmethod
    def dev(cls) -> "AntigravityPipeline":
        """
        Minimal pipeline for local development — no Redis, no Docker, no Brave.
        Uses subprocess sandbox and DuckDuckGo fallback.
        """
        return cls(
            retriever=AGRetrieve(),
            executor=AGExecute(use_gvisor=False, fallback_to_subprocess=True),
            verifier=AGVerify(),
            cache=None,
            updater=None,
            use_cache=False,
            use_update=False,
        )

    def run(
        self,
        query: str,
        model_generate_fn: Callable[[str], str],
        detected_language: Optional[str] = "python",
    ) -> PipelineResult:
        """
        Run the full Antigravity inference pipeline.

        Args:
            query:              User's raw query string
            model_generate_fn:  Callable(enriched_prompt: str) → raw model output
            detected_language:  Expected output language (for sandbox)

        Returns:
            PipelineResult with final_text, verification, latency stats
        """
        t_total = time.perf_counter()

        # ── Step 1: AG-Cache lookup ───────────────────────────────────────────
        if self.use_cache and self.cache:
            hit = self.cache.lookup(query)
            if hit:
                return PipelineResult(
                    query=query,
                    final_text=hit.response_text,
                    verified=True,
                    confidence_score=hit.score_composite,
                    intercepted=False,
                    from_cache=True,
                    cache_entry=hit,
                    total_latency_ms=(time.perf_counter() - t_total) * 1000,
                )

        # ── Step 2: AG-Retrieve ───────────────────────────────────────────────
        retrieved: List[RetrievedPassage] = []
        retrieval_latency = 0.0
        if self.use_retrieval and self.retriever:
            retrieval_result: RetrievalResult = self.retriever.retrieve(query)
            retrieved = retrieval_result.passages
            retrieval_latency = retrieval_result.latency_ms
            logger.info(
                f"AG-Retrieve: {len(retrieved)} passages in {retrieval_latency:.0f}ms"
            )

        # ── Step 3: Build enriched prompt ─────────────────────────────────────
        from .rag import build_rag_prompt
        docs = [{"text": p.text, "url": p.url, "title": p.title} for p in retrieved]
        enriched_prompt = build_rag_prompt(query, docs, query, max_context_chars=2000)

        # ── Step 4: Model generation + AG-Execute self-correction loop ─────────
        final_code = None
        exec_result: Optional[ExecutionResult] = None
        raw_output = ""

        for attempt in range(1, self.max_retries + 1):
            raw_output = model_generate_fn(enriched_prompt)

            if not self.use_sandbox or not self.executor:
                break

            # Extract code blocks from model output
            code_blocks = self.executor.extract_code_blocks(raw_output)
            if not code_blocks:
                break   # Text-only answer — no code to verify

            lang, code = code_blocks[0]
            exec_result = self.executor.run(code, detected_language or lang)

            if exec_result.success:
                final_code = code
                logger.info(f"AG-Execute: code passed on attempt {attempt}.")
                break
            else:
                if attempt < self.max_retries:
                    # Feed error back into the prompt for correction
                    enriched_prompt += f"\n\n{exec_result.feedback_context()}"
                    logger.info(
                        f"AG-Execute: attempt {attempt} failed. "
                        f"Error: {exec_result.short_error[:80]}. Retrying..."
                    )
                else:
                    logger.warning(
                        f"AG-Execute: code failed after {self.max_retries} attempts. "
                        f"Response will be intercepted by AG-Verify."
                    )

        # ── Step 5: AG-Verify ─────────────────────────────────────────────────
        verified_resp: Optional[VerifiedResponse] = None
        code_was_generated = exec_result is not None

        if self.use_verify and self.verifier:
            verified_resp = self.verifier.verify(
                response_text=raw_output,
                retrieved_passages=docs,
                execution_result=exec_result,
                code_was_generated=code_was_generated,
            )
            final_text = verified_resp.final_text
            confidence = verified_resp.score.composite
            intercepted = verified_resp.intercepted
            verified_ok = verified_resp.score.passed
        else:
            final_text = raw_output
            confidence = 1.0
            intercepted = False
            verified_ok = True

        # ── Step 6: AG-Cache store ────────────────────────────────────────────
        if self.use_cache and self.cache and verified_ok and not intercepted:
            source_tier = retrieved[0].source_tier if retrieved else "web"
            self.cache.store(
                query=query,
                response_text=final_text,
                retrieved_urls=[p.url for p in retrieved],
                score_composite=confidence,
                source_tier=source_tier,
            )

        # ── Step 7: AG-Update (collect training example) ──────────────────────
        if self.use_update and self.updater and verified_ok and not intercepted:
            example = TrainingExample(
                prompt=query,
                response=raw_output,
                language=detected_language or "python",
                sources=[p.url for p in retrieved],
                confidence_score=confidence,
                timestamp=time.time(),
            )
            self.updater.collect_training_example(example)

        total_latency = (time.perf_counter() - t_total) * 1000
        logger.info(
            f"Antigravity pipeline complete: "
            f"verified={verified_ok}, intercepted={intercepted}, "
            f"confidence={confidence:.2f}, latency={total_latency:.0f}ms"
        )

        return PipelineResult(
            query=query,
            final_text=final_text,
            verified=verified_ok,
            confidence_score=confidence,
            intercepted=intercepted,
            from_cache=False,
            retrieved_passages=retrieved,
            execution=exec_result,
            retrieval_latency_ms=retrieval_latency,
            total_latency_ms=total_latency,
        )

    def status(self) -> dict:
        """Return health/status of all components."""
        return {
            "retrieval": self.use_retrieval and self.retriever is not None,
            "sandbox": self.use_sandbox and self.executor is not None,
            "verify": self.use_verify and self.verifier is not None,
            "cache": self.cache.stats() if self.cache else {"status": "disabled"},
            "update": repr(self.updater) if self.updater else "disabled",
        }
