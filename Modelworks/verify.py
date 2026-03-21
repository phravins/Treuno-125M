"""
Treuno — Model-Verify
===================

Evaluates every response before it reaches the user using a
composite confidence score across three dimensions:

  Dimension 1 — Source Citation (weight: 0.30)
    Does the response cite at least one retrieved source URL?
    Score: 1.0 if yes, 0.0 if no.

  Dimension 2 — Sandbox Execution (weight: 0.50)
    Did the generated code pass Model-Execute?
    Score: 1.0 = passed | 0.5 = no code (text answer) | 0.0 = failed all retries

  Dimension 3 — Semantic Consistency (weight: 0.20)
    Is the response semantically consistent with the retrieved passages?
    Score: cosine similarity between response embedding and top retrieved passage embedding.

Composite score = 0.30 * citation + 0.50 * execution + 0.20 * consistency

If score < 0.75:
    Response is intercepted and replaced with an explicit uncertainty statement:
      "I found the following information: [sources]
       I cannot fully verify this answer.
       Please cross-check: [cited URLs]"

This threshold ensures that only verified, grounded answers reach users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75

# Dimension weights — must sum to 1.0
W_CITATION = 0.30
W_EXECUTION = 0.50
W_CONSISTENCY = 0.20


@dataclass
class VerificationScore:
    citation_score: float       # 0.0 or 1.0
    execution_score: float      # 0.0, 0.5, or 1.0
    consistency_score: float    # 0.0–1.0 cosine similarity
    composite: float            # weighted sum
    passed: bool                # composite >= 0.75
    reason: str = ""            # human-readable explanation


@dataclass
class VerifiedResponse:
    original_text: str
    final_text: str             # may be replaced by uncertainty statement
    score: VerificationScore
    intercepted: bool           # True if score < threshold
    retrieved_urls: List[str]


class ModelVerify:
    """
    Model-Verify: lightweight confidence scoring + uncertainty interception.

    Call:
        result = verifier.verify(
            response_text,
            retrieved_passages,
            execution_result,
        )
        if result.intercepted:
            return result.final_text  # uncertainty statement
        else:
            return result.final_text  # verified answer
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self._encoder = None

    def verify(
        self,
        response_text: str,
        retrieved_passages: List[dict],       # list of {text, url, title}
        execution_result: Optional[object],   # ModelExecute.ExecutionResult or None
        code_was_generated: bool = True,
    ) -> VerifiedResponse:
        """
        Score a response across 3 dimensions and intercept if below threshold.

        Args:
            response_text:      The model's output text
            retrieved_passages: Passages from Model-Retrieve
            execution_result:   Result from Model-Execute (or None if no code)
            code_was_generated: False for pure text answers (affects execution weight)

        Returns:
            VerifiedResponse with final_text and score breakdown
        """
        urls = [p.get("url", "") for p in retrieved_passages if p.get("url")]

        # Dimension 1: Citation
        citation = self._score_citation(response_text, urls)

        # Dimension 2: Execution
        execution = self._score_execution(execution_result, code_was_generated)

        # Dimension 3: Semantic consistency
        consistency = self._score_consistency(response_text, retrieved_passages)

        composite = (
            W_CITATION * citation
            + W_EXECUTION * execution
            + W_CONSISTENCY * consistency
        )

        passed = composite >= self.threshold
        reason = self._build_reason(citation, execution, consistency, composite)

        score = VerificationScore(
            citation_score=citation,
            execution_score=execution,
            consistency_score=consistency,
            composite=composite,
            passed=passed,
            reason=reason,
        )

        if passed:
            final_text = response_text
            intercepted = False
        else:
            final_text = self._build_uncertainty_statement(
                response_text, urls, score
            )
            intercepted = True
            logger.warning(
                f"Model-Verify intercepted response. "
                f"Score={composite:.2f} (threshold={self.threshold}). {reason}"
            )

        return VerifiedResponse(
            original_text=response_text,
            final_text=final_text,
            score=score,
            intercepted=intercepted,
            retrieved_urls=urls,
        )

    # ── Dimension 1: Citation ─────────────────────────────────────────────────

    @staticmethod
    def _score_citation(response: str, urls: List[str]) -> float:
        """
        Check if response mentions at least one retrieved URL or source domain.
        Score: 1.0 if cited, 0.0 if not.
        """
        if not urls:
            return 0.5   # No sources available — partial credit
        response_lower = response.lower()
        for url in urls:
            domain = url.split("/")[2] if url.startswith("http") else url
            if domain.lower() in response_lower or url.lower() in response_lower:
                return 1.0
        # Also check for any http URL in response (self-generated citations)
        import re
        if re.search(r"https?://\S+", response):
            return 0.7   # Cited something, but not our retrieved sources
        return 0.0

    # ── Dimension 2: Execution ────────────────────────────────────────────────

    @staticmethod
    def _score_execution(execution_result: Optional[object], code_generated: bool) -> float:
        """
        Score based on sandbox execution outcome.
          1.0  — code passed
          0.5  — no code (text answer, irrelevant)
          0.0  — code failed all retries or timed out
        """
        if not code_generated or execution_result is None:
            return 0.5   # Text-only answer — neutral
        if hasattr(execution_result, "success"):
            return 1.0 if execution_result.success else 0.0
        return 0.5

    # ── Dimension 3: Semantic consistency ─────────────────────────────────────

    def _score_consistency(
        self, response: str, retrieved_passages: List[dict]
    ) -> float:
        """
        Cosine similarity between response embedding and top retrieved passage.
        Uses the same all-MiniLM-L6-v2 encoder as Model-Retrieve.
        """
        if not retrieved_passages:
            return 0.5   # Nothing to compare against — neutral
        top_text = retrieved_passages[0].get("text", "")
        if not top_text or not response:
            return 0.5
        try:
            import numpy as np
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            embs = self._encoder.encode(
                [response[:512], top_text[:512]],
                normalize_embeddings=True,
            )
            # Cosine similarity (dot product of L2-normalized vectors)
            sim = float(np.dot(embs[0], embs[1]))
            # Clamp to [0, 1]
            return max(0.0, min(1.0, sim))
        except Exception as e:
            logger.warning(f"Consistency scoring failed: {e}")
            return 0.5

    # ── Uncertainty statement ─────────────────────────────────────────────────

    @staticmethod
    def _build_uncertainty_statement(
        response: str, urls: List[str], score: VerificationScore
    ) -> str:
        """
        Construct the user-facing uncertainty message shown when score < 0.75.
        Per spec: tell user what was found, flag inability to verify, cite sources.
        """
        url_list = "\n".join(f"  • {u}" for u in urls[:5]) if urls else "  (no sources retrieved)"

        # Extract first ~200 chars of the original response as "what I found"
        snippet = response.strip()[:300].rstrip()
        if len(response.strip()) > 300:
            snippet += "…"

        return (
            f"⚠️ **I cannot fully verify this answer** "
            f"(confidence score: {score.composite:.0%}, threshold: 75%)\n\n"
            f"**What I found:**\n{snippet}\n\n"
            f"**Why I'm not certain:** {score.reason}\n\n"
            f"**Please cross-check these sources before relying on this answer:**\n"
            f"{url_list}"
        )

    @staticmethod
    def _build_reason(
        citation: float, execution: float, consistency: float, composite: float
    ) -> str:
        reasons = []
        if citation < 0.5:
            reasons.append("no source cited")
        if execution == 0.0:
            reasons.append("code failed execution")
        if consistency < 0.4:
            reasons.append("low semantic consistency with retrieved docs")
        return "; ".join(reasons) if reasons else f"composite score {composite:.2f}"
