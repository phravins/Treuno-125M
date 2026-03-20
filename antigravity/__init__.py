"""
Treuno 125M — Antigravity Framework
=====================================
Five-component inference augmentation system.

  AG-Retrieve  — Hybrid FAISS IVF-PQ + BM25 + Brave Search + cross-encoder rerank
  AG-Execute   — Docker + gVisor secure code sandbox (11 language runtimes)
  AG-Verify    — Composite confidence scoring + uncertainty interception (threshold 0.75)
  AG-Cache     — Redis 7 semantic cache (cosine similarity, threshold 0.92)
  AG-Update    — Continuous learning: source refresh + weekly LoRA fine-tune

Quick start:
    from antigravity.pipeline import AntigravityPipeline
    ag = AntigravityPipeline.default()   # production (Docker + Redis + Brave)
    ag = AntigravityPipeline.dev()       # development (subprocess + no cache)
"""

from .ag_retrieve import AGRetrieve, RetrievedPassage, RetrievalResult
from .ag_execute  import AGExecute, ExecutionResult
from .ag_verify   import AGVerify, VerifiedResponse, VerificationScore
from .ag_cache    import AGCache, CacheEntry
from .ag_update   import AGUpdate, TrainingExample, UpdateJob
from .pipeline    import AntigravityPipeline, PipelineResult

# Legacy exports (from initial stub — kept for backwards compat)
from .retriever   import AntigravityRetriever, SearchResult
from .indexer     import DocumentIndexer, DocumentChunk
from .rag         import build_rag_prompt, RAGContext

__all__ = [
    # Five canonical components
    "AGRetrieve",   "RetrievedPassage", "RetrievalResult",
    "AGExecute",    "ExecutionResult",
    "AGVerify",     "VerifiedResponse",  "VerificationScore",
    "AGCache",      "CacheEntry",
    "AGUpdate",     "TrainingExample",   "UpdateJob",
    "AntigravityPipeline", "PipelineResult",
    # Legacy
    "AntigravityRetriever", "SearchResult",
    "DocumentIndexer",      "DocumentChunk",
    "build_rag_prompt",     "RAGContext",
]
