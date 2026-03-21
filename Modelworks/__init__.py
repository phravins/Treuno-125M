"""
Treuno 125M — Modelworks Framework
=====================================
Five-component inference augmentation system.

  Model-Retrieve  — Hybrid FAISS IVF-PQ + BM25 + Brave Search + cross-encoder rerank
  Model-Execute   — Docker + gVisor secure code sandbox (11 language runtimes)
  Model-Verify    — Composite confidence scoring + uncertainty interception (threshold 0.75)
  Model-Cache     — Redis 7 semantic cache (cosine similarity, threshold 0.92)
  Model-Update    — Continuous learning: source refresh + weekly LoRA fine-tune

Quick start:
    from Modelworks.pipeline import ModelPipeline
    mw = ModelPipeline.default()   # production (Docker + Redis + Brave)
    mw = ModelPipeline.dev()       # development (subprocess + no cache)
"""

from .retrieve import ModelRetrieve, RetrievedPassage, RetrievalResult
from .execute  import ModelExecute, ExecutionResult
from .verify   import ModelVerify, VerifiedResponse, VerificationScore
from .cache    import ModelCache, CacheEntry
from .update   import ModelUpdate, TrainingExample, UpdateJob
from .pipeline import ModelPipeline, PipelineResult

# Legacy exports (from initial stub — kept for backwards compat)
from .retriever   import ModelRetriever, SearchResult
from .indexer     import DocumentIndexer, DocumentChunk
from .rag         import build_rag_prompt, RAGContext

__all__ = [
    # Five canonical components
    "ModelRetrieve",   "RetrievedPassage", "RetrievalResult",
    "ModelExecute",    "ExecutionResult",
    "ModelVerify",     "VerifiedResponse",  "VerificationScore",
    "ModelCache",      "CacheEntry",
    "ModelUpdate",     "TrainingExample",   "UpdateJob",
    "ModelPipeline",   "PipelineResult",
    # Legacy
    "ModelRetriever",  "SearchResult",
    "DocumentIndexer", "DocumentChunk",
    "build_rag_prompt", "RAGContext",
]
