"""
Treuno — AG-Retrieve
=====================

Pipeline per query:
  1. Embed query → 384-dim vector (all-MiniLM-L6-v2)
  2. FAISS IVF-PQ vector search  ┐ run in parallel
     BM25 keyword search          ┘
  3. Merge & deduplicate results (Reciprocal Rank Fusion)
  4. Cross-encoder rerank top-10 → keep top-3
  5. Return top-3 passages for context injection

Source tiers (searched in priority order):
  GitHub repos    (filtered: stars > 100, CI passing, updated < 1yr)
  Stack Overflow  (accepted answers, weighted by vote score)
  MDN Web Docs
  PyPI / npm package docs
  Official lang refs: python.org, go.dev, docs.rust-lang.org, kotlinlang.org
  arXiv CS section
  Open web (Brave Search API)

Cache TTLs:
  Open web:      24 hours
  Docs/SO/MDN:    7 days
  GitHub data:    1 hour

Performance target: p95 retrieval < 300ms
"""

from __future__ import annotations

import re
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── Source tier definitions ───────────────────────────────────────────────────
SOURCE_TIERS: Dict[str, Dict[str, Any]] = {
    "github": {
        "cache_ttl": 3600,           # 1 hour
        "priority": 1,
        "filter_stars": 100,
    },
    "stackoverflow": {
        "cache_ttl": 604800,         # 7 days
        "priority": 2,
        "accepted_only": True,
    },
    "mdn": {
        "cache_ttl": 604800,
        "priority": 3,
        "base_url": "https://developer.mozilla.org",
    },
    "pypi_npm": {
        "cache_ttl": 604800,
        "priority": 4,
    },
    "lang_refs": {
        "cache_ttl": 604800,
        "priority": 5,
        "domains": [
            "python.org", "go.dev", "docs.rust-lang.org",
            "kotlinlang.org", "doc.rust-lang.org",
        ],
    },
    "arxiv": {
        "cache_ttl": 604800,
        "priority": 6,
        "section": "cs",
    },
    "web": {
        "cache_ttl": 86400,          # 24 hours
        "priority": 7,
    },
}


@dataclass
class RetrievedPassage:
    text: str
    url: str
    title: str = ""
    source_tier: str = "web"
    vector_score: float = 0.0      # FAISS cosine similarity
    bm25_score: float = 0.0        # BM25 keyword match score
    rerank_score: float = 0.0      # Cross-encoder rerank score
    from_cache: bool = False

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score > 0 else self.vector_score


@dataclass
class RetrievalResult:
    query: str
    passages: List[RetrievedPassage]         # Top-3 after reranking
    raw_candidates: int = 0                  # How many were retrieved before rerank
    latency_ms: float = 0.0
    cache_hit: bool = False


class AGRetrieve:
    """
    Antigravity AG-Retrieve: hybrid semantic + keyword retrieval with reranking.

    Embed → (FAISS IVF-PQ ∥ BM25) → RRF merge → cross-encoder rerank → top-3
    """

    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_retrieve: int = 10,
        top_k_final: int = 3,
        cache: Optional["AGCache"] = None,
    ):
        import os
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.faiss_index_path = faiss_index_path
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.cache = cache

        # Lazy-loaded heavy models
        self._encoder = None
        self._reranker = None
        self._faiss_index = None
        self._faiss_chunks: List[Dict] = []
        self._bm25 = None
        self._bm25_corpus: List[Dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Full retrieval pipeline: embed → hybrid search → rerank → return top-3.
        p95 target: < 300ms
        """
        t0 = time.perf_counter()

        # Step 1: Embed query (384-dim)
        query_emb = self._embed([query])[0]

        # Step 2: Parallel FAISS + BM25 + web search
        candidates = self._hybrid_search(query, query_emb)
        raw_count = len(candidates)

        if not candidates:
            return RetrievalResult(query=query, passages=[], latency_ms=(time.perf_counter()-t0)*1000)

        # Step 3: Rerank top-10 with cross-encoder
        top_candidates = candidates[:self.top_k_retrieve]
        reranked = self._rerank(query, top_candidates)

        # Step 4: Return top-3
        final = reranked[:self.top_k_final]
        latency = (time.perf_counter() - t0) * 1000
        if latency > 300:
            logger.warning(f"AG-Retrieve p95 miss: {latency:.0f}ms for '{query[:40]}'")

        return RetrievalResult(
            query=query,
            passages=final,
            raw_candidates=raw_count,
            latency_ms=latency,
        )

    # ── Step 1: Embedding ─────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> "np.ndarray":
        """Embed texts using all-MiniLM-L6-v2 → 384-dim L2-normalized vectors."""
        import numpy as np
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embed_model_name)
            logger.info(f"Loaded encoder: {self.embed_model_name}")
        vecs = self._encoder.encode(texts, normalize_embeddings=True, batch_size=32)
        return np.array(vecs, dtype=np.float32)

    # ── Step 2a: FAISS IVF-PQ search ──────────────────────────────────────────

    def build_faiss_ivfpq(self, texts: List[str], urls: List[str], titles: List[str]) -> None:
        """
        Build a FAISS IVF-PQ index from a document corpus.
        IVF-PQ compresses each 384-dim vector into quantized codes,
        reducing memory ~8x vs. flat IndexFlatIP while keeping recall > 95%.

        nlist=256 (number of Voronoi cells), m=48 (subspaces), nbits=8 (bits/subspace).
        """
        import numpy as np
        import faiss
        logger.info(f"Building FAISS IVF-PQ index over {len(texts)} documents...")
        embs = self._embed(texts)
        d = embs.shape[1]   # 384
        nlist = min(256, max(4, len(texts) // 40))
        m = 48              # must divide d (384 / 48 = 8, valid)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        index.train(embs)
        index.add(embs)
        index.nprobe = 32   # cells to search; higher = more recall, more latency
        self._faiss_index = index
        self._faiss_chunks = [
            {"text": t, "url": u, "title": ti}
            for t, u, ti in zip(texts, urls, titles)
        ]
        if self.faiss_index_path:
            faiss.write_index(index, self.faiss_index_path)
        logger.info(f"FAISS IVF-PQ index built: {index.ntotal} vectors, d={d}")

    def _faiss_search(self, query_emb, k: int) -> List[Tuple[float, Dict]]:
        """Vector similarity search. Returns (score, chunk_dict) pairs."""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        import numpy as np
        q = np.array([query_emb], dtype=np.float32)
        k = min(k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._faiss_chunks):
                results.append((float(score), self._faiss_chunks[idx]))
        return results

    # ── Step 2b: BM25 keyword search ──────────────────────────────────────────

    def build_bm25(self, texts: List[str], metadata: List[Dict]) -> None:
        """Build a BM25 index over the same corpus."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize_for_bm25(t) for t in texts]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_corpus = metadata
            logger.info(f"BM25 index built over {len(texts)} documents.")
        except ImportError:
            logger.warning("rank-bm25 not installed. BM25 search disabled.")

    def _bm25_search(self, query: str, k: int) -> List[Tuple[float, Dict]]:
        """Keyword search. Returns (score, metadata_dict) pairs."""
        if self._bm25 is None:
            return []
        tokens = self._tokenize_for_bm25(query)
        scores = self._bm25.get_scores(tokens)
        import numpy as np
        top_k = int(min(k, len(scores)))
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((float(scores[idx]), self._bm25_corpus[idx]))
        return results

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_.]{1,}\b", text.lower())

    # ── Step 2c: Brave Search (open web) ──────────────────────────────────────

    def _brave_search(self, query: str, k: int) -> List[Dict]:
        """Brave Search API for open web coverage when local index misses."""
        if not self.brave_api_key:
            return self._ddg_fallback(query, k)
        try:
            import requests
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": k, "result_filter": "web"},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_api_key,
                },
                timeout=5,
            )
            resp.raise_for_status()
            results = []
            for item in resp.json().get("web", {}).get("results", []):
                results.append({
                    "text": item.get("description", ""),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "source_tier": self._classify_source(item.get("url", "")),
                })
            return results
        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return self._ddg_fallback(query, k)

    def _ddg_fallback(self, query: str, k: int) -> List[Dict]:
        """DuckDuckGo fallback when Brave key is unavailable."""
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS(timeout=5) as ddgs:
                for r in ddgs.text(query, max_results=k):
                    results.append({
                        "text": r.get("body", ""),
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                        "source_tier": self._classify_source(r.get("href", "")),
                    })
            return results
        except Exception as e:
            logger.error(f"DDG fallback also failed: {e}")
            return []

    @staticmethod
    def _classify_source(url: str) -> str:
        url_l = url.lower()
        if "github.com" in url_l:
            return "github"
        if "stackoverflow.com" in url_l:
            return "stackoverflow"
        if "developer.mozilla.org" in url_l:
            return "mdn"
        if "pypi.org" in url_l or "npmjs.com" in url_l:
            return "pypi_npm"
        for d in ["python.org", "go.dev", "docs.rust-lang.org", "kotlinlang.org"]:
            if d in url_l:
                return "lang_refs"
        if "arxiv.org" in url_l:
            return "arxiv"
        return "web"

    # ── Step 2 combined: hybrid search with RRF merge ─────────────────────────

    def _hybrid_search(
        self, query: str, query_emb
    ) -> List[RetrievedPassage]:
        """
        Run FAISS + BM25 + Brave in parallel, merge with Reciprocal Rank Fusion.
        """
        k = self.top_k_retrieve

        faiss_results: List[Tuple[float, Dict]] = []
        bm25_results: List[Tuple[float, Dict]] = []
        web_results: List[Dict] = []

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_faiss = pool.submit(self._faiss_search, query_emb, k)
            f_bm25 = pool.submit(self._bm25_search, query, k)
            f_web = pool.submit(self._brave_search, query, k)
            faiss_results = f_faiss.result()
            bm25_results = f_bm25.result()
            web_results = f_web.result()

        # Build passage dict keyed by URL for dedup
        passages: Dict[str, RetrievedPassage] = {}

        for rank, (score, chunk) in enumerate(faiss_results):
            url = chunk.get("url", f"faiss_{rank}")
            p = passages.setdefault(url, RetrievedPassage(
                text=chunk.get("text", ""), url=url,
                title=chunk.get("title", ""),
                source_tier=self._classify_source(url),
            ))
            p.vector_score = score
            # RRF contribution: 1 / (60 + rank)
            p.vector_score += 1.0 / (60 + rank)

        for rank, (score, chunk) in enumerate(bm25_results):
            url = chunk.get("url", f"bm25_{rank}")
            p = passages.setdefault(url, RetrievedPassage(
                text=chunk.get("text", ""), url=url,
                title=chunk.get("title", ""),
                source_tier=self._classify_source(url),
            ))
            p.bm25_score = score
            p.bm25_score += 1.0 / (60 + rank)

        for rank, chunk in enumerate(web_results):
            url = chunk.get("url", f"web_{rank}")
            p = passages.setdefault(url, RetrievedPassage(
                text=chunk.get("text", ""), url=url,
                title=chunk.get("title", ""),
                source_tier=chunk.get("source_tier", "web"),
            ))
            # RRF
            p.vector_score += 1.0 / (60 + rank)

        # Sort by combined RRF score (vector + bm25)
        all_passages = list(passages.values())
        all_passages.sort(
            key=lambda p: p.vector_score + p.bm25_score,
            reverse=True,
        )
        return all_passages[:k]

    # ── Step 3: Cross-encoder reranking ───────────────────────────────────────

    def _rerank(
        self, query: str, passages: List[RetrievedPassage]
    ) -> List[RetrievedPassage]:
        """
        Rerank passages using a cross-encoder (full attention over query+passage).
        Significantly more accurate than bi-encoder similarity alone.
        Model: cross-encoder/ms-marco-MiniLM-L-6-v2
        """
        if not passages:
            return passages
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            if self._reranker is None:
                self._reranker = CrossEncoder(self.rerank_model_name)
                logger.info(f"Loaded reranker: {self.rerank_model_name}")
            pairs = [(query, p.text[:512]) for p in passages]
            scores = self._reranker.predict(pairs)
            for p, score in zip(passages, scores):
                p.rerank_score = float(score)
            passages.sort(key=lambda p: p.rerank_score, reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed ({e}), using RRF order.")
        return passages
