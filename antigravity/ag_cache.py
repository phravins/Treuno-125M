"""
Treuno — AG-Cache
==================
Component 4 of the Antigravity framework.

Semantic response cache built on Redis 7.
Instead of exact string matching, AG-Cache compares incoming query embeddings
against previously answered queries using cosine similarity.

Behaviour:
  - Incoming query → embed → 384-dim vector
  - Scan recent cache entries for closest match (cosine similarity)
  - If similarity >= 0.92 → return cached verified response immediately
  - Otherwise → full pipeline (AG-Retrieve → model → AG-Execute → AG-Verify)
  - Results stored in Redis with TTL: 24h for web, 7d for doc-sourced answers

Expected latency reduction: ~60% for repeated or near-identical questions.

Redis schema:
  Key:   ag:cache:{hash}
  Value: JSON {
    "query":     original query string,
    "embedding": list[float] (384-dim),
    "response":  VerifiedResponse payload,
    "timestamp": unix epoch,
    "ttl":       seconds,
  }

  Index: ag:cache:index (Redis Set of all cache keys for scanning)
"""

from __future__ import annotations

import json
import time
import logging
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, List

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92     # Per spec: 0.92 → cache hit
DEFAULT_TTL = 86400             # 24h for open web answers
DOC_TTL = 604800                # 7d for documentation-sourced answers
INDEX_KEY = "ag:cache:index"
KEY_PREFIX = "ag:cache:"


@dataclass
class CacheEntry:
    query: str
    embedding: List[float]
    response_text: str
    score_composite: float
    retrieved_urls: List[str]
    timestamp: float
    ttl: int
    source_tier: str = "web"    # determines TTL on write


class AGCache:
    """
    AG-Cache: Redis 7 semantic query cache.

    Architecture:
      - All cache entries stored in Redis as JSON blobs
      - Embeddings stored inline (384 floats ≈ 6 KB per entry)
      - Scan-and-compare on every cache miss (fast for < 10K entries)
      - For larger deployments: upgrade to Redis Vector Similarity Search (VSS)

    Usage:
        cache = AGCache(redis_url="redis://localhost:6379")
        hit = cache.lookup("how to use asyncio in python")
        if hit:
            return hit.response_text
        # ...generate response...
        cache.store(query, embedding, response, urls, source_tier="lang_refs")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_scan_entries: int = 5000,
    ):
        self.redis_url = redis_url
        self.threshold = similarity_threshold
        self.max_scan = max_scan_entries
        self._redis = None
        self._encoder = None

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """
        Check if a semantically similar query has been answered before.

        Returns:
            CacheEntry if similarity >= 0.92, else None.
        """
        if not self._connect():
            return None
        try:
            query_emb = self._embed(query)
            return self._scan_and_match(query_emb)
        except Exception as e:
            logger.warning(f"AG-Cache lookup failed: {e}")
            return None

    def store(
        self,
        query: str,
        response_text: str,
        retrieved_urls: List[str],
        score_composite: float,
        source_tier: str = "web",
    ) -> bool:
        """
        Store a verified response in the cache.

        TTL is determined by source_tier:
          docs/lang_refs/SO: 7 days
          github:            1 hour
          web:               24 hours
        """
        if not self._connect():
            return False
        try:
            emb = self._embed(query)
            ttl = self._ttl_for_tier(source_tier)
            entry_key = KEY_PREFIX + hashlib.md5(query.encode()).hexdigest()[:16]
            payload = {
                "query": query,
                "embedding": emb.tolist(),
                "response_text": response_text,
                "score_composite": score_composite,
                "retrieved_urls": retrieved_urls,
                "timestamp": time.time(),
                "ttl": ttl,
                "source_tier": source_tier,
            }
            self._redis.setex(entry_key, ttl, json.dumps(payload))
            self._redis.sadd(INDEX_KEY, entry_key)
            self._redis.expire(INDEX_KEY, DOC_TTL)
            logger.debug(f"Cached response for '{query[:40]}' (TTL={ttl}s, key={entry_key})")
            return True
        except Exception as e:
            logger.warning(f"AG-Cache store failed: {e}")
            return False

    def invalidate(self, query: str) -> None:
        """Remove a specific cache entry (e.g. when source data has changed)."""
        if not self._connect():
            return
        entry_key = KEY_PREFIX + hashlib.md5(query.encode()).hexdigest()[:16]
        self._redis.delete(entry_key)
        self._redis.srem(INDEX_KEY, entry_key)

    def flush(self) -> int:
        """Clear all AG-Cache entries. Returns number deleted."""
        if not self._connect():
            return 0
        keys = self._redis.smembers(INDEX_KEY)
        if keys:
            self._redis.delete(*keys)
        self._redis.delete(INDEX_KEY)
        logger.info(f"Flushed {len(keys)} cache entries.")
        return len(keys)

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self._connect():
            return {"status": "disconnected"}
        try:
            keys = self._redis.smembers(INDEX_KEY)
            return {
                "status": "connected",
                "entries": len(keys),
                "threshold": self.threshold,
                "redis_url": self.redis_url,
            }
        except Exception as e:
            return {"status": f"error: {e}"}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        """Lazily connect to Redis. Returns False if Redis is unavailable."""
        if self._redis is not None:
            return True
        try:
            import redis
            self._redis = redis.from_url(self.redis_url, decode_responses=True, socket_timeout=1)
            self._redis.ping()
            logger.info(f"AG-Cache connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"AG-Cache Redis unavailable ({e}). Cache disabled.")
            self._redis = None
            return False

    def _embed(self, text: str):
        """Embed a query string → 384-dim numpy array."""
        import numpy as np
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        vec = self._encoder.encode([text], normalize_embeddings=True)
        return vec[0]

    def _scan_and_match(self, query_emb) -> Optional[CacheEntry]:
        """
        Scan all cache keys and return the entry with highest cosine similarity
        if it exceeds the threshold.
        """
        import numpy as np
        keys = list(self._redis.smembers(INDEX_KEY))
        if not keys:
            return None

        # Scan up to max_scan most recent entries
        keys = keys[:self.max_scan]

        best_sim = -1.0
        best_entry = None

        for key in keys:
            raw = self._redis.get(key)
            if raw is None:
                self._redis.srem(INDEX_KEY, key)
                continue
            try:
                data = json.loads(raw)
                cached_emb = np.array(data["embedding"], dtype=np.float32)
                # Cosine similarity (both are L2-normalized)
                sim = float(np.dot(query_emb, cached_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = data
            except (json.JSONDecodeError, KeyError):
                continue

        if best_sim >= self.threshold and best_entry is not None:
            logger.info(
                f"AG-Cache HIT (sim={best_sim:.3f} >= {self.threshold}) "
                f"for query: '{best_entry.get('query','')[:40]}'"
            )
            return CacheEntry(
                query=best_entry["query"],
                embedding=best_entry["embedding"],
                response_text=best_entry["response_text"],
                score_composite=best_entry.get("score_composite", 1.0),
                retrieved_urls=best_entry.get("retrieved_urls", []),
                timestamp=best_entry.get("timestamp", 0.0),
                ttl=best_entry.get("ttl", DEFAULT_TTL),
                source_tier=best_entry.get("source_tier", "web"),
            )
        return None

    @staticmethod
    def _ttl_for_tier(source_tier: str) -> int:
        """Return cache TTL in seconds based on source tier."""
        ttl_map = {
            "github":       3600,    # 1 hour
            "web":          86400,   # 24 hours
            "stackoverflow": 604800, # 7 days
            "mdn":          604800,
            "pypi_npm":     604800,
            "lang_refs":    604800,
            "arxiv":        604800,
        }
        return ttl_map.get(source_tier, DEFAULT_TTL)

    def __repr__(self) -> str:
        connected = self._redis is not None
        return (
            f"AGCache(redis={self.redis_url}, "
            f"threshold={self.threshold}, "
            f"connected={connected})"
        )
