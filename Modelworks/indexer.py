"""
Treuno 125M â€”  Document Indexer
Chunks retrieved documents, embeds them with sentence-transformers,
and stores them in a FAISS index for vector similarity search.

This gives Modelworks a persistent local knowledge base that
augments the real-time web retrieval with cached document chunks.

Usage:
    indexer = DocumentIndexer()
    indexer.add_documents([{"text": "...", "url": "..."}])
    results = indexer.query("how to use requests.Session", top_k=3)
"""

from __future__ import annotations
import os
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    url: str
    title: str = ""
    score: float = 0.0


class DocumentIndexer:
    """
    Chunk â†’ Embed â†’ FAISS index pipeline for Modelworks.

    Workflow:
      1. add_documents([{text, url, title}]) â€” chunks and indexes documents
      2. query("...") â€” returns top-K relevant chunks by cosine similarity
      3. save() / load() â€” persist index to disk between sessions
    """

    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        index_path: Optional[str] = None,
    ):
        self.embed_model_name = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = Path(index_path) if index_path else None

        self._encoder = None   # lazy load
        self._index = None     # FAISS index
        self._chunks: List[DocumentChunk] = []

    def _get_encoder(self):
        """Lazy-load sentence-transformers encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embed_model_name)
                logger.info(f"Loaded encoder: {self.embed_model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._encoder

    def _get_index(self, dim: int):
        """Create or return FAISS index."""
        if self._index is None:
            try:
                import faiss
                self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2 norm)
                logger.info(f"Created FAISS index with dim={dim}")
            except ImportError:
                raise ImportError("faiss required: pip install faiss-cpu")
        return self._index

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the index.

        Args:
            documents: list of dicts with keys: text, url, title (optional)

        Returns:
            Number of chunks added
        """
        import numpy as np
        encoder = self._get_encoder()
        new_chunks = []
        texts_to_embed = []

        for doc in documents:
            raw_text = doc.get("text", "")
            url = doc.get("url", "")
            title = doc.get("title", "")
            if not raw_text.strip():
                continue
            for chunk_text in self._chunk_text(raw_text):
                chunk_id = hashlib.md5(
                    f"{url}{chunk_text[:50]}".encode()
                ).hexdigest()[:12]
                new_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    url=url,
                    title=title,
                ))
                texts_to_embed.append(chunk_text)

        if not texts_to_embed:
            return 0

        # Embed all chunks
        embeddings = encoder.encode(
            texts_to_embed,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via IP
            show_progress_bar=len(texts_to_embed) > 20,
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Add to FAISS
        dim = embeddings.shape[1]
        index = self._get_index(dim)
        index.add(embeddings)
        self._chunks.extend(new_chunks)

        logger.info(f"Indexed {len(new_chunks)} chunks from {len(documents)} documents.")
        return len(new_chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[DocumentChunk]:
        """
        Semantic search over indexed chunks.

        Args:
            query_text: natural language or code query
            top_k:      number of results to return

        Returns:
            List of DocumentChunk sorted by relevance (highest first)
        """
        import numpy as np
        if not self._chunks or self._index is None or self._index.ntotal == 0:
            return []

        encoder = self._get_encoder()
        q_emb = encoder.encode(
            [query_text],
            normalize_embeddings=True,
        )
        q_emb = np.array(q_emb, dtype=np.float32)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            chunk.score = float(score)
            results.append(chunk)

        return results

    def save(self, path: Optional[str] = None) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        import faiss
        save_dir = Path(path) if path else self.index_path
        if save_dir is None:
            logger.warning("No save path specified, skipping save.")
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        if self._index and self._index.ntotal > 0:
            faiss.write_index(self._index, str(save_dir / "index.faiss"))
        meta = [
            {"chunk_id": c.chunk_id, "text": c.text, "url": c.url, "title": c.title}
            for c in self._chunks
        ]
        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved index to {save_dir} ({len(self._chunks)} chunks).")

    def load(self, path: Optional[str] = None) -> None:
        """Load FAISS index and chunk metadata from disk."""
        import faiss
        load_dir = Path(path) if path else self.index_path
        if load_dir is None or not load_dir.exists():
            return
        idx_path = load_dir / "index.faiss"
        chunks_path = load_dir / "chunks.json"
        if idx_path.exists():
            self._index = faiss.read_index(str(idx_path))
        if chunks_path.exists():
            with open(chunks_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._chunks = [
                DocumentChunk(**{k: v for k, v in m.items()}) for m in meta
            ]
        logger.info(f"Loaded index from {load_dir} ({len(self._chunks)} chunks).")

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return (
            f"DocumentIndexer(model={self.embed_model_name}, "
            f"chunks={len(self._chunks)}, "
            f"indexed={self._index.ntotal if self._index else 0})"
        )
