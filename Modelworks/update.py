"""
Treuno — AG-Update
===================

The continuous learning pipeline that keeps Treuno current without full retraining.

Two-track update system:

  Track A — Source Index Refresh (runs continuously)
    Crawls registered source tiers on their respective TTL schedules:
      GitHub repos:    re-indexed every 1 hour
      Stack Overflow:  re-indexed every 7 days
      Docs/MDN:        re-indexed every 7 days
      Open web:        re-indexed every 24 hours
    New documents are embedded and added to the FAISS IVF-PQ index.
    The AG-Cache is selectively invalidated when a new version of a
    frequently-retrieved document is detected.

  Track B — LoRA Weight Update (runs weekly via cron)
    Collects verified (confidence >= 0.75) response pairs from the past week.
    Formats them as (prompt, code) supervised examples.
    Runs a LoRA fine-tune (r=64, α=128) for 1 epoch on 2x A100 GPUs.
    Completes in approximately 2 hours.
    New weights are swapped in with zero downtime using a shadow-load pattern:
      1. Load new weights into a second model instance
      2. Warm up with 100 queries
      3. If validation perplexity improved → atomically swap to new instance
      4. Old instance released from memory

Trigger schedule (cron expressions):
    Source refresh:   "0 * * * *"      (every hour, selective per tier)
    LoRA update:      "0 2 * * 1"      (Mondays at 02:00 UTC)
    Cache cleanup:    "30 * * * *"     (every hour at :30)
"""

from __future__ import annotations

import json
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class UpdateJob:
    job_id: str
    job_type: str           # "source_refresh" | "lora_update" | "cache_cleanup"
    triggered_at: float
    completed_at: Optional[float] = None
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return self.completed_at - self.triggered_at
        return time.time() - self.triggered_at


@dataclass
class TrainingExample:
    """A verified (prompt, response) pair collected for LoRA training."""
    prompt: str
    response: str
    language: str
    sources: List[str]
    confidence_score: float
    timestamp: float


class AGUpdate:
    """
    AG-Update: continuous two-track update pipeline.

    Track A: source index refresh (background thread, per-TTL schedule)
    Track B: weekly LoRA fine-tune (cron-triggered, 2h on 2x A100)
    """

    def __init__(
        self,
        retriever=None,              # AGRetrieve instance
        examples_db_path: str = "d:/MODEL/data/update_buffer.jsonl",
        weights_dir: str = "d:/MODEL/weights",
        min_examples_for_lora: int = 500,
        confidence_threshold: float = 0.75,
    ):
        self.retriever = retriever
        self.examples_db_path = Path(examples_db_path)
        self.weights_dir = Path(weights_dir)
        self.confidence_threshold = confidence_threshold
        self.min_examples_for_lora = min_examples_for_lora

        self._jobs: List[UpdateJob] = []
        self._running = False
        self._refresh_thread: Optional[threading.Thread] = None

    # ── Track A: Source refresh ───────────────────────────────────────────────

    def start_background_refresh(self, interval_seconds: int = 3600) -> None:
        """
        Start a background thread that refreshes the source index periodically.
        interval_seconds: how often to check all tiers (default: 1 hour)
        """
        if self._running:
            logger.warning("AG-Update background refresh already running.")
            return
        self._running = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            args=(interval_seconds,),
            daemon=True,
            name="ag-update-refresh",
        )
        self._refresh_thread.start()
        logger.info(f"AG-Update background refresh started (every {interval_seconds}s).")

    def stop_background_refresh(self) -> None:
        self._running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        logger.info("AG-Update background refresh stopped.")

    def _refresh_loop(self, interval: int) -> None:
        """Main refresh loop running in background thread."""
        while self._running:
            try:
                self.run_source_refresh()
            except Exception as e:
                logger.error(f"Source refresh error: {e}", exc_info=True)
            time.sleep(interval)

    def run_source_refresh(self) -> UpdateJob:
        """
        Refresh the FAISS + BM25 index with new documents from all source tiers.
        Returns an UpdateJob with statistics.
        """
        import hashlib, random
        job = UpdateJob(
            job_id=f"refresh_{int(time.time())}",
            job_type="source_refresh",
            triggered_at=time.time(),
        )
        logger.info("AG-Update: running source index refresh...")
        new_docs = 0

        # In production: crawl each source tier per its TTL schedule.
        # Here we stub the crawl logic and focus on the architecture.
        try:
            sources_to_refresh = self._get_sources_due_for_refresh()
            for source_config in sources_to_refresh:
                docs = self._crawl_source(source_config)
                if docs and self.retriever:
                    texts = [d["text"] for d in docs]
                    urls  = [d["url"] for d in docs]
                    titles = [d.get("title", "") for d in docs]
                    self.retriever.build_faiss_ivfpq(texts, urls, titles)
                    self.retriever.build_bm25(texts, docs)
                    new_docs += len(docs)
                    logger.info(f"Refreshed {source_config['tier']}: {len(docs)} documents.")

            job.success = True
            job.details = {"new_documents": new_docs, "sources_refreshed": len(sources_to_refresh)}
        except Exception as e:
            job.success = False
            job.details = {"error": str(e)}
            logger.error(f"Source refresh failed: {e}")

        job.completed_at = time.time()
        self._jobs.append(job)
        return job

    def _get_sources_due_for_refresh(self) -> List[Dict]:
        """
        Returns source configs whose TTL has elapsed since last crawl.
        In production: check last-crawled timestamps from Redis.
        """
        # Stub: return placeholder configs for all tiers
        return [
            {"tier": "stackoverflow", "query": "python latest", "ttl": 604800},
            {"tier": "github",        "query": "python library stars:>1000", "ttl": 3600},
            {"tier": "lang_refs",     "query": "python documentation", "ttl": 604800},
        ]

    def _crawl_source(self, source_config: Dict) -> List[Dict]:
        """
        Crawl a single source tier and return document dicts.
        In production: use source-specific spiders (GitHub API, SO API, etc.)
        """
        # Stub — returns empty list; real implementation uses source APIs
        logger.debug(f"Crawling source tier: {source_config['tier']}")
        return []

    # ── Track B: LoRA weight update ───────────────────────────────────────────

    def collect_training_example(self, example: TrainingExample) -> None:
        """
        Save a verified response pair to the training buffer for next LoRA run.
        Only examples with confidence_score >= 0.75 are collected.
        """
        if example.confidence_score < self.confidence_threshold:
            return
        self.examples_db_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "prompt": example.prompt,
            "response": example.response,
            "language": example.language,
            "sources": example.sources,
            "confidence_score": example.confidence_score,
            "timestamp": example.timestamp,
        }
        with open(self.examples_db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def count_buffered_examples(self) -> int:
        """How many training examples are in the buffer?"""
        if not self.examples_db_path.exists():
            return 0
        with open(self.examples_db_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def run_lora_update(
        self,
        num_gpus: int = 2,
        dry_run: bool = False,
    ) -> UpdateJob:
        """
        Run the weekly LoRA fine-tune on collected training examples.

        Steps:
          1. Load buffered training examples
          2. Format as SFT dataset (prompt→response)
          3. LoRA fine-tune (r=64, α=128, q_proj + v_proj)
          4. Validate perplexity on held-out set
          5. Shadow-load and compare to current weights
          6. Swap if improved, rollback if not

        Target: ~2 hours on 2x A100 80GB GPUs
        """
        job = UpdateJob(
            job_id=f"lora_{int(time.time())}",
            job_type="lora_update",
            triggered_at=time.time(),
        )
        n = self.count_buffered_examples()
        logger.info(f"AG-Update LoRA: {n} examples in buffer (min={self.min_examples_for_lora})")

        if n < self.min_examples_for_lora:
            job.success = False
            job.completed_at = time.time()
            job.details = {"reason": f"Insufficient examples: {n} < {self.min_examples_for_lora}"}
            logger.warning(f"LoRA update skipped: not enough examples ({n}).")
            return job

        if dry_run:
            logger.info("AG-Update LoRA: dry_run=True, skipping actual training.")
            job.success = True
            job.completed_at = time.time()
            job.details = {"dry_run": True, "examples": n}
            return job

        try:
            logger.info("AG-Update LoRA: launching training...")
            import subprocess, sys
            cmd = [
                sys.executable,
                "scripts/train.py",
                "--lora",
                "--epochs", "1",
                "--data-dir", str(self.examples_db_path.parent),
                "--output-dir", str(self.weights_dir / "lora_candidate"),
            ]
            result = subprocess.run(cmd, cwd="d:/MODEL", capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("LoRA training completed. Running shadow validation...")
                swapped = self._shadow_swap(str(self.weights_dir / "lora_candidate"))
                job.success = swapped
                job.details = {"examples": n, "swapped": swapped}
                if swapped:
                    # Clear the training buffer after successful update
                    self.examples_db_path.unlink(missing_ok=True)
                    logger.info("AG-Update: new weights deployed, buffer cleared.")
            else:
                job.success = False
                job.details = {"error": result.stderr[-500:]}
        except Exception as e:
            job.success = False
            job.details = {"error": str(e)}
            logger.error(f"LoRA update failed: {e}", exc_info=True)

        job.completed_at = time.time()
        self._jobs.append(job)
        return job

    def _shadow_swap(self, candidate_path: str) -> bool:
        """
        Shadow-load candidate weights, validate, and optionally hot-swap.
        Returns True if swap was performed.
        """
        # In production: load candidate weights → compute perplexity on
        # held-out eval set → if improved by > 0.5% → atomic pointer swap.
        # For now: stub that always approves the swap.
        logger.info(f"Shadow validation for {candidate_path}...")
        return True

    def job_history(self) -> List[dict]:
        return [
            {
                "job_id": j.job_id,
                "type": j.job_type,
                "success": j.success,
                "duration_s": round(j.duration_seconds, 2),
                "details": j.details,
            }
            for j in self._jobs[-20:]
        ]

    def __repr__(self) -> str:
        n = self.count_buffered_examples()
        return (
            f"AGUpdate(buffer_examples={n}, "
            f"min_for_lora={self.min_examples_for_lora}, "
            f"running={self._running})"
        )
