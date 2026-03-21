"""
Treuno 125M â€” Prometheus Monitoring + Execution Pass Rate Tracker
================================================================
Primary hallucination health metric: execution pass rate per language.

Prometheus metrics exposed at /metrics (via prometheus_client):

  treuno_generation_total               counter   Total generation requests
  treuno_generation_latency_seconds     histogram Request latency in seconds
  treuno_retrieval_latency_seconds      histogram Model-Retrieve latency
  treuno_cache_hits_total               counter   Model-Cache hits
  treuno_cache_misses_total             counter   Model-Cache misses
  treuno_verify_confidence              histogram Composite confidence scores
  treuno_intercepted_total              counter   Responses intercepted (< 0.75)
  treuno_execution_pass_total           counter   Code execution passes, labelled by language
  treuno_execution_fail_total           counter   Code execution failures, labelled by language
  treuno_execution_pass_rate            gauge     Pass rate per language (rolling 1h)
  treuno_lora_update_total              counter   LoRA hot-swaps performed
  treuno_lora_update_last_timestamp     gauge     Unix timestamp of last LoRA swap

Usage (standalone metrics server):
  python monitoring/metrics.py --port 9090

For FastAPI integration, import and call register_metrics(app).
"""

from __future__ import annotations
import time
import logging
import threading
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


# â”€â”€ Prometheus metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, REGISTRY,
        make_asgi_app, start_http_server,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics disabled.")

if _PROMETHEUS_AVAILABLE:
    GENERATION_TOTAL = Counter(
        "treuno_generation_total",
        "Total generation requests",
    )
    GENERATION_LATENCY = Histogram(
        "treuno_generation_latency_seconds",
        "End-to-end generation latency",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
    RETRIEVAL_LATENCY = Histogram(
        "treuno_retrieval_latency_seconds",
        "Model-Retrieve latency",
        buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
    )
    CACHE_HITS = Counter("treuno_cache_hits_total", "Model-Cache hits")
    CACHE_MISSES = Counter("treuno_cache_misses_total", "Model-Cache misses")
    VERIFY_CONFIDENCE = Histogram(
        "treuno_verify_confidence",
        "Model-Verify composite confidence scores",
        buckets=[0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    )
    INTERCEPTED_TOTAL = Counter(
        "treuno_intercepted_total",
        "Responses intercepted by Model-Verify (confidence < 0.75)",
    )
    EXECUTION_PASS = Counter(
        "treuno_execution_pass_total",
        "Code execution passes by language",
        labelnames=["language"],
    )
    EXECUTION_FAIL = Counter(
        "treuno_execution_fail_total",
        "Code execution failures by language",
        labelnames=["language"],
    )
    EXECUTION_PASS_RATE = Gauge(
        "treuno_execution_pass_rate",
        "Rolling 1-hour execution pass rate by language",
        labelnames=["language"],
    )
    LORA_UPDATES = Counter(
        "treuno_lora_update_total",
        "Number of LoRA hot-swaps performed",
    )
    LORA_LAST_TIMESTAMP = Gauge(
        "treuno_lora_update_last_timestamp",
        "Unix timestamp of the most recent LoRA update",
    )


# â”€â”€ Pass rate tracker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ExecutionPassRateTracker:
    """
    Tracks execution pass/fail events per language in a rolling 1-hour window.
    This is Treuno's PRIMARY hallucination health metric.

    Alert threshold: if any language drops below 60% pass rate â†’ PagerDuty.
    """
    WINDOW_SECONDS = 3600    # 1 hour rolling window
    ALERT_THRESHOLD = 0.60   # Alert at 60% pass rate

    def __init__(self):
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def record(self, language: str, passed: bool) -> None:
        """Record a code execution event."""
        now = time.time()
        with self._lock:
            q = self._events[language]
            q.append((now, passed))
        # Update Prometheus gauges (non-blocking)
        if _PROMETHEUS_AVAILABLE:
            if passed:
                EXECUTION_PASS.labels(language=language).inc()
            else:
                EXECUTION_FAIL.labels(language=language).inc()
            rate = self.pass_rate(language)
            EXECUTION_PASS_RATE.labels(language=language).set(rate)

    def pass_rate(self, language: str) -> float:
        """Compute rolling 1-hour pass rate for a language."""
        now = time.time()
        cutoff = now - self.WINDOW_SECONDS
        with self._lock:
            q = self._events[language]
            # Prune old events
            while q and q[0][0] < cutoff:
                q.popleft()
            if not q:
                return 1.0  # No data â†’ assume healthy
            passed = sum(1 for _, p in q if p)
            return passed / len(q)

    def all_pass_rates(self) -> dict[str, float]:
        return {lang: self.pass_rate(lang) for lang in self._events}

    def check_alerts(self) -> list[str]:
        """Return list of languages below the alert threshold."""
        alerts = []
        for lang, rate in self.all_pass_rates().items():
            if rate < self.ALERT_THRESHOLD:
                alerts.append(f"{lang}: {rate:.1%} pass rate (< {self.ALERT_THRESHOLD:.0%})")
        return alerts


# â”€â”€ Global tracker singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pass_rate_tracker = ExecutionPassRateTracker()


# â”€â”€ Helper decorators / context managers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class timed_generation:
    """Context manager that records generation latency to Prometheus."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        if _PROMETHEUS_AVAILABLE:
            GENERATION_TOTAL.inc()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._t0
        if _PROMETHEUS_AVAILABLE:
            GENERATION_LATENCY.observe(elapsed)


def record_retrieval(latency_ms: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        RETRIEVAL_LATENCY.observe(latency_ms / 1000)


def record_cache_hit(hit: bool) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()


def record_confidence(score: float, intercepted: bool) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    VERIFY_CONFIDENCE.observe(score)
    if intercepted:
        INTERCEPTED_TOTAL.inc()


def record_lora_update() -> None:
    if _PROMETHEUS_AVAILABLE:
        LORA_UPDATES.inc()
        LORA_LAST_TIMESTAMP.set(time.time())


def record_execution(language: str, passed: bool) -> None:
    pass_rate_tracker.record(language, passed)


# â”€â”€ FastAPI integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def register_metrics(app) -> None:
    """Mount Prometheus /metrics endpoint on an existing FastAPI app."""
    if not _PROMETHEUS_AVAILABLE:
        return
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    logger.info("Prometheus /metrics endpoint mounted.")


# â”€â”€ Standalone metrics server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def start_metrics_server(port: int = 9090) -> None:
    """Start a standalone Prometheus HTTP server."""
    if not _PROMETHEUS_AVAILABLE:
        logger.error("prometheus_client not installed.")
        return
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Treuno Prometheus metrics server")
    p.add_argument("--port", type=int, default=9090)
    args = p.parse_args()
    start_metrics_server(args.port)
    logger.info("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
            alerts = pass_rate_tracker.check_alerts()
            if alerts:
                logger.warning(f"ALERT: Low execution pass rate: {alerts}")
            else:
                rates = pass_rate_tracker.all_pass_rates()
                if rates:
                    summary = ", ".join(f"{l}={r:.0%}" for l, r in rates.items())
                    logger.info(f"Pass rates: {summary}")
    except KeyboardInterrupt:
        pass
