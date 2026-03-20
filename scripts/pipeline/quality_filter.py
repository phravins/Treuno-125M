"""
Treuno — DistilBERT Quality Classifier
========================================
Filters low-quality code examples before they enter the training pipeline.

The classifier is a DistilBERT model fine-tuned to predict whether a code
document is high-quality (educational, idiomatic, well-structured) vs
low-quality (auto-generated boilerplate, obfuscated, test data, empty stubs).

Score threshold: > 0.6 to pass (set conservatively to avoid false rejections).

In production: replace with the CodeBERT or StarEncoder classifier trained on
internally labeled quality data, as described in the Treuno spec.

Usage:
  python scripts/pipeline/quality_filter.py \\
      --input-dir d:/MODEL/data/deduped \\
      --output-dir d:/MODEL/data/quality_filtered \\
      --threshold 0.6
"""

from __future__ import annotations
import os, json, argparse, logging, re
from pathlib import Path
from typing import Dict, List, Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.6


# ── Heuristic pre-filter (fast, no model) ────────────────────────────────────

def heuristic_quality_score(text: str) -> float:
    """
    Fast heuristic quality score (0.0–1.0) before invoking DistilBERT.
    Used to quickly reject obvious garbage without GPU inference.
    """
    score = 1.0
    lines = text.splitlines()
    if not lines:
        return 0.0

    # Penalize very short or very long files
    if len(text) < 200:
        score -= 0.3
    if len(text) > 500_000:
        score -= 0.2

    # Penalize high ratio of numeric / symbol lines
    symbol_lines = sum(1 for l in lines if re.match(r"^[\d\s,;.!?|+\-=_*]{10,}$", l))
    if symbol_lines / max(len(lines), 1) > 0.3:
        score -= 0.3

    # Penalize dead code / TODO stubs
    todo_count = sum(1 for l in lines if re.search(r"#\s*(TODO|FIXME|HACK|XXX)", l, re.I))
    if todo_count > 5:
        score -= 0.1

    # Penalize minified code (single very long line)
    if lines and max(len(l) for l in lines) > 500:
        score -= 0.3

    return max(0.0, min(1.0, score))


# ── DistilBERT classifier ─────────────────────────────────────────────────────

class DistilBERTQualityClassifier:
    """
    DistilBERT-based code quality classifier.

    In production: load a fine-tuned checkpoint from d:/MODEL/classifiers/quality/.
    Fallback: uses heuristics only if the checkpoint is not present.
    """

    def __init__(self, model_path: str = "d:/MODEL/classifiers/quality",
                 use_heuristics_only: bool = False):
        self.model_path = model_path
        self.use_heuristics_only = use_heuristics_only
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """Lazy-load the classifier model."""
        if self._model is not None:
            return
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            if os.path.isdir(self.model_path):
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                # Fall back to base DistilBERT (untrained — heuristics will dominate)
                logger.warning(
                    f"Quality classifier not found at {self.model_path}. "
                    "Using base DistilBERT (scores will not be meaningful). "
                    "Fine-tune on your own quality-labeled data to fix this."
                )
                self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2
                )
            self._model = self._model.to(self._device)
            self._model.eval()
        except ImportError:
            logger.warning("transformers not installed. Heuristics-only mode.")
            self.use_heuristics_only = True

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a batch of texts, returning quality probability in [0, 1]."""
        if self.use_heuristics_only:
            return [heuristic_quality_score(t) for t in texts]
        self._load_model()
        if self._model is None:
            return [heuristic_quality_score(t) for t in texts]
        import torch
        import torch.nn.functional as F
        results = []
        for text in texts:
            h_score = heuristic_quality_score(text)
            if h_score < 0.2:
                results.append(h_score)
                continue
            try:
                enc = self._tokenizer(
                    text[:2048], return_tensors="pt",
                    truncation=True, max_length=512, padding=True,
                ).to(self._device)
                with torch.no_grad():
                    logits = self._model(**enc).logits
                    prob = F.softmax(logits, dim=-1)[0, 1].item()
                # Blend DistilBERT score with heuristic (70/30)
                results.append(0.7 * prob + 0.3 * h_score)
            except Exception:
                results.append(h_score)
        return results


def filter_directory(
    input_dir: str,
    output_dir: str,
    threshold: float = QUALITY_THRESHOLD,
    batch_size: int = 32,
) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    classifier = DistilBERTQualityClassifier()
    stats = {"total": 0, "passed": 0, "rejected": 0}

    for input_path in sorted(Path(input_dir).glob("*.jsonl")):
        out_path = Path(output_dir) / input_path.name
        batch_records, batch_texts = [], []

        with open(str(out_path), "w", encoding="utf-8") as fout:
            def flush_batch():
                if not batch_records:
                    return
                scores = classifier.score_batch(batch_texts)
                for rec, score in zip(batch_records, scores):
                    stats["total"] += 1
                    if score >= threshold:
                        rec["quality_score"] = round(score, 3)
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats["passed"] += 1
                    else:
                        stats["rejected"] += 1
                batch_records.clear()
                batch_texts.clear()

            for record in _iter_jsonl(str(input_path)):
                batch_records.append(record)
                batch_texts.append(record.get("text", ""))
                if len(batch_records) >= batch_size:
                    flush_batch()
            flush_batch()

        pass_rate = 100 * stats["passed"] / max(stats["total"], 1)
        logger.info(f"{input_path.name}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")

    return stats


def _iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def main():
    p = argparse.ArgumentParser(description="DistilBERT quality filter for Treuno")
    p.add_argument("--input-dir",  default="d:/MODEL/data/deduped")
    p.add_argument("--output-dir", default="d:/MODEL/data/quality_filtered")
    p.add_argument("--threshold",  type=float, default=QUALITY_THRESHOLD)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--heuristics-only", action="store_true",
                   help="Skip DistilBERT, use heuristics only (faster)")
    args = p.parse_args()

    stats = filter_directory(args.input_dir, args.output_dir, args.threshold, args.batch_size)
    pass_rate = 100 * stats["passed"] / max(stats["total"], 1)
    print(f"\nQuality filter complete: {stats['passed']:,}/{stats['total']:,} passed ({pass_rate:.1f}%)")
    print(f"Rejected: {stats['rejected']:,}")


if __name__ == "__main__":
    main()
