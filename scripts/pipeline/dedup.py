"""
Treuno — MinHash LSH Deduplication
=====================================
MinHash Locality Sensitive Hashing deduplication using datasketch.

Removes near-duplicate code examples before they enter the training pipeline.
Prevents the model from over-fitting to repeated code patterns.

Config:
  N-gram size:      5         (character 5-grams give best recall for code)
  Hash functions:   128       (balance between precision and speed)
  LSH threshold:    0.8       (pairs with Jaccard similarity >= 0.8 → duplicate)
  Bands:            Computed from threshold and num_perm automatically

Expected dedup rate: ~15–25% of raw web code is near-duplicate.

Usage:
  python scripts/pipeline/dedup.py \\
      --input-dir d:/MODEL/data/raw \\
      --output-dir d:/MODEL/data/deduped \\
      --threshold 0.8
"""

from __future__ import annotations
import os, json, argparse, logging, time
from pathlib import Path
from typing import Iterator, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NGRAM_SIZE = 5
NUM_PERM = 128


def text_to_minhash(text: str, n: int = NGRAM_SIZE, num_perm: int = NUM_PERM):
    """Compute a MinHash signature from character n-grams of text."""
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - n + 1):
        m.update(text[i:i+n].encode("utf8"))
    return m


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def deduplicate_directory(
    input_dir: str,
    output_dir: str,
    threshold: float = 0.8,
    text_field: str = "text",
) -> Dict[str, int]:
    """
    Run MinHash LSH deduplication over all JSONL files in input_dir.

    Returns counts: {total, unique, duplicates_removed}
    """
    from datasketch import MinHashLSH
    os.makedirs(output_dir, exist_ok=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    stats = {"total": 0, "unique": 0, "duplicates_removed": 0}

    for input_path in sorted(Path(input_dir).glob("*.jsonl")):
        out_path = Path(output_dir) / input_path.name
        logger.info(f"Deduplicating {input_path.name}...")
        t0 = time.perf_counter()
        written = 0

        with open(str(out_path), "w", encoding="utf-8") as fout:
            for i, record in enumerate(iter_jsonl(str(input_path))):
                stats["total"] += 1
                text = record.get(text_field, "")
                if len(text) < 100:
                    stats["duplicates_removed"] += 1
                    continue

                key = f"{input_path.stem}_{i}"
                m = text_to_minhash(text)

                # Check if this document is a near-duplicate of anything we've seen
                result = lsh.query(m)
                if result:
                    stats["duplicates_removed"] += 1
                    continue

                lsh.insert(key, m)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                stats["unique"] += 1

        elapsed = time.perf_counter() - t0
        logger.info(f"  {input_path.name}: {stats['total']} in, {written} out ({elapsed:.1f}s)")

    dedup_rate = 100 * stats["duplicates_removed"] / max(stats["total"], 1)
    logger.info(f"\nDeduplication complete:")
    logger.info(f"  Total:     {stats['total']:,}")
    logger.info(f"  Unique:    {stats['unique']:,}")
    logger.info(f"  Removed:   {stats['duplicates_removed']:,} ({dedup_rate:.1f}%)")
    return stats


def main():
    p = argparse.ArgumentParser(description="MinHash LSH deduplication for Treuno training data")
    p.add_argument("--input-dir",  default="d:/MODEL/data/raw")
    p.add_argument("--output-dir", default="d:/MODEL/data/deduped")
    p.add_argument("--threshold",  type=float, default=0.8)
    p.add_argument("--text-field", default="text")
    args = p.parse_args()
    deduplicate_directory(args.input_dir, args.output_dir, args.threshold, args.text_field)


if __name__ == "__main__":
    main()
