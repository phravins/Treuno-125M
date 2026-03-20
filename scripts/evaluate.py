"""
Treuno 125M — Evaluation Harness
Benchmarks the model on HumanEval and MBPP coding benchmarks.

Metrics:
  - pass@1: fraction of problems solved on the first attempt
  - pass@k: fraction solved within k attempts (k=1,10,100)

Usage:
    python scripts/evaluate.py --model-path d:/MODEL/weights
    python scripts/evaluate.py --benchmark mbpp --samples 50
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k from Chen et al. (HumanEval paper).
    n = total samples, c = correct samples, k = k in pass@k.
    """
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_humaneval(split: str = "openai_humaneval") -> List[dict]:
    """Load HumanEval dataset from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset(split, split="test", trust_remote_code=True)
        return list(ds)
    except Exception as e:
        logger.error(f"Could not load HumanEval: {e}")
        return []


def load_mbpp(split: str = "mbpp") -> List[dict]:
    """Load MBPP dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset(split, split="test", trust_remote_code=True)
        return list(ds)
    except Exception as e:
        logger.error(f"Could not load MBPP: {e}")
        return []


def evaluate_humaneval(engine, problems: List[dict], n_samples: int = 1) -> dict:
    """Evaluate on HumanEval problems."""
    from sandbox.executor import CodeExecutor
    executor = CodeExecutor(default_timeout=10)
    results = []

    for i, problem in enumerate(problems):
        task_id = problem.get("task_id", f"HumanEval/{i}")
        prompt_text = problem.get("prompt", "")
        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")

        correct = 0
        for _ in range(n_samples):
            result = engine.generate(
                prompt=prompt_text,
                language="python",
                use_retrieval=False,    # eval without retrieval for fair comparison
                use_sandbox=False,
                max_new_tokens=256,
            )
            code = result.code or ""
            # Run code + test harness
            full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"
            exec_result = executor.run(full_code, "python")
            if exec_result.success:
                correct += 1

        results.append({
            "task_id": task_id,
            "n": n_samples,
            "correct": correct,
            "pass@1": estimate_pass_at_k(n_samples, correct, 1),
        })

        if (i + 1) % 10 == 0:
            avg = sum(r["pass@1"] for r in results) / len(results)
            logger.info(f"HumanEval [{i+1}/{len(problems)}] — running pass@1: {avg:.3f}")

    avg_pass1 = sum(r["pass@1"] for r in results) / len(results)
    return {
        "benchmark": "HumanEval",
        "n_problems": len(problems),
        "samples_per_problem": n_samples,
        "pass@1": avg_pass1,
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Treuno 125M")
    p.add_argument("--model-path", default="d:/MODEL/weights")
    p.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    p.add_argument("--samples", type=int, default=1, help="Samples per problem")
    p.add_argument("--max-problems", type=int, default=0, help="0 = all")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args()

    from inference.engine import TreunoEngine
    logger.info("Loading engine...")
    engine = TreunoEngine.from_pretrained(args.model_path, use_retrieval=False)

    if args.benchmark == "humaneval":
        problems = load_humaneval()
    else:
        problems = load_mbpp()

    if args.max_problems:
        problems = problems[:args.max_problems]

    logger.info(f"Running {args.benchmark} on {len(problems)} problems...")
    eval_result = evaluate_humaneval(engine, problems, n_samples=args.samples)

    print(f"\n{'='*50}")
    print(f"  {eval_result['benchmark']} Results")
    print(f"{'='*50}")
    print(f"  Problems:     {eval_result['n_problems']}")
    print(f"  Samples:      {eval_result['samples_per_problem']}")
    print(f"  pass@1:       {eval_result['pass@1']:.4f}  ({eval_result['pass@1']*100:.1f}%)")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(eval_result, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
