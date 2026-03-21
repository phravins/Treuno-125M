"""
Treuno 125M â€” Phase 4: Execution-Driven DPO
=============================================
Reward code that runs. Penalize code that crashes.

FULLY AUTOMATED label generation â€” no human annotators required:
  1. For each of 50,000 coding prompts, sample 2 completions from Phase 3 model
  2. Run both through Model-Execute (Docker+gVisor, 5s timeout)
  3. chosen  = completion whose code passed execution
     rejected = completion whose code failed
  4. Discard pairs where both pass or both fail

DPO config:
  Beta:    0.1  (per spec â€” controls deviation from reference policy)
  LR:      5e-5
  Epochs:  1

This phase eliminates hallucinated APIs and syntactically invalid code patterns
that survived SFT, by directly optimizing for runtime correctness.

Usage:
  # Step 1: Generate preference pairs (run on GPU)
  python scripts/train_phase4_dpo.py --generate-pairs \\
      --phase3-checkpoint d:/MODEL/checkpoints/phase3 \\
      --prompts-path d:/MODEL/data/dpo_prompts_50k.txt \\
      --pairs-output d:/MODEL/data/dpo_pairs_50k.jsonl

  # Step 2: DPO training
  python scripts/train_phase4_dpo.py --train \\
      --phase3-checkpoint d:/MODEL/checkpoints/phase3 \\
      --pairs-path d:/MODEL/data/dpo_pairs_50k.jsonl \\
      --output-dir d:/MODEL/checkpoints/phase4
"""

import os, sys, json, argparse, logging, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# â”€â”€ Step 1: Automatic pair generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_preference_pairs(args):
    """
    Run model twice on each prompt, execute both, label chosen/rejected.
    Writes to pairs_output as JSONL: {prompt, chosen, rejected}.
    """
    from sandbox.executor import CodeExecutor

    executor = CodeExecutor(default_timeout=5)
    prompts = []
    with open(args.prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    logger.info(f"Generating pairs for {len(prompts)} prompts...")
    from inference.engine import TreunoEngine
    engine = TreunoEngine.from_pretrained(
        args.phase3_checkpoint, use_retrieval=False, use_sandbox=False
    )

    written = 0
    with open(args.pairs_output, "w", encoding="utf-8") as fout:
        for i, prompt in enumerate(prompts):
            try:
                # Sample two completions at temperature 0.8 for diversity
                out_a = engine._run_model(prompt, max_new_tokens=256, temperature=0.8, top_p=0.95)
                out_b = engine._run_model(prompt, max_new_tokens=256, temperature=0.8, top_p=0.95)

                code_a = CodeExecutor.extract_code_block(out_a, "python") or out_a
                code_b = CodeExecutor.extract_code_block(out_b, "python") or out_b

                res_a = executor.run(code_a, "python")
                res_b = executor.run(code_b, "python")

                # Only keep mixed pairs
                if res_a.success and not res_b.success:
                    chosen, rejected = out_a, out_b
                elif res_b.success and not res_a.success:
                    chosen, rejected = out_b, out_a
                else:
                    continue   # Both pass or both fail â†’ discard

                fout.write(json.dumps({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }, ensure_ascii=False) + "\n")
                written += 1

                if (i + 1) % 500 == 0:
                    logger.info(f"  [{i+1}/{len(prompts)}] pairs so far: {written}")
            except Exception as e:
                logger.warning(f"  Pair generation failed for prompt {i}: {e}")

    logger.info(f"Generated {written} preference pairs â†’ {args.pairs_output}")


# â”€â”€ Step 2: DPO training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DPOPairsDataset:
    """JSONL dataset of {prompt, chosen, rejected} triples for DPO training."""
    def __init__(self, path: str):
        self.pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        logger.info(f"DPO dataset: {len(self.pairs)} preference pairs")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


def train_dpo(args):
    import torch
    from transformers import AutoTokenizer, TrainingArguments
    from trl import DPOTrainer, DPOConfig
    from model.config import TreunoConfig
    from model.transformer import TreunoModel

    cfg = TreunoConfig.treuno_125m()

    # Policy model (to be trained)
    model = TreunoModel(cfg).to(torch.bfloat16)
    if os.path.exists(os.path.join(args.phase3_checkpoint, "model.pt")):
        model.load_state_dict(torch.load(
            os.path.join(args.phase3_checkpoint, "model.pt"), map_location="cpu"
        ), strict=False)

    # Reference model (frozen Phase 3 checkpoint)
    ref_model = TreunoModel(cfg).to(torch.bfloat16)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = DPOPairsDataset(args.pairs_path)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        beta=0.1,           # Per spec
        max_length=512,
        max_prompt_length=256,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        report_to="mlflow" if args.mlflow else "none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"Phase 4 (Execution DPO) complete. Saved to {args.output_dir}")


def main():
    p = argparse.ArgumentParser(description="Treuno Phase 4: Execution-Driven DPO")
    p.add_argument("--generate-pairs", action="store_true", help="Generate preference pairs")
    p.add_argument("--train",          action="store_true", help="Run DPO training")
    p.add_argument("--phase3-checkpoint", default="d:/MODEL/checkpoints/phase3")
    p.add_argument("--prompts-path",   default="d:/MODEL/data/dpo_prompts_50k.txt")
    p.add_argument("--pairs-path",     default="d:/MODEL/data/dpo_pairs_50k.jsonl")
    p.add_argument("--pairs-output",   default="d:/MODEL/data/dpo_pairs_50k.jsonl")
    p.add_argument("--output-dir",     default="d:/MODEL/checkpoints/phase4")
    p.add_argument("--mlflow",         action="store_true")
    args = p.parse_args()

    if args.generate_pairs:
        generate_preference_pairs(args)
    if args.train:
        train_dpo(args)
    if not args.generate_pairs and not args.train:
        logger.error("Use --generate-pairs and/or --train to run this script.")


if __name__ == "__main__":
    main()
