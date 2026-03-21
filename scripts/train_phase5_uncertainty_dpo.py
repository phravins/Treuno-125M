"""
Treuno 125M â€” Phase 5: Uncertainty DPO
========================================
Train the model to say "I don't know" rather than hallucinate confidently.

10,000 preference pairs:
  chosen:   Response that acknowledges uncertainty, cites sources, recommends cross-checking
  rejected: Response that makes up an answer or invents an API that doesn't exist

This phase is what makes Treuno's hallucination behaviour qualitatively different
from standard LLMs â€” it has an explicit incentive to abstain.

DPO config: same as Phase 4 (beta=0.1).

Usage:
  python scripts/train_phase5_uncertainty_dpo.py \\
      --phase4-checkpoint d:/MODEL/checkpoints/phase4 \\
      --pairs-path d:/MODEL/data/uncertainty_pairs_10k.jsonl \\
      --output-dir d:/MODEL/checkpoints/phase5
"""

import os, sys, json, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# â”€â”€ Data validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

UNCERTAINTY_MARKERS = [
    "i'm not sure", "i cannot verify", "i'm uncertain",
    "you should check", "please cross-check", "i don't have",
    "as of my knowledge", "i cannot confirm",
]

def validate_pair(pair: dict) -> bool:
    """
    A valid uncertainty pair must:
      - chosen:   contain an uncertainty marker
      - rejected: NOT contain an uncertainty marker
    """
    chosen = pair.get("chosen", "").lower()
    rejected = pair.get("rejected", "").lower()
    chosen_ok = any(m in chosen for m in UNCERTAINTY_MARKERS)
    rejected_overconfident = not any(m in rejected for m in UNCERTAINTY_MARKERS)
    return chosen_ok and rejected_overconfident


class UncertaintyDPODataset:
    def __init__(self, path: str, validate: bool = True):
        self.pairs = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pair = json.loads(line)
                    if validate and not validate_pair(pair):
                        skipped += 1
                        continue
                    self.pairs.append(pair)
                except json.JSONDecodeError:
                    pass
        logger.info(f"Uncertainty DPO: {len(self.pairs)} valid pairs ({skipped} skipped validation)")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


def train(args):
    import torch
    from transformers import AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from model.config import TreunoConfig
    from model.transformer import TreunoModel

    cfg = TreunoConfig.treuno_125m()

    model = TreunoModel(cfg).to(torch.bfloat16)
    ckpt = os.path.join(args.phase4_checkpoint, "model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        logger.info("Loaded Phase 4 checkpoint.")

    ref_model = TreunoModel(cfg).to(torch.bfloat16)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = UncertaintyDPODataset(args.pairs_path, validate=True)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
        bf16=True,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to="mlflow" if args.mlflow else "none",
    )

    trainer = DPOTrainer(
        model=model, ref_model=ref_model,
        args=dpo_config, train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    logger.info(f"Phase 5 (Uncertainty DPO) complete. Saved to {args.output_dir}")
    logger.info("This is the final pre-deployment checkpoint.")


def main():
    p = argparse.ArgumentParser(description="Treuno Phase 5: Uncertainty DPO")
    p.add_argument("--phase4-checkpoint", default="d:/MODEL/checkpoints/phase4")
    p.add_argument("--pairs-path",  default="d:/MODEL/data/uncertainty_pairs_10k.jsonl")
    p.add_argument("--output-dir",  default="d:/MODEL/checkpoints/phase5")
    p.add_argument("--mlflow",      action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
