"""
Treuno 125M — Training Script
HuggingFace Trainer-based training loop with:
  - bfloat16 mixed precision
  - Gradient checkpointing for memory efficiency
  - LoRA-compatible (PEFT integration)
  - Data collator for causal LM (labels = input_ids shifted by 1)

Usage:
    python scripts/train.py --data-dir d:/MODEL/data/tokenized --output-dir d:/MODEL/weights
    python scripts/train.py --lora --epochs 1  # LoRA fine-tune (~2h on 2x A100)
"""

import os
import sys
import argparse
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenizedCodeDataset(Dataset):
    """
    Dataset for pre-tokenized code files (.bin shards from tokenize_data.py).
    Each shard is a numpy array of token IDs, packed into context_length chunks.
    """

    def __init__(self, data_dir: str, context_length: int = 8192):
        import numpy as np
        from pathlib import Path
        self.context_length = context_length
        self.chunks = []

        for shard_path in sorted(Path(data_dir).glob("*.bin")):
            data = np.fromfile(str(shard_path), dtype=np.uint16)
            # Pack into non-overlapping context_length windows
            n_chunks = len(data) // context_length
            for i in range(n_chunks):
                chunk = data[i * context_length: (i + 1) * context_length]
                self.chunks.append(torch.tensor(chunk.astype(int), dtype=torch.long))
        logger.info(f"Dataset: {len(self.chunks):,} chunks from {data_dir}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def causal_lm_collator(batch):
    """Data collator: input_ids = chunk, labels = chunk (model shifts internally)."""
    input_ids = torch.stack(batch)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    from transformers import TrainingArguments, Trainer
    from model.config import TreunoConfig
    from model.transformer import TreunoModel
    from model.tokenizer import TreunoTokenizer

    config = TreunoConfig.treuno_125m()
    model = TreunoModel(config)

    # Cast to bfloat16
    model = model.to(torch.bfloat16)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # ── Optional LoRA ────────────────────────────────────────────────────────
    if args.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA enabled: ~2M trainable parameters (~2h on 2x A100).")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = TokenizedCodeDataset(args.data_dir, config.context_length)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=causal_lm_collator,
    )
    trainer.train()

    # Save final weights
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    logger.info(f"Saved to {args.output_dir}/model.pt")


def main():
    p = argparse.ArgumentParser(description="Train Treuno 125M")
    p.add_argument("--data-dir", default="d:/MODEL/data/tokenized")
    p.add_argument("--output-dir", default="d:/MODEL/weights")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
