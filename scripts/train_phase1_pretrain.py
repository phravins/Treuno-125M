"""
Treuno 125M — Phase 1: Pretraining
=====================================
100 billion tokens of deduplicated, license-checked code.

Data sources:
  The Stack v2       55B tokens
  GitHub/RedPajama   25B tokens  (stars >= 100, CI passing)
  CodeSearchNet      10B tokens
  Stack Overflow     10B tokens  (accepted answers)

FIM: 20% of batches formatted as fill-in-the-middle.

Config:
  Optimizer:   AdamW (β1=0.9, β2=0.95, ε=1e-8, wd=0.1)
  LR schedule: cosine 2e-4 → 2e-5
  Warmup:      2,000 steps
  Context:     4,096 tokens (extended in Phase 2)
  Batch:       512 seq × 4096 tokens = ~2.1M tokens/step
  Precision:   bfloat16
  Attention:   FlashAttention-2
  Distributed: DeepSpeed ZeRO-2, 4× A100 40GB
  Duration:    ~4 days (~47,000 steps at 2.1M tok/step = 100B tokens)

Usage:
  deepspeed --num_gpus 4 scripts/train_phase1_pretrain.py \\
      --data-dir d:/MODEL/data/pretrain \\
      --output-dir d:/MODEL/checkpoints/phase1 \\
      --deepspeed configs/deepspeed_zero2.json
"""

import os
import sys
import math
import random
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── FIM Transform ─────────────────────────────────────────────────────────────

def apply_fim_transform(token_ids: list, fim_prefix_id: int, fim_middle_id: int,
                         fim_suffix_id: int, eos_id: int) -> list:
    """
    Randomly restructure a token sequence into FIM format.

    SPM (suffix-prefix-middle) format:
        <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle <EOS>
    """
    n = len(token_ids)
    if n < 10:
        return token_ids
    # Pick two random split points
    a = random.randint(1, n // 3)
    b = random.randint(a + 1, 2 * n // 3)
    prefix = token_ids[:a]
    middle = token_ids[a:b]
    suffix = token_ids[b:]
    return (
        [fim_prefix_id] + prefix +
        [fim_suffix_id] + suffix +
        [fim_middle_id] + middle + [eos_id]
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class PretrainDataset:
    """
    Streams tokenized shards from disk and applies FIM transform to 20% of batches.
    Each shard is a .bin file of uint16 token IDs packed without padding.
    """

    def __init__(self, data_dir: str, context_length: int, fim_rate: float = 0.20,
                 fim_prefix_id: int = 32765, fim_middle_id: int = 32766,
                 fim_suffix_id: int = 32767, eos_id: int = 2):
        import numpy as np
        import torch
        from pathlib import Path

        self.context = context_length
        self.fim_rate = fim_rate
        self.fim_prefix_id = fim_prefix_id
        self.fim_middle_id = fim_middle_id
        self.fim_suffix_id = fim_suffix_id
        self.eos_id = eos_id

        self.chunks = []
        shard_paths = sorted(Path(data_dir).glob("*.bin"))
        logger.info(f"Loading {len(shard_paths)} shards from {data_dir}...")
        for path in shard_paths:
            data = np.fromfile(str(path), dtype=np.uint16).astype(np.int64)
            n = len(data) // context_length
            for i in range(n):
                self.chunks.append(torch.tensor(
                    data[i * context_length: (i + 1) * context_length], dtype=torch.long
                ))
        logger.info(f"Loaded {len(self.chunks):,} chunks ({len(self.chunks) * context_length / 1e9:.2f}B tokens)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        import torch
        chunk = self.chunks[idx].tolist()
        if random.random() < self.fim_rate:
            chunk = apply_fim_transform(
                chunk, self.fim_prefix_id, self.fim_middle_id, self.fim_suffix_id, self.eos_id
            )
            chunk = chunk[:self.context]
            if len(chunk) < self.context:
                chunk += [self.eos_id] * (self.context - len(chunk))
        ids = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


# ── FlashAttention-2 patch ────────────────────────────────────────────────────

def enable_flash_attention():
    """
    Monkey-patch TreunoAttention to use PyTorch SDPA with Flash backend.
    PyTorch >= 2.2 + Flash-Attention-2 package enables this automatically
    when is_causal=True and dtype is bfloat16.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            logger.info("FlashAttention-2 via torch SDPA enabled.")
    except Exception as e:
        logger.warning(f"FlashAttention-2 not available: {e}. Using standard SDPA.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    import torch
    from transformers import TrainingArguments, Trainer
    from model.config import TreunoConfig

    enable_flash_attention()

    # Phase 1 uses 4096 context
    cfg = TreunoConfig.treuno_125m()
    cfg.context_length = 4096

    from model.transformer import TreunoModel
    model = TreunoModel(cfg).to(torch.bfloat16)
    total_params = model.num_parameters()
    logger.info(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    dataset = PretrainDataset(
        data_dir=args.data_dir,
        context_length=cfg.context_length,
        fim_rate=0.20,
    )
    logger.info(f"Dataset size: {len(dataset):,} chunks")

    # Cosine LR: 2e-4 → 2e-5
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        weight_decay=0.1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=100,
        save_steps=10000,
        save_total_limit=5,
        dataloader_num_workers=4,
        deepspeed=args.deepspeed,
        report_to="mlflow" if args.mlflow else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels":    torch.stack([b["labels"]    for b in batch]),
        },
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    logger.info(f"Phase 1 complete. Checkpoint saved to {args.output_dir}")


def main():
    p = argparse.ArgumentParser(description="Treuno Phase 1: Pretraining")
    p.add_argument("--data-dir",   default="d:/MODEL/data/pretrain")
    p.add_argument("--output-dir", default="d:/MODEL/checkpoints/phase1")
    p.add_argument("--deepspeed",  default="configs/deepspeed_zero2.json")
    p.add_argument("--resume",     default=None, help="Resume from checkpoint path")
    p.add_argument("--mlflow",     action="store_true", help="Enable MLflow logging")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
