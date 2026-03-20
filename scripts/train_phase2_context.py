"""
Treuno 125M — Phase 2: Context Window Extension
=================================================
Extends the context window from 4,096 → 8,192 tokens via RoPE scaling.

Continue training from Phase 1 checkpoint on 5 billion tokens of long-file code
samples (files > 2,000 tokens), with RoPE frequency scaling factor = 2.0.

This allows the model to attend over full function bodies, classes, and
multi-file contexts without losing the knowledge from Phase 1.

Usage:
  deepspeed --num_gpus 4 scripts/train_phase2_context.py \\
      --phase1-checkpoint d:/MODEL/checkpoints/phase1 \\
      --data-dir d:/MODEL/data/longfiles \\
      --output-dir d:/MODEL/checkpoints/phase2
"""

import os, sys, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def patch_rope_for_extension(model, scale_factor: float = 2.0):
    """
    Apply RoPE frequency scaling to extend context without retraining from scratch.
    Scales the inverse frequency by 1/scale_factor, effectively doubling usable range.
    """
    for layer in model.layers:
        rope = layer.attn.rope
        rope.inv_freq = rope.inv_freq / scale_factor
        # Recompute cached sin/cos tables for 8192 context
        import torch
        max_seq = rope.max_seq_len * int(scale_factor)
        t = torch.arange(max_seq, device=rope.inv_freq.device, dtype=rope.inv_freq.dtype)
        freqs = torch.outer(t, rope.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        rope.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        rope.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        rope.max_seq_len = max_seq
    logger.info(f"RoPE scaling factor {scale_factor}x applied. New max_seq={model.layers[0].attn.rope.max_seq_len}")


class LongFileDataset:
    """Files > 2000 tokens, padded/truncated to 8192."""
    def __init__(self, data_dir: str, context_length: int = 8192):
        import numpy as np, torch
        from pathlib import Path
        self.chunks = []
        for path in sorted(Path(data_dir).glob("*.bin")):
            data = np.fromfile(str(path), dtype=np.uint16).astype(np.int64)
            n = len(data) // context_length
            for i in range(n):
                chunk = data[i * context_length: (i + 1) * context_length]
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
        logger.info(f"LongFileDataset: {len(self.chunks):,} × 8192 chunks")

    def __len__(self): return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def train(args):
    import torch
    from transformers import TrainingArguments, Trainer
    from model.config import TreunoConfig
    from model.transformer import TreunoModel

    cfg = TreunoConfig.treuno_125m()
    cfg.context_length = 8192

    # Load Phase 1 checkpoint
    model = TreunoModel(cfg).to(torch.bfloat16)
    if args.phase1_checkpoint:
        ckpt = os.path.join(args.phase1_checkpoint, "model.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
            logger.info(f"Loaded Phase 1 checkpoint from {ckpt}")

    # Apply RoPE 2× scaling
    patch_rope_for_extension(model, scale_factor=2.0)

    dataset = LongFileDataset(args.data_dir, context_length=8192)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,    # 8192 context → smaller batch
        gradient_accumulation_steps=64,
        learning_rate=2e-5,               # No warmup — continue from Phase 1 LR floor
        lr_scheduler_type="constant",
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=3,
        deepspeed=args.deepspeed,
        report_to="mlflow" if args.mlflow else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "labels":    torch.stack([x["labels"]    for x in b]),
        },
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"Phase 2 complete. Context extended to 8192. Saved to {args.output_dir}")


def main():
    p = argparse.ArgumentParser(description="Treuno Phase 2: Context Extension")
    p.add_argument("--phase1-checkpoint", default="d:/MODEL/checkpoints/phase1")
    p.add_argument("--data-dir",          default="d:/MODEL/data/longfiles")
    p.add_argument("--output-dir",        default="d:/MODEL/checkpoints/phase2")
    p.add_argument("--deepspeed",         default="configs/deepspeed_zero2.json")
    p.add_argument("--mlflow",            action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
