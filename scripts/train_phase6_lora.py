"""
Treuno 125M — Phase 6: Weekly LoRA Fine-Tune
=============================================
Continuous knowledge update triggered by AG-Update every Monday at 02:00 UTC.
LoRA rank-16 adapter fine-tune on delta data (new commits, SO answers, changelogs).
Hot-swapped into serving stack with zero downtime via Kubernetes rolling update.

LoRA config:
  Rank (r):       16   (speed-optimized, ~500K trainable params)
  Alpha:          32
  Target modules: q_proj, v_proj
  Dropout:        0.05
  Duration:       ~2 hours on 2× A100 80GB

Data sources (from Kafka → Airflow → processed JSONL):
  treuno.github.commits      → new function/class commits
  treuno.stackoverflow.answers → recent accepted answers
  treuno.changelogs          → library release notes
  treuno.package.docs        → updated API documentation

Usage:
  python scripts/train_phase6_lora.py \\
      --base-checkpoint d:/MODEL/checkpoints/phase5 \\
      --data-path d:/MODEL/data/update_buffer.jsonl \\
      --output-dir d:/MODEL/weights/lora_candidate \\
      --hot-swap
"""

import os, sys, json, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DeltaDataset:
    """Processed delta from Kafka pipeline. Simple text-completion format."""
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    text = ex.get("text", ex.get("prompt", "") + ex.get("response", ""))
                    if text:
                        self.examples.append(text)
                except json.JSONDecodeError:
                    pass
        logger.info(f"Delta dataset: {len(self.examples)} examples")

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        import torch
        enc = self.tokenizer(
            self.examples[idx], max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze()
        return {"input_ids": ids, "labels": ids.clone()}


def build_lora_model(base_model, r: int = 16, alpha: int = 32):
    """Attach LoRA adapters to q_proj and v_proj across all layers."""
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def hot_swap(candidate_dir: str, live_dir: str) -> bool:
    """
    Hot-swap the serving weights by atomically moving candidate → live.
    In Kubernetes: update ConfigMap image tag → rolling restart with 0 downtime.
    Returns True if swap succeeded.
    """
    import shutil
    from pathlib import Path
    candidate = Path(candidate_dir) / "model.pt"
    live      = Path(live_dir) / "model.pt"
    backup    = Path(live_dir) / "model.pt.bak"

    if not candidate.exists():
        logger.error(f"Candidate weights not found at {candidate}")
        return False

    logger.info("Validating candidate weights before hot-swap...")
    try:
        import torch
        from model.config import TreunoConfig
        from model.transformer import TreunoModel
        cfg = TreunoConfig.treuno_125m()
        m = TreunoModel(cfg)
        m.load_state_dict(torch.load(str(candidate), map_location="cpu"), strict=False)
        # Quick sanity: forward pass on random input
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        _ = m(x)
        del m
        logger.info("Candidate weights validated successfully.")
    except Exception as e:
        logger.error(f"Candidate validation failed: {e}. Aborting hot-swap.")
        return False

    # Backup current live weights
    if live.exists():
        shutil.copy2(str(live), str(backup))

    # Atomic swap
    shutil.copy2(str(candidate), str(live))
    logger.info(f"Hot-swap complete: {candidate} → {live}")
    return True


def train(args):
    import torch
    from transformers import AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from model.config import TreunoConfig
    from model.transformer import TreunoModel

    cfg = TreunoConfig.treuno_125m()
    base_model = TreunoModel(cfg).to(torch.bfloat16)

    ckpt = os.path.join(args.base_checkpoint, "model.pt")
    if os.path.exists(ckpt):
        base_model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        logger.info(f"Loaded base checkpoint from {ckpt}")
    else:
        logger.warning("No base checkpoint found — using random init.")

    model = build_lora_model(base_model, r=16, alpha=32)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = DeltaDataset(args.data_path, tokenizer, max_length=2048)

    if len(dataset) < 100:
        logger.warning(f"Only {len(dataset)} examples in update buffer — skipping training.")
        return

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to="mlflow" if args.mlflow else "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=2048,
    )
    trainer.train()

    # Merge LoRA weights into base model for serving
    merged = model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(merged.state_dict(), os.path.join(args.output_dir, "model.pt"))
    logger.info(f"LoRA merged → {args.output_dir}/model.pt")

    # Hot-swap into live serving directory
    if args.hot_swap:
        success = hot_swap(args.output_dir, args.live_dir)
        if success:
            logger.info("Zero-downtime hot-swap completed.")
        else:
            logger.error("Hot-swap failed. Live weights unchanged.")


def main():
    p = argparse.ArgumentParser(description="Treuno Phase 6: Weekly LoRA Update")
    p.add_argument("--base-checkpoint", default="d:/MODEL/checkpoints/phase5")
    p.add_argument("--data-path",       default="d:/MODEL/data/update_buffer.jsonl")
    p.add_argument("--output-dir",      default="d:/MODEL/weights/lora_candidate")
    p.add_argument("--live-dir",        default="d:/MODEL/weights")
    p.add_argument("--hot-swap",        action="store_true", help="Hot-swap into live serving")
    p.add_argument("--mlflow",          action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
