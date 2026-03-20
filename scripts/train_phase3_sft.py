"""
Treuno 125M — Phase 3: RAG-Aware Supervised Fine-Tuning
=========================================================
Teach the model to read and cite retrieved source documents —
using the EXACT Antigravity prompt template it will see at inference time.

Data: 20,000 instruction examples formatted as:
    [ANTIGRAVITY LIVE CONTEXT — Retrieved for: "{query}"]
    Source: {url}
    ----------------------------------------
    {retrieved_text}
    [END CONTEXT]

    {user_instruction}

    ### Response:
    {code_or_answer}

Expected outcome: model learns to synthesize retrieved context
instead of ignoring the injected passages.

Usage:
  python scripts/train_phase3_sft.py \\
      --phase2-checkpoint d:/MODEL/checkpoints/phase2 \\
      --data-path d:/MODEL/data/rag_sft_20k.jsonl \\
      --output-dir d:/MODEL/checkpoints/phase3
"""

import os, sys, json, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = "### Response:"

RAG_PROMPT_TEMPLATE = """\
[ANTIGRAVITY LIVE CONTEXT — Retrieved for: "{query}"]
Source: {url}
----------------------------------------
{context}
[END CONTEXT]

{instruction}

### Response:
{response}"""


class RAGSFTDataset:
    """
    Dataset for RAG-aware SFT from a JSONL file.

    Each line: {
        "query":       "how to parse JSON in Python",
        "instruction": "Write a Python function to parse a JSON string.",
        "context":     "import json\njson.loads(s)",
        "url":         "https://docs.python.org/3/library/json.html",
        "response":    "```python\nimport json\n...\n```"
    }
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 8192):
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
                    self.examples.append(ex)
                except json.JSONDecodeError:
                    continue
        logger.info(f"RAG SFT Dataset: {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        import torch
        ex = self.examples[idx]
        text = RAG_PROMPT_TEMPLATE.format(
            query=ex.get("query", ex.get("instruction", "")),
            url=ex.get("url", "https://example.com"),
            context=ex.get("context", "No retrieved context available."),
            instruction=ex.get("instruction", ""),
            response=ex.get("response", ""),
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        # Only compute loss on the response portion (after RESPONSE_TEMPLATE)
        labels = input_ids.clone()
        response_start = self._find_response_start(input_ids.tolist())
        labels[:response_start] = -100
        return {"input_ids": input_ids, "labels": labels, "attention_mask": enc["attention_mask"].squeeze()}

    def _find_response_start(self, ids: list) -> int:
        # Find the first token of "### Response:" in the sequence
        marker = self.tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
        for i in range(len(ids) - len(marker)):
            if ids[i:i+len(marker)] == marker:
                return i + len(marker)
        return len(ids) // 2  # Fallback: train on second half


def train(args):
    import torch
    from transformers import AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from model.config import TreunoConfig
    from model.transformer import TreunoModel

    cfg = TreunoConfig.treuno_125m()

    # Load Phase 2 checkpoint
    model = TreunoModel(cfg).to(torch.bfloat16)
    if args.phase2_checkpoint and os.path.exists(os.path.join(args.phase2_checkpoint, "model.pt")):
        model.load_state_dict(torch.load(
            os.path.join(args.phase2_checkpoint, "model.pt"), map_location="cpu"
        ), strict=False)
        logger.info("Loaded Phase 2 checkpoint.")

    # Use GPT-2 tokenizer as stand-in (replace with trained BPE tokenizer in production)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = RAGSFTDataset(args.data_path, tokenizer, max_length=cfg.context_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=3,
        report_to="mlflow" if args.mlflow else "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=cfg.context_length,
        dataset_text_field=None,  # We handle tokenization ourselves
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"Phase 3 (RAG-SFT) complete. Saved to {args.output_dir}")


def main():
    p = argparse.ArgumentParser(description="Treuno Phase 3: RAG-Aware SFT")
    p.add_argument("--phase2-checkpoint", default="d:/MODEL/checkpoints/phase2")
    p.add_argument("--data-path",   default="d:/MODEL/data/rag_sft_20k.jsonl")
    p.add_argument("--output-dir",  default="d:/MODEL/checkpoints/phase3")
    p.add_argument("--mlflow",      action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
