# Treuno 125M â€” Training Pipeline

## Overview

Treuno trains in **6 sequential phases** on top of a 125M decoder-only transformer.
Each phase builds on the previous checkpoint.
Total cost estimate: **~$2,000 on A100 spot instances**.

```
Phase 1  Pretraining         100B tokens  ~4 days   4Ã— A100 40GB  DeepSpeed ZeRO-2
Phase 2  Context Extension     5B tokens  ~12 hrs   4Ã— A100 40GB  RoPE scaling
Phase 3  RModel-Aware SFT        20K pairs  ~2 hrs    2Ã— A100 40GB  TRL SFTTrainer
Phase 4  Execution DPO        50K pairs  ~4 hrs    2Ã— A100 40GB  TRL DPOTrainer
Phase 5  Uncertainty DPO      10K pairs  ~1.5 hrs  2Ã— A100 40GB  TRL DPOTrainer
Phase 6  Weekly LoRA Î”Update   variable  ~2 hrs    2Ã— A100 40GB  PEFT LoRA rank-16
```

---

## Phase 1 â€” Pretraining

**Goal:** Learn general code structure, syntax, and semantics across all major languages.

**Data sources:**
| Source | Tokens | Notes |
|---|---|---|
| The Stack v2 | 55B | Deduplicated, license-checked |
| GitHub via RedPajama | 25B | Stars â‰¥ 100, CI passing |
| CodeSearchNet | 10B | Python, JS, Ruby, Go, Java, PHP |
| Stack Overflow | 10B | Accepted answers, text + code |

**FIM training:** 20% of batches reformatted as `<fim_prefix>...<fim_suffix>...<fim_middle>...`

**Config:**
```
Optimizer:       AdamW (Î²1=0.9, Î²2=0.95, Îµ=1e-8, weight_decay=0.1)
Learning rate:   2e-4 â†’ 2e-5 cosine decay
Warmup steps:    2,000
Context length:  4,096 (extended in Phase 2)
Batch size:      512 sequences Ã— 4,096 tokens = 2.1M tokens/step
Gradient clip:   1.0
Precision:       bfloat16
Attention:       FlashAttention-2
Distributed:     DeepSpeed ZeRO-2 across 4Ã— A100 40GB
Checkpointing:   Every 10,000 steps
```

---

## Phase 2 â€” Context Window Extension

**Goal:** Extend context from 4,096 to 8,192 tokens for long-file code understanding.

**Method:** Continue training with RoPE frequency scaling (`rope_scaling_factor=2.0`).
Trains on 5 billion tokens of long files (> 2,000 tokens each) scraped from GitHub.

```
LR:      2e-5 (no warmup â€” continuing from Phase 1 checkpoint)
Context: 8,192
Batches: Long-file only (filtered from The Stack v2)
Steps:   ~12,000
```

---

## Phase 3 â€” RModel-Aware Supervised Fine-Tuning

**Goal:** Teach the model to read retrieved documents, synthesize, and cite sources.

**Data:** 20,000 instruction examples formatted with the exact Modelworks prompt template:
```
[MODELWORKS LIVE CONTEXT â€” Retrieved for: "{query}"]
Source: {url}
----------------------------------------
{retrieved_text}
[END CONTEXT]

{user_instruction}
```

**These examples must use the same template Modelworks injects at inference time.**
This trains the model to use context rather than ignore it.

```
Framework:  TRL SFTTrainer
LR:         1e-4 cosine decay
Epochs:     3
Batch:      32
Max length: 8,192 tokens
```

---

## Phase 4 â€” Execution-Driven DPO

**Goal:** Prefer code that runs over code that crashes â€” automatically, without human labels.

**Data generation (fully automated):**
1. For each of 50,000 coding prompts, sample 2 completions from Phase 3 checkpoint
2. Run both through Model-Execute (Docker+gVisor, 5s timeout)
3. Label: `chosen = passing code`, `rejected = failing code`
4. If both pass or both fail â†’ discard pair

```
Framework:  TRL DPOTrainer
Beta:       0.1    (per spec)
LR:         5e-5
Epochs:     1
Batch:      16
```

---

## Phase 5 â€” Uncertainty DPO

**Goal:** Train the model to say "I'm not sure" rather than hallucinate confidently.

**Data:** 10,000 preference pairs:
- `chosen`:   Response that acknowledges uncertainty and cites sources
- `rejected`: Response that asserts a hallucinated fact confidently

```
Framework:  TRL DPOTrainer
Beta:       0.1
LR:         5e-5
Epochs:     1
Batch:      16
```

---

## Phase 6 â€” Continuous Weekly LoRA Update

**Goal:** Stay current with new library releases, API changes, and code patterns â€” without retraining.

**Trigger:** Apache Airflow cron: `0 2 * * 1` (Monday 02:00 UTC)

**Data pipeline (Apache Kafka â†’ Airflow):**
```
Kafka topics:
  treuno.github.commits
  treuno.stackoverflow.answers
  treuno.changelogs
  treuno.package.docs

Airflow DAG steps:
  1. Consume Kafka â†’ raw JSONL (Parquet on S3/GCS)
  2. MinHash LSH deduplication (datasketch, n-gram=5, LSH threshold=0.8)
  3. DistilBERT quality classifier (score > 0.6 to pass)
  4. Format as SFT examples
  5. Launch LoRA rank-16 fine-tune (~2 hours on 2Ã— A100)
  6. Shadow load â†’ validate perplexity â†’ hot-swap if improved
  7. Zero-downtime deploy via Kubernetes rolling update
```

**LoRA config:**
```
Rank (r):       16    (smaller than Phase 3 â€” optimized for speed)
Alpha:          32
Target modules: q_proj, v_proj
Trainable params: ~500K (0.4% of 125M)
Duration:       ~2 hours on 2Ã— A100 80GB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Training framework | PyTorch 2.2 + HuggingFace Transformers |
| SFT / DPO | TRL (SFTTrainer, DPOTrainer) |
| Distributed training | DeepSpeed ZeRO-2 |
| Attention | FlashAttention-2 |
| Retrieval | sentence-transformers + FAISS + cross-encoder (ms-marco) |
| Execution sandbox | Docker + gVisor (runsc) |
| Response cache | Redis 7 |
| Streaming ingest | Apache Kafka |
| Orchestration | Apache Airflow |
| Deduplication | datasketch (MinHash LSH) |
| Quality filtering | DistilBERT classifier |
| Storage | Parquet on S3 / GCS |
| Inference serving | vLLM + PagedAttention |
| Quantization | AutoGPTQ int4 |
| API | FastAPI (OpenAI-compatible) |
| Deployment | Kubernetes + Helm |
| Experiment tracking | MLflow |
| Metrics | Prometheus + Grafana |
| Eval benchmarks | HumanEval, MBPP, MultiPL-E |
| Primary health metric | Execution pass rate per language |

---

## Estimated Training Cost

| Phase | GPU | Hours | Cost (spot ~$1.5/GPU/hr) |
|---|---|---|---|
| Phase 1 | 4Ã— A100 40GB | 96h | ~$576 |
| Phase 2 | 4Ã— A100 40GB | 12h | ~$72 |
| Phase 3 | 2Ã— A100 40GB | 2h | ~$6 |
| Phase 4 | 2Ã— A100 40GB | 4h | ~$12 |
| Phase 5 | 2Ã— A100 40GB | 1.5h | ~$4.50 |
| Storage + egress | â€” | â€” | ~$100 |
| Buffer / reruns | â€” | â€” | ~$230 |
| **Total** | | | **~$1,000â€“$2,000** |

---

## Evaluation

Continuous evaluation after each phase:
- **HumanEval** (164 Python problems) â€” primary pass@1
- **MBPP** (500 Python problems) â€” broad functional correctness
- **MultiPL-E** (HumanEval ported to 18 languages) â€” cross-language breadth

Primary hallucination health metric: **execution pass rate per language** (tracked in Grafana).
