# Treuno 125M Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        TREUNO INFERENCE CALL                    │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐    ┌──────────────────────────────────────────┐   │
│  │Tokenizer │    │         ANTIGRAVITY ENGINE               │   │
│  │ BPE 32K  │    │  1. Parse query for APIs/libs            │   │
│  │ + FIM    │    │  2. DuckDuckGo/Serper search (top 5)     │   │
│  └────┬─────┘    │  3. Chunk + embed (sentence-transformers)│   │
│       │          │  4. FAISS retrieval (top 3 chunks)       │   │
│       │    ┌─────┤  5. Inject into prompt as system context │   │
│       │    │     └──────────────────────────────────────────┘   │
│       ▼    ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │           TREUNO MODEL (125M)        │                       │
│  │                                      │                       │
│  │  Token Embedding (tied, 32768×768)   │                       │
│  │           │                          │                       │
│  │  ┌────────▼────────────────────┐     │                       │
│  │  │      TreunoBlock × 12       │     │                       │
│  │  │  ┌──────────────────────┐   │     │                       │
│  │  │  │ RMSNorm (pre-norm)   │   │     │                       │
│  │  │  │ GQA Attention        │   │     │                       │
│  │  │  │  12 Q heads          │   │     │                       │
│  │  │  │   4 KV heads         │   │     │                       │
│  │  │  │  RoPE on Q and K     │   │     │                       │
│  │  │  │ + Residual           │   │     │                       │
│  │  │  │ RMSNorm (pre-norm)   │   │     │                       │
│  │  │  │ SwiGLU FFN (3072)    │   │     │                       │
│  │  │  │ + Residual           │   │     │                       │
│  │  │  └──────────────────────┘   │     │                       │
│  │  └─────────────────────────────┘     │                       │
│  │           │                          │                       │
│  │  Final RMSNorm                       │                       │
│  │  LM Head (weight tied to embedding)  │                       │
│  └──────────────────────────────────────┘                       │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────┐                           │
│  │       CODE SANDBOX               │                           │
│  │  1. Extract code block           │                           │
│  │  2. subprocess.run (timeout 10s) │                           │
│  │  3. If error → append to prompt  │                           │
│  │  4. Regenerate (up to 3 tries)   │                           │
│  │  5. Return verified output       │                           │
│  └──────────────────────────────────┘                           │
│       │                                                         │
│  Final Answer (verified code + output)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Files | Core Responsibility |
|---|---|---|
| Config | `model/config.py` | Single source of truth for all hyperparams |
| Tokenizer | `model/tokenizer.py` | BPE encoding with FIM special tokens |
| Embedding | `model/embedding.py` | Token embedding + RoPE frequency cache |
| Attention | `model/attention.py` | GQA with 12Q/4KV heads, RoPE, causal mask |
| Transformer | `model/transformer.py` | TreunoBlock stack + tied LM head |
| Retriever | `antigravity/retriever.py` | Web search, returns ranked snippets |
| Indexer | `antigravity/indexer.py` | Chunk + embed + FAISS vector store |
| RAG | `antigravity/rag.py` | Assemble prompt with retrieved context |
| Executor | `sandbox/executor.py` | Subprocess code runner per language |
| Verifier | `sandbox/verifier.py` | Run → check → feedback loop |
| Engine | `inference/engine.py` | Orchestrates all three systems |
| API | `inference/api.py` | FastAPI REST endpoint |
| CLI | `inference/cli.py` | Interactive terminal REPL |
| Pipeline | `scripts/data_pipeline.py` | Corpus collection + quality filters |
| Train | `scripts/train.py` | HuggingFace Trainer (bf16, LoRA-ready) |
| Evaluate | `scripts/evaluate.py` | HumanEval / MBPP benchmarks |

## Key Design Invariants

1. **Tied embeddings** — `model.lm_head.weight is model.embed_tokens.weight` always true
2. **GQA ratio** — `num_q_heads / num_kv_heads == 3` (12/4), KV tensors are expanded before softmax
3. **Pre-norm** — RMSNorm is applied *before* attention and FFN, not after
4. **No bias** — All `nn.Linear` layers have `bias=False`
5. **FIM at 50%** — Half of training sequences use fill-in-the-middle format
6. **Causal mask** — Upper-triangular mask always applied; no bidirectional attention
7. **bfloat16** — All parameters and activations in bf16 during training
