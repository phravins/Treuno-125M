# Treuno 125M Architecture

## System Diagram

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        TREUNO INFERENCE CALL                    â”‚
â”‚                                                                 â”‚
â”‚  User Query                                                     â”‚
â”‚      â”‚                                                          â”‚
â”‚      â–¼                                                          â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚  â”‚Tokenizer â”‚    â”‚         Modelworks ENGINE               â”‚   â”‚
â”‚  â”‚ BPE 32K  â”‚    â”‚  1. Parse query for APIs/libs            â”‚   â”‚
â”‚  â”‚ + FIM    â”‚    â”‚  2. DuckDuckGo/Serper search (top 5)     â”‚   â”‚
â”‚  â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜    â”‚  3. Chunk + embed (sentence-transformers)â”‚   â”‚
â”‚       â”‚          â”‚  4. FAISS retrieval (top 3 chunks)       â”‚   â”‚
â”‚       â”‚    â”Œâ”€â”€â”€â”€â”€â”¤  5. Inject into prompt as system context â”‚   â”‚
â”‚       â”‚    â”‚     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â”‚       â–¼    â–¼                                                    â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                       â”‚
â”‚  â”‚           TREUNO MODEL (125M)        â”‚                       â”‚
â”‚  â”‚                                      â”‚                       â”‚
â”‚  â”‚  Token Embedding (tied, 32768Ã—768)   â”‚                       â”‚
â”‚  â”‚           â”‚                          â”‚                       â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”‚                       â”‚
â”‚  â”‚  â”‚      TreunoBlock Ã— 12       â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚ RMSNorm (pre-norm)   â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚ GQA Attention        â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚  12 Q heads          â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚   4 KV heads         â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚  RoPE on Q and K     â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚ + Residual           â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚ RMSNorm (pre-norm)   â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚ SwiGLU FFN (3072)    â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â”‚ + Residual           â”‚   â”‚     â”‚                       â”‚
â”‚  â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚     â”‚                       â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â”‚                       â”‚
â”‚  â”‚           â”‚                          â”‚                       â”‚
â”‚  â”‚  Final RMSNorm                       â”‚                       â”‚
â”‚  â”‚  LM Head (weight tied to embedding)  â”‚                       â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                       â”‚
â”‚       â”‚                                                         â”‚
â”‚       â–¼                                                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                           â”‚
â”‚  â”‚       CODE SANDBOX               â”‚                           â”‚
â”‚  â”‚  1. Extract code block           â”‚                           â”‚
â”‚  â”‚  2. subprocess.run (timeout 10s) â”‚                           â”‚
â”‚  â”‚  3. If error â†’ append to prompt  â”‚                           â”‚
â”‚  â”‚  4. Regenerate (up to 3 tries)   â”‚                           â”‚
â”‚  â”‚  5. Return verified output       â”‚                           â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                           â”‚
â”‚       â”‚                                                         â”‚
â”‚  Final Answer (verified code + output)                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Component Responsibilities

| Component | Files | Core Responsibility |
|---|---|---|
| Config | `model/config.py` | Single source of truth for all hyperparams |
| Tokenizer | `model/tokenizer.py` | BPE encoding with FIM special tokens |
| Embedding | `model/embedding.py` | Token embedding + RoPE frequency cache |
| Attention | `model/attention.py` | GQA with 12Q/4KV heads, RoPE, causal mask |
| Transformer | `model/transformer.py` | TreunoBlock stack + tied LM head |
| Retriever | `Modelworks/retriever.py` | Web search, returns ranked snippets |
| Indexer | `Modelworks/indexer.py` | Chunk + embed + FAISS vector store |
| RAG | `Modelworks/rag.py` | Assemble prompt with retrieved context |
| Executor | `sandbox/executor.py` | Subprocess code runner per language |
| Verifier | `sandbox/verifier.py` | Run â†’ check â†’ feedback loop |
| Engine | `inference/engine.py` | Orchestrates all three systems |
| API | `inference/api.py` | FastAPI REST endpoint |
| CLI | `inference/cli.py` | Interactive terminal REPL |
| Pipeline | `scripts/data_pipeline.py` | Corpus collection + quality filters |
| Train | `scripts/train.py` | HuggingFace Trainer (bf16, LoRA-ready) |
| Evaluate | `scripts/evaluate.py` | HumanEval / MBPP benchmarks |

## Key Design Invariants

1. **Tied embeddings** â€” `model.lm_head.weight is model.embed_tokens.weight` always true
2. **GQA ratio** â€” `num_q_heads / num_kv_heads == 3` (12/4), KV tensors are expanded before softmax
3. **Pre-norm** â€” RMSNorm is applied *before* attention and FFN, not after
4. **No bias** â€” All `nn.Linear` layers have `bias=False`
5. **FIM at 50%** â€” Half of training sequences use fill-in-the-middle format
6. **Causal mask** â€” Upper-triangular mask always applied; no bidirectional attention
7. **bfloat16** â€” All parameters and activations in bf16 during training
