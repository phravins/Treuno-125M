# Treuno 125M — Full Technical Specification

Treuno 125M is a high-efficiency decoder-only Transformer designed for real-time code generation and verification.

## Model Architecture

- **Parameters**: 125M (122.8M non-embedding)
- **Layers**: 12
- **Hidden Dimension**: 768
- **FFN Dimension**: 3072 (using SwiGLU: `W1·x ⊙ SiLU(W2·x) · W3`)
- **Heads**: 12 Query / 4 Key-Value (Grouped-Query Attention)
- **Context Window**: 8192 tokens
- **Positional Encoding**: Rotary Positional Embeddings (RoPE)
- **Normalization**: RMSNorm (epsilon 1e-5, pre-norm)
- **Vocabulary**: 32,768 (BPE-based with special FIM tokens)
- **Tied Embeddings**: Input and output weights are shared (saves 25M params)
- **Precision**: Trained in bfloat16, quantized to GPTQ int4 (~300MB footprint)

---

## RAG System

The model's knowledge is augmented by a real-time retrieval system consisting of:

1.  **Hybrid Retriever**: 
    - **Keyword**: BM25 index of official documentation.
    - **Dense**: FAISS IVF-PQ index of GitHub/arXiv snippets.
    - **Web**: Real-time DuckDuckGo/Serper.dev fallback.
2.  **Cross-Encoder Reranker**: Re-ranks top-10 candidates; injects top-3 into prompt.
3.  **Semantic Cache**: Redis 7 backend stores query-response pairs (cosine similarity threshold 0.92).
4.  **Confidence Scoring**: Model output is verified by a 0.75 threshold scorer before generation begins.

---

## Code Execution Sandbox

- **Execution**: Runs in a secure gVisor-hardened Docker container.
- **Languages**: 13 (Python, JavaScript, Go, Rust, C++, Java, Swift, Kotlin, Bash, Ruby, PHP, C, TypeScript).
- **Control**: 5-second CPU limit, 512MB RAM limit, no network access.
- **Verification**: If code fails unit tests, it enters a **Self-Correction Loop** (max 3 retries) with error logs fed back to the model.

---

## Training & Data Pipeline

### 6-Phase Training
1.  **Phase 1: Pretraining**: 100B tokens (GitHub / The Stack v2).
2.  **Phase 2: Context Extension**: 4k → 8k context using RoPE scaling.
3.  **Phase 3: RAG-Aware SFT**: Taught to read and cite retrieved context.
4.  **Phase 5: Execution DPO**: Preference tuning using sandbox results.
5.  **Phase 5: Uncertainty DPO**: Penalizes hallucinations.
6.  **Phase 6: Weekly LoRA**: Rank-16 LoRA hot-swapping for continuous learning.

### Automated Ingest
- **Kafka**: Real-time event streaming from GitHub/npm webhooks.
- **Deduplication**: MinHash LSH (datasketch) at 0.8 Jaccard similarity.
- **Quality Filtering**: DistilBERT-based classifier (threshold 0.6).
- **Orchestration**: Apache Airflow DAG.

---

## Operations

- **Serving**: vLLM with PagedAttention and AutoGPTQ.
- **API**: FastAPI providing OpenAI-compatible `/v1/completions`.
- **Metrics**: Prometheus (execution pass rate, latency) + Grafana.
- **CLI**: Interactive REPL with syntax highlighting and retrieve/sandbox toggles.

*Documentation version: 1.0.0 (Treuno 125M Canonical)*
