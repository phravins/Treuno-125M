# Treuno 125M — Canonical Implementation

Treuno 125M is a coding-specialized language model powered by the RAG framework and a secure **Code Execution Sandbox**.

## Architecture Overview

- **Model**: 125M parameters, 12 layers, GQA, SwiGLU, 8k context.
- **RAG**: Hybrid retrieval (BM25 + FAISS) with real-time web fallback.
- **Sandbox**: Secure multi-language execution (13 languages) with gVisor.
- **Training**: 6-phase pipeline including Pretraining, Context Extension, RAG-SFT, and Execution DPO.

For full technical details, see [TECHNICAL_SPEC.md](file:///d:/MODEL/TECHNICAL_SPEC.md).

---

## Quickstart

### 1. Setup (One-Click)
Run the master setup script to install all dependencies and prepare the environment:
```bash
python scripts/setup_treuno.py
```

### 2. Interactive CLI (Test Mode)
The CLI starts with **HuggingFace weights** (e.g. GPT-2) so you can see the system working immediately while your model trains:
```bash
python inference/cli.py --model-path gpt2
```

### 3. Training
To begin the 100B token pretraining journey:
```bash
python scripts/train_phase1_pretrain.py --data-dir d:/MODEL/data/tokenized --output-dir d:/MODEL/weights/phase1
```

---


*Treuno 125M — Grounding code in reality.*
