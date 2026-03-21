"""
Treuno 125M â€” FastAPI REST Server
Exposes the full Treuno pipeline (retrieval + model + sandbox) as a REST API.

Endpoints:
    GET  /health                 â†’ liveness check
    POST /generate               â†’ generate code with verification
    POST /generate/fim           â†’ fill-in-the-middle completion
    GET  /info                   â†’ model config info

Usage:
    python inference/api.py
    # or: uvicorn inference.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Optional, List

# Ensure the project root is always on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Treuno 125M",
    description="Coding-specialized LM with real-time retrieval and code verification.",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ Lazy engine init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        import os
        from inference.engine import TreunoEngine
        model_path = os.environ.get("TREUNO_MODEL_PATH", "d:/MODEL/weights")
        _engine = TreunoEngine.from_pretrained(
            model_path,
            use_retrieval=os.environ.get("TREUNO_NO_RETRIEVAL", "0") != "1",
            use_sandbox=os.environ.get("TREUNO_NO_SANDBOX", "0") != "1",
        )
        logger.info(f"Engine loaded from {model_path}")
    return _engine


# â”€â”€ Request / Response schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Coding question or instruction", min_length=1)
    language: Optional[str] = Field(None, description="Target language, e.g. 'python'")
    max_new_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    use_retrieval: bool = Field(True, description="Use Modelworks web retrieval")
    use_sandbox: bool = Field(True, description="Verify code in sandbox")
    verify_retries: int = Field(3, ge=1, le=5)


class FIMRequest(BaseModel):
    prefix: str = Field(..., description="Code before the completion hole")
    suffix: str = Field(..., description="Code after the completion hole")
    language: Optional[str] = Field("python")
    max_new_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.1, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    text: str
    code: Optional[str]
    language: Optional[str]
    verified: bool
    stdout: str
    retrieved_sources: List[str]
    num_attempts: int


class FIMResponse(BaseModel):
    middle: str
    language: Optional[str]


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health")
async def health():
    return {"status": "ok", "model": "treuno-125m", "version": "0.1.0"}


@app.get("/info")
async def info():
    from model.config import TreunoConfig
    cfg = TreunoConfig.treuno_125m()
    return {
        "model": cfg.model_name,
        "version": cfg.model_version,
        "architecture": {
            "layers": cfg.num_layers,
            "hidden_size": cfg.hidden_size,
            "ffn_size": cfg.ffn_size,
            "num_q_heads": cfg.num_q_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "context_length": cfg.context_length,
            "vocab_size": cfg.vocab_size,
            "tied_embeddings": cfg.tie_embeddings,
        }
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        engine = get_engine()
        result = engine.generate(
            prompt=req.prompt,
            language=req.language,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            use_retrieval=req.use_retrieval,
            use_sandbox=req.use_sandbox,
            verify_retries=req.verify_retries,
        )
        return GenerateResponse(
            text=result.text,
            code=result.code,
            language=result.language,
            verified=result.verified,
            stdout=result.stdout,
            retrieved_sources=result.retrieved_sources,
            num_attempts=result.verification.attempt if result.verification else 1,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/fim", response_model=FIMResponse)
async def generate_fim(req: FIMRequest):
    """Fill-in-the-Middle: complete the code between prefix and suffix."""
    try:
        engine = get_engine()
        # Build FIM prompt using tokenizer
        fim_ids = engine.tokenizer.encode_fim(
            prefix=req.prefix,
            suffix=req.suffix,
            middle="",
            add_eos=False,
        )
        import torch
        input_ids = torch.tensor([fim_ids], device=engine.device)
        output_ids = engine.model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            eos_token_id=engine.tokenizer.eos_token_id,
        )
        new_ids = output_ids[0][len(fim_ids):]
        middle = engine.tokenizer.decode(new_ids.tolist())
        return FIMResponse(middle=middle, language=req.language)
    except Exception as e:
        logger.error(f"FIM error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Pass the app object directly â€” not "inference.api:app" â€” so uvicorn
    # doesn't try to re-import the module from scratch with an empty sys.path.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
