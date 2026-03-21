"""
Treuno 125M — One-Click Setup & Verification
============================================
The master script that makes Treuno "correctly working" by:
  1. Installing all Python dependencies
  2. Creating the file structure (data, weights, etc.)
  3. Pre-downloading GPT-2 weights/tokenizer for "Immediate Test Mode"
  4. Verifying every core module (Antigravity, Sandbox, AI Engine)

Usage:
  python scripts/setup_treuno.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging to both console and file
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("treuno-setup")

ROOT = Path("d:/MODEL")
SUBDIRS = [
    "data/raw", "data/deduped", "data/quality_filtered", "data/tokenized",
    "weights/gpt2", "weights/lora_candidate", "configs", "scripts/pipeline",
    "inference", "monitoring", "helm", "sandbox", "antigravity", "model"
]

def run(cmd, desc):
    logger.info(f"--- {desc} ---")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {desc} ({e})")
        return False

def main():
    logger.info("Starting Treuno 125M Master Setup...")

    # 1. Create directories
    logger.info("Creating directory structure...")
    for sd in SUBDIRS:
        (ROOT / sd).mkdir(parents=True, exist_ok=True)

    # 2. Define dependency groups
    core_deps = [
        "torch", "transformers", "datasets", "tokenizers", "rich", 
        "requests", "duckduckgo-search", "beautifulsoup4", "lxml", 
        "sentence-transformers", "faiss-cpu", "rank-bm25", "numpy",
        "fastapi", "uvicorn", "pydantic", "tqdm", "tenacity",
        "sentencepiece", "tiktoken"
    ]
    
    # Set HuggingFace home to D: drive to avoid C: drive disk space issues
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    logger.info(f"Redirecting HuggingFace cache to {os.environ['HF_HOME']}")

    # 3. Install core dependencies first
    logger.info("Installing CORE dependencies (Inference/CLI)...")
    core_cmd = f"{sys.executable} -m pip install {' '.join(core_deps)} --user --upgrade"
    if not run(core_cmd, "Installing Core ML & Inference dependencies"):
        logger.error("Core dependency installation failed! CLI might not work.")
    else:
        logger.info("Core dependencies installed successfully.")

    # 4. Try installing full requirements as "Advanced"
    logger.info("Attempting to install ADVANCED dependencies (Training/Quantization)...")
    logger.info("Note: these may fail on Windows without specific build tools (deepspeed, flash-attn).")
    run(f"{sys.executable} -m pip install -r {ROOT}/requirements.txt --user --upgrade",
        "Installing full requirements.txt (Optional/Advanced)")

    # 5. Handle specific critical fixes
    run(f"{sys.executable} -m pip install numpy==1.26.4 duckduckgo-search py-cpuinfo --user",
        "Ensuring stable numpy, retrieval, and cpuinfo")

    # 6. Pre-download GPT-2 for "Immediate Test Mode"
    logger.info("Checking/Downloading GPT-2 base weights/tokenizer for testing...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Set a reasonable timeout for the download meta-check
        os.environ["TRANSFORMERS_OFFLINE"] = "0" 
        logger.info("Fetching GPT-2...")
        AutoTokenizer.from_pretrained("gpt2")
        AutoModelForCausalLM.from_pretrained("gpt2")
        logger.info("GPT-2 ready in cache.")
    except KeyboardInterrupt:
        logger.warning("GPT-2 download interrupted by user.")
    except Exception as e:
        logger.warning(f"Could not pre-download GPT-2: {e}. CLI will attempt download on first run.")

    # 7. Run structural verification
    logger.info("Running Treuno structural verification...")
    if os.path.isfile(ROOT / "verify_training.py"):
        try:
            # Add current dir to path so it can find 'model', etc.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, str(ROOT / "verify_training.py")], 
                           env=env, check=False)
        except Exception as e:
            logger.warning(f"Verification script failed: {e}")

    # 8. Final message
    print("\n" + "="*60)
    print("  TREUNO 125M SETUP COMPLETE (with warnings/skips if any)")
    print("="*60)
    print("\nTo start talking to the model immediately:")
    print("  python inference/cli.py --model-path gpt2")
    print("\nTo start pretraining (Phase 1) - REQUIRES FULL DEPS:")
    print("  python scripts/train_phase1_pretrain.py")
    print("\nTo view the project walkthrough:")
    print(f"  Check {ROOT}/walkthrough.md")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
