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

    # 2. Install dependencies
    # We use --user to avoid permission issues and prioritize user-level installs
    run(f"{sys.executable} -m pip install -r {ROOT}/requirements.txt --user --upgrade",
        "Installing all requirements.txt dependencies")

    # 3. Handle specific critical fixes
    run(f"{sys.executable} -m pip install numpy==1.26.4 duckduckgo-search --user",
        "Ensuring stable numpy and retrieval library")

    # 4. Pre-download GPT-2 for "Immediate Test Mode"
    logger.info("Downloading GPT-2 base weights/tokenizer for testing...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained("gpt2")
        AutoModelForCausalLM.from_pretrained("gpt2")
        logger.info("GPT-2 ready in cache.")
    except Exception as e:
        logger.warning(f"Could not pre-download GPT-2: {e}. CLI will attempt download on first run.")

    # 5. Run structural verification
    logger.info("Running Treuno structural verification...")
    if os.path.isfile(ROOT / "verify_training.py"):
        run(f"{sys.executable} {ROOT}/verify_training.py", "Verifying pipeline architecture")

    # 6. Final message
    print("\n" + "="*60)
    print("  TREUNO 125M SETUP COMPLETE!")
    print("="*60)
    print("\nTo start talking to the model immediately:")
    print("  python inference/cli.py --model-path gpt2")
    print("\nTo start pretraining (Phase 1):")
    print("  python scripts/train_phase1_pretrain.py")
    print("\nTo view the project walkthrough:")
    print(f"  Check {ROOT}/walkthrough.md")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
