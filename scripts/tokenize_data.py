"""
Treuno 125M — Data Tokenization Pipeline
=========================================
Converts raw JSONL/Parquet code archives into tokenized binary shards (.bin).
Uses the TreunoTokenizer (BPE with FIM support).

Output format:
  A sequence of uint16 token IDs packed into .bin files.
  Includes FIM (Fill-in-the-Middle) transformations in 20% of the data.

Usage:
  python scripts/tokenize_data.py \\
      --input-dir d:/MODEL/data/quality_filtered \\
      --output-dir d:/MODEL/data/tokenized \\
      --fim-rate 0.2
"""

import os
import glob
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import TreunoTokenizer
from model.config import TreunoConfig


def tokenize_file(args):
    """Worker function for parallel tokenization."""
    input_path, output_path, tokenizer_name, fim_rate = args
    
    tokenizer = TreunoTokenizer.from_pretrained(tokenizer_name)
    all_tokens = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text:
                    continue
                
                # Apply FIM (Fill-in-the-Middle) transformation randomly
                if random.random() < fim_rate:
                    # Logic to split text for FIM
                    # In practice, for code we split at logical boundaries
                    lines = text.splitlines()
                    if len(lines) > 5:
                        split_idx = random.randint(1, len(lines) - 2)
                        prefix = "\n".join(lines[:split_idx])
                        suffix = "\n".join(lines[split_idx:])
                        # We use encode_fim without middle text for pretraining layout
                        tokens = tokenizer.encode_fim(prefix=prefix, suffix=suffix, middle="", add_eos=True)
                    else:
                        tokens = tokenizer.encode(text, add_special_tokens=True)
                else:
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                
                all_tokens.extend(tokens)
            except Exception:
                continue
                
    # Save as uint16 binary file
    if all_tokens:
        arr = np.array(all_tokens, dtype=np.uint16)
        with open(output_path, 'wb') as f:
            f.write(arr.tobytes())
        return len(all_tokens)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Treuno Data Tokenization")
    parser.add_argument("--input-dir", default="d:/MODEL/data/quality_filtered")
    parser.add_argument("--output-dir", default="d:/MODEL/data/tokenized")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--fim-rate", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    tasks = []
    for i, f in enumerate(input_files):
        out_name = os.path.basename(f).replace(".jsonl", f"_{i}.bin")
        tasks.append((f, os.path.join(args.output_dir, out_name), args.tokenizer, args.fim_rate))
        
    print(f"Tokenizing {len(tasks)} files using {args.workers} workers...")
    
    with Pool(args.workers) as p:
        results = list(tqdm(p.imap(tokenize_file, tasks), total=len(tasks)))
        
    total_tokens = sum(results)
    print(f"Finished! Processed {total_tokens:,} tokens into {args.output_dir}")


if __name__ == "__main__":
    main()
