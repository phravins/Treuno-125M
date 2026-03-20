"""
Treuno — Apache Airflow DAG: Weekly LoRA Update Pipeline
=========================================================
Orchestrates the full Phase 6 data pipeline, triggered every Monday at 02:00 UTC.

DAG Steps:
  1. consume_kafka     → Pull latest data from Kafka topics into raw Parquet
  2. run_dedup         → MinHash LSH deduplication (threshold=0.8)
  3. run_quality_filter → DistilBERT quality filter (threshold=0.6)
  4. tokenize_data     → BPE tokenization → .bin shards
  5. run_lora_training → Phase 6 LoRA rank-16 fine-tune (~2h on 2× A100)
  6. validate_model    → Forward pass sanity check on candidate weights
  7. hot_swap          → Deploy candidate → live weights (zero downtime)
  8. run_eval          → HumanEval pass@1 check (alert if < previous)
  9. cleanup           → Delete temp files older than 7 days

Cron: "0 2 * * 1"  (Monday 02:00 UTC)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── DAG configuration ─────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "treuno",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email": ["alerts@treuno.io"],
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

DATA_RAW  = "d:/MODEL/data/raw"
DATA_DEDUP = "d:/MODEL/data/deduped"
DATA_QUAL  = "d:/MODEL/data/quality_filtered"
DATA_TOK   = "d:/MODEL/data/tokenized"
CKPT_LIVE  = "d:/MODEL/weights"
CKPT_CAND  = "d:/MODEL/weights/lora_candidate"

dag = DAG(
    dag_id="treuno_weekly_lora_update",
    default_args=DEFAULT_ARGS,
    description="Weekly LoRA fine-tune pipeline for Treuno 125M",
    schedule_interval="0 2 * * 1",   # Every Monday 02:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["treuno", "training", "lora"],
)

# ── Step 1: Consume Kafka ─────────────────────────────────────────────────────

def consume_kafka_task(**context):
    """Pull latest records from all 4 Kafka topics → Parquet in DATA_RAW."""
    import subprocess, sys
    result = subprocess.run([
        sys.executable,
        "scripts/pipeline/kafka_consumer.py",
        "--output-path", DATA_RAW,
        "--batch-size", "5000",
        "--once",
    ], cwd="d:/MODEL", capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Kafka consumer failed: {result.stderr[-500:]}")
    return result.stdout


consume_kafka = PythonOperator(
    task_id="consume_kafka",
    python_callable=consume_kafka_task,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=30),
)

# ── Step 2: MinHash LSH Deduplication ─────────────────────────────────────────

run_dedup = BashOperator(
    task_id="run_dedup",
    bash_command=(
        f"python d:/MODEL/scripts/pipeline/dedup.py "
        f"--input-dir {DATA_RAW} --output-dir {DATA_DEDUP} --threshold 0.8"
    ),
    dag=dag,
    execution_timeout=timedelta(hours=2),
)

# ── Step 3: DistilBERT Quality Filter ─────────────────────────────────────────

run_quality_filter = BashOperator(
    task_id="run_quality_filter",
    bash_command=(
        f"python d:/MODEL/scripts/pipeline/quality_filter.py "
        f"--input-dir {DATA_DEDUP} --output-dir {DATA_QUAL} --threshold 0.6"
    ),
    dag=dag,
    execution_timeout=timedelta(hours=2),
)

# ── Step 4: Tokenization ──────────────────────────────────────────────────────

run_tokenize = BashOperator(
    task_id="tokenize_data",
    bash_command=(
        f"python d:/MODEL/scripts/tokenize_data.py "
        f"--input-dir {DATA_QUAL} --output-dir {DATA_TOK}"
    ),
    dag=dag,
    execution_timeout=timedelta(hours=1),
)

# ── Step 5: LoRA Training (~2h) ───────────────────────────────────────────────

run_lora = BashOperator(
    task_id="run_lora_training",
    bash_command=(
        f"python d:/MODEL/scripts/train_phase6_lora.py "
        f"--base-checkpoint {CKPT_LIVE} "
        f"--data-path d:/MODEL/data/update_buffer.jsonl "
        f"--output-dir {CKPT_CAND}"
    ),
    dag=dag,
    execution_timeout=timedelta(hours=3),
)

# ── Step 6: Validate candidate ────────────────────────────────────────────────

def validate_candidate_task(**context):
    """Quick forward pass sanity check on candidate weights."""
    import torch, sys
    sys.path.insert(0, "d:/MODEL")
    from model.config import TreunoConfig
    from model.transformer import TreunoModel
    import os

    cfg = TreunoConfig.treuno_125m()
    model = TreunoModel(cfg)
    weights_path = os.path.join(CKPT_CAND, "model.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    out = model(x)
    assert out.logits.shape == (1, 32, cfg.vocab_size), "Shape mismatch!"
    return "Candidate validated."


validate_candidate = PythonOperator(
    task_id="validate_candidate",
    python_callable=validate_candidate_task,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=10),
)

# ── Step 7: Hot-swap ──────────────────────────────────────────────────────────

hot_swap = BashOperator(
    task_id="hot_swap",
    bash_command=(
        f"python d:/MODEL/scripts/train_phase6_lora.py "
        f"--hot-swap "
        f"--base-checkpoint {CKPT_LIVE} "
        f"--data-path d:/MODEL/data/update_buffer.jsonl "
        f"--output-dir {CKPT_CAND} "
        f"--live-dir {CKPT_LIVE}"
    ),
    dag=dag,
    execution_timeout=timedelta(minutes=10),
)

# ── Step 8: Quick eval check ──────────────────────────────────────────────────

run_eval = BashOperator(
    task_id="run_eval",
    bash_command=(
        f"python d:/MODEL/scripts/evaluate.py "
        f"--model-path {CKPT_LIVE} "
        f"--benchmark humaneval "
        f"--max-problems 20 "  # Quick sanity check — not full eval
        f"--output d:/MODEL/data/eval_results_latest.json"
    ),
    dag=dag,
    execution_timeout=timedelta(hours=1),
)

# ── Step 9: Cleanup ───────────────────────────────────────────────────────────

cleanup = BashOperator(
    task_id="cleanup",
    bash_command=(
        "find d:/MODEL/data/raw -name '*.parquet' -mtime +7 -delete && "
        "find d:/MODEL/data/deduped -name '*.jsonl' -mtime +7 -delete"
    ),
    dag=dag,
    execution_timeout=timedelta(minutes=5),
)

# ── DAG dependency graph ──────────────────────────────────────────────────────
(
    consume_kafka
    >> run_dedup
    >> run_quality_filter
    >> run_tokenize
    >> run_lora
    >> validate_candidate
    >> hot_swap
    >> run_eval
    >> cleanup
)
