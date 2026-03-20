"""
Treuno — Kafka Consumer (Data Pipeline)
=======================================
Consumes incoming code data from Apache Kafka topics and writes to
Parquet files on S3/GCS for the Airflow training pipeline.

Topics consumed:
  treuno.github.commits         GitHub commit diffs with context
  treuno.stackoverflow.answers  New accepted SO answers
  treuno.changelogs             Library release notes
  treuno.package.docs           Updated API documentation

Usage:
  python scripts/pipeline/kafka_consumer.py \\
      --bootstrap-servers kafka:9092 \\
      --output-path s3://treuno-data/raw/ \\
      --batch-size 1000
"""

from __future__ import annotations
import os, json, time, argparse, logging
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOPICS = [
    "treuno.github.commits",
    "treuno.stackoverflow.answers",
    "treuno.changelogs",
    "treuno.package.docs",
]

TOPIC_SCHEMAS = {
    "treuno.github.commits": {
        "required": ["repo", "sha", "diff", "language"],
        "text_field": "diff",
    },
    "treuno.stackoverflow.answers": {
        "required": ["question_id", "answer_id", "body", "score"],
        "text_field": "body",
        "filter": lambda r: r.get("is_accepted", False) and r.get("score", 0) > 2,
    },
    "treuno.changelogs": {
        "required": ["package", "version", "changelog_text"],
        "text_field": "changelog_text",
    },
    "treuno.package.docs": {
        "required": ["package", "version", "doc_text"],
        "text_field": "doc_text",
    },
}


class TreunoKafkaConsumer:
    """
    Kafka consumer that reads from all 4 Treuno topics,
    applies basic validation, and batches records to Parquet on S3/GCS.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        output_path: str,
        batch_size: int = 1000,
        group_id: str = "treuno-data-pipeline",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.output_path = output_path
        self.batch_size = batch_size
        self.group_id = group_id
        self._consumer = None
        self._buffers: Dict[str, List[Dict]] = {t: [] for t in TOPICS}
        self._stats = {"consumed": 0, "written": 0, "filtered": 0}

    def start(self, run_forever: bool = True) -> None:
        """Start consuming from all topics."""
        from kafka import KafkaConsumer
        self._consumer = KafkaConsumer(
            *TOPICS,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_poll_records=500,
        )
        logger.info(f"Kafka consumer started. Bootstrap: {self.bootstrap_servers}")
        logger.info(f"Subscribed to: {TOPICS}")

        try:
            while True:
                records = self._consumer.poll(timeout_ms=5000)
                for tp, messages in records.items():
                    for msg in messages:
                        self._process_message(tp.topic, msg.value)
                self._flush_if_full()
                if not run_forever:
                    break
        finally:
            self._flush_all()
            self._consumer.close()
            logger.info(f"Consumer stopped. Stats: {self._stats}")

    def _process_message(self, topic: str, record: Dict[str, Any]) -> None:
        schema = TOPIC_SCHEMAS.get(topic, {})

        # Validate required fields
        for field in schema.get("required", []):
            if field not in record:
                self._stats["filtered"] += 1
                return

        # Apply topic-specific filter
        filter_fn = schema.get("filter")
        if filter_fn and not filter_fn(record):
            self._stats["filtered"] += 1
            return

        # Normalize: add text field for unified downstream processing
        text_field = schema.get("text_field", "text")
        record["text"] = record.get(text_field, "")
        record["source_topic"] = topic
        record["ingested_at"] = time.time()

        self._buffers[topic].append(record)
        self._stats["consumed"] += 1

    def _flush_if_full(self) -> None:
        for topic, buffer in self._buffers.items():
            if len(buffer) >= self.batch_size:
                self._write_parquet(topic, buffer)
                self._buffers[topic] = []

    def _flush_all(self) -> None:
        for topic, buffer in self._buffers.items():
            if buffer:
                self._write_parquet(topic, buffer)
                self._buffers[topic] = []

    def _write_parquet(self, topic: str, records: List[Dict]) -> None:
        """Write records as Parquet to S3/GCS."""
        try:
            import pandas as pd
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            topic_slug = topic.replace(".", "_")
            filename = f"{topic_slug}_{ts}.parquet"

            df = pd.DataFrame(records)
            if self.output_path.startswith("s3://") or self.output_path.startswith("gs://"):
                # Cloud storage (requires s3fs / gcsfs)
                df.to_parquet(f"{self.output_path.rstrip('/')}/{filename}", index=False)
            else:
                # Local storage
                os.makedirs(self.output_path, exist_ok=True)
                local_path = os.path.join(self.output_path, filename)
                df.to_parquet(local_path, index=False)
                filename = local_path

            self._stats["written"] += len(records)
            logger.info(f"Wrote {len(records)} records → {filename}")
        except Exception as e:
            logger.error(f"Failed to write Parquet for topic {topic}: {e}")


def main():
    p = argparse.ArgumentParser(description="Treuno Kafka Data Consumer")
    p.add_argument("--bootstrap-servers", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"))
    p.add_argument("--output-path",       default=os.environ.get("TREUNO_DATA_PATH", "d:/MODEL/data/raw"))
    p.add_argument("--batch-size",        type=int, default=1000)
    p.add_argument("--once",              action="store_true", help="Run one poll cycle and exit")
    args = p.parse_args()

    consumer = TreunoKafkaConsumer(
        bootstrap_servers=args.bootstrap_servers,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    consumer.start(run_forever=not args.once)


if __name__ == "__main__":
    main()
