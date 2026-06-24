"""Audit logging utilities."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR

LOG_DIR = DATA_DIR / "audit_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rm_copilot")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)


def log_audit_event(event_type: str, payload: dict) -> None:
    """Write an audit log entry for compliance traceability."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    logger.info(json.dumps(entry, default=str))

    log_file = LOG_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
