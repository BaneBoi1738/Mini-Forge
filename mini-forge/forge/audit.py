"""Append-only JSON-lines audit log.

One line per step. Every line carries run_id + correlation_id so a single letter's
journey (tokenize -> model calls -> policy decisions -> tool calls -> outputs) can be
reconstructed with `grep <correlation_id> logs/audit.jsonl`.

Callers must only pass tokenized text / field names / counts here, never raw PHI.
eval.py scans this file for PHI values to prove that.
"""
import json
from datetime import datetime, timezone

from forge import settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def audit(event: str, *, run_id: str, correlation_id: str | None,
          document_id: str | None = None, **details) -> None:
    append_jsonl(settings.AUDIT_LOG, {
        "ts": now_iso(),
        "run_id": run_id,
        "correlation_id": correlation_id,
        "document_id": document_id,
        "event": event,
        **details,
    })
