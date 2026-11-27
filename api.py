"""Optional upload endpoint. Upload a letter, get its correlation ID back immediately; processing runs
in the background through the exact same pipeline as run.py.

    uvicorn api:app --port 8000
    curl -F "file=@documents/DOC-001.txt" http://localhost:8000/letters
    curl http://localhost:8000/letters/<correlation_id>     # grep it in logs/audit.jsonl too
"""
from __future__ import annotations
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile

from forge import settings
from forge.catalogue import load_agent
from forge.pipeline import process_document
from forge.policy import PolicyEngine
from forge.sop_index import SOPIndex
from forge.vault import Vault

app = FastAPI(title="Mini-Forge")
AGENT, POLICY, SOPS, VAULT = load_agent("denial_triage"), PolicyEngine(), SOPIndex(), Vault()
INBOX = Path(settings.DATA_DIR) / "inbox"
JOBS: dict[str, dict] = {}   # correlation id -> status (in-memory; fine for a demo)


def _process(correlation_id: str, path: Path):
    run_id = datetime.now(timezone.utc).strftime("api-%Y%m%dT%H%M%SZ-") + correlation_id[:4]
    result = process_document(path, run_id=run_id, agent=AGENT, policy=POLICY, sops=SOPS,
                              vault=VAULT, correlation_id=correlation_id)
    results = Path(settings.RESULTS_DIR) / f"{run_id}.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(result) + "\n")
    JOBS[correlation_id] = {"state": "done", **result}


@app.post("/letters")
async def upload(file: UploadFile, background: BackgroundTasks):
    correlation_id = str(uuid.uuid4())
    name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(file.filename or "letter").stem)[:60] or "letter"
    INBOX.mkdir(parents=True, exist_ok=True)
    path = INBOX / f"{name}__{correlation_id[:8]}.txt"
    path.write_bytes(await file.read())
    JOBS[correlation_id] = {"state": "processing"}
    background.add_task(_process, correlation_id, path)
    return {"correlation_id": correlation_id, "state": "processing"}


@app.get("/letters/{correlation_id}")
def status(correlation_id: str):
    if correlation_id not in JOBS:
        raise HTTPException(404, "unknown correlation id")
    return JOBS[correlation_id]
