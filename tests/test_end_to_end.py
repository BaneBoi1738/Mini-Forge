"""Full batch in mock mode + the eval script, including a check that the leak scanner works."""
from __future__ import annotations
import json
from pathlib import Path

import eval as evaluator
import run
from forge import settings
from forge.vault import Vault

ROOT = Path(__file__).resolve().parent.parent


def test_full_batch_mock_run_scores_and_leaks():
    assert run.main(["--input", str(ROOT / "documents"), "--provider", "mock"]) == 0
    assert evaluator.main([]) == 0
    run_id = (Path(settings.RESULTS_DIR) / "LATEST").read_text()
    report = json.loads((Path(settings.RESULTS_DIR) / f"{run_id}.eval.json").read_text())
    assert report["phi_leaks_model"] == 0 and report["phi_leaks_audit"] == 0
    assert report["model_requests_scanned"] > 0
    assert report["category_accuracy"] == 1.0 and report["human_review_accuracy"] == 1.0
    # one outbox file per action
    assert len(list(Path(settings.OUTBOX_DIR).rglob("*.json"))) == 14


def test_every_audit_line_has_a_correlation_id():
    run.main(["--input", str(ROOT / "documents" / "DOC-001.txt"), "--provider", "mock"])
    rows = [json.loads(l) for l in Path(settings.AUDIT_LOG).read_text().splitlines()]
    per_doc = [r for r in rows if r["document_id"]]
    assert per_doc and all(r["correlation_id"] for r in per_doc)
    events = {r["event"] for r in per_doc}
    assert {"tokenize", "model_call", "policy_decision", "tool_call", "output", "run_complete"} <= events


def test_leak_scanner_actually_detects_a_leak():
    labels = json.loads((ROOT / "labels.json").read_text())["labels"]
    fake = [{"document_id": "DOC-001", "request": {"messages": ["Patient TEST PATIENT ALPHA, dob 02/11/1962"]}}]
    hits = evaluator.leak_scan(fake, labels)
    assert {h["field"] for h in hits} == {"patient_name", "date_of_birth"}


def test_unapproved_agent_is_refused():
    assert run.main(["--input", str(ROOT / "documents"), "--agent", "rogue"]) == 2


def test_consent_revocation():
    import revoke
    run.main(["--input", str(ROOT / "documents" / "DOC-004.txt"), "--provider", "mock"])
    v = Vault()
    rows = v.tokens_for_document("DOC-004")
    assert rows and all(v.resolve(c, t) for c, t, _ in rows)
    assert revoke.main(["--document", "DOC-004"]) == 0
    assert all(v.resolve(c, t) is None for c, t, _ in rows)
