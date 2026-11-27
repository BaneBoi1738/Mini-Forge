import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from forge import settings
from forge.catalogue import load_agent
from forge.gateway import BudgetExceeded, ModelGateway, PHIBlocked
from forge.llm_backends import LLMResponse, OpenAICompatBackend
from forge.phi import tokenize
from forge.policy import PolicyEngine
from forge.sop_index import SOPIndex
from forge.tools import RunContext, ToolExecutor, parse_billed_amount
from forge.vault import Vault

ROOT = Path(__file__).resolve().parent.parent


def make_executor(doc_id: str, agent=None):
    raw = (ROOT / "documents" / f"{doc_id}.txt").read_text()
    tok = tokenize(raw)
    cid = str(uuid.uuid4())
    vault = Vault()
    vault.store(cid, doc_id, tok.mapping)
    ctx = RunContext("test-run", cid, doc_id, tok.fields, parse_billed_amount(raw),
                     agent or load_agent("denial_triage"), PolicyEngine(), SOPIndex(), vault)
    return ToolExecutor(ctx), tok


def outbox_files():
    return sorted(Path(settings.OUTBOX_DIR).rglob("*.json"))


def test_route_releases_only_claim_and_dos_and_redacts_name_in_free_text():
    ex, tok = make_executor("DOC-001")
    ex.handle("search_sop", {"query": "CARC 16"})
    name_tok = tok.fields["patient_name"]
    r = ex.handle("route_to_work_queue", {"action": "correct_and_resubmit",
                                          "category": "missing_or_invalid_info", "sop_id": "SOP-101",
                                          "note": f"fix modifier for {name_tok}"})
    assert r["status"] == "ok" and "TEST-CLM" not in json.dumps(r)   # model never sees values
    out = json.loads(outbox_files()[0].read_text())
    assert out["identifiers"] == {"claim_number": "TEST-CLM-2601", "date_of_service": "2026-08-14"}
    assert "Test Patient Alpha" not in json.dumps(out)
    assert "[REDACTED:patient_name]" in out["note"]


def test_denied_call_writes_nothing_and_is_counted():
    ex, _ = make_executor("DOC-005")
    ex.handle("search_sop", {"query": "CARC 50 not medically necessary"})
    r = ex.handle("draft_appeal", {"sop_id": "SOP-104", "appeal_letter": "x", "records_to_attach": ["a"]})
    assert r["status"] == "denied"
    assert ex.ctx.policy_denials == 1
    assert outbox_files() == []


def test_decision_must_cite_a_retrieved_sop_with_matching_action():
    ex, _ = make_executor("DOC-001")
    r = ex.handle("route_to_work_queue", {"action": "correct_and_resubmit", "category": "missing_or_invalid_info",
                                          "sop_id": "SOP-101", "note": "x"})
    assert r["status"] == "error" and "search_sop" in r["reason"]
    ex.handle("search_sop", {"query": "CARC 16"})
    r = ex.handle("route_to_work_queue", {"action": "close_no_action", "category": "missing_or_invalid_info",
                                          "sop_id": "SOP-101", "note": "x"})
    assert r["status"] == "error" and "does not match" in r["reason"]


def test_appeal_route_requires_a_draft_first():
    ex, _ = make_executor("DOC-004")
    ex.handle("search_sop", {"query": "CARC 50"})
    r = ex.handle("route_to_work_queue", {"action": "appeal_with_records", "category": "medical_necessity",
                                          "sop_id": "SOP-104", "note": "x"})
    assert r["status"] == "error"
    ex.handle("draft_appeal", {"sop_id": "SOP-104", "appeal_letter": "x", "records_to_attach": ["a"]})
    r = ex.handle("route_to_work_queue", {"action": "appeal_with_records", "category": "medical_necessity",
                                          "sop_id": "SOP-104", "note": "x"})
    assert r["status"] == "ok"


def test_tool_outside_catalogue_is_denied():
    ex, _ = make_executor("DOC-010")
    assert ex.handle("submit_to_payer", {"status": "paid"})["status"] == "denied"


# ---- gateway ------------------------------------------------------------------------
class FakeBackend:
    def __init__(self, in_tok=100, out_tok=10):
        self.in_tok, self.out_tok = in_tok, out_tok

    def complete(self, system, messages, tools, max_tokens):
        return LLMResponse("ok", [], self.in_tok, self.out_tok, "end_turn")


def gateway(agent=None, backend=None, phi=()):
    return ModelGateway(agent or load_agent("denial_triage"), run_id="t", correlation_id="c",
                        document_id="D", phi_values=list(phi), provider="mock",
                        backend=backend or FakeBackend())


def test_gateway_logs_every_call():
    gateway().chat("s", [{"role": "user", "content": "hi"}], [])
    rec = json.loads(Path(settings.MODEL_CALL_LOG).read_text().splitlines()[0])
    assert rec["correlation_id"] == "c" and rec["usage"]["input_tokens"] == 100 and "request" in rec


def test_gateway_enforces_max_steps():
    g = gateway(agent=replace(load_agent("denial_triage"), max_steps=1))
    g.chat("s", [{"role": "user", "content": "hi"}], [])
    with pytest.raises(BudgetExceeded):
        g.chat("s", [{"role": "user", "content": "hi"}], [])


def test_gateway_enforces_token_budget_after_call():
    g = gateway(backend=FakeBackend(in_tok=25_000))
    with pytest.raises(BudgetExceeded):
        g.chat("s", [{"role": "user", "content": "hi"}], [])


def test_gateway_enforces_cost_budget_before_call():
    g = gateway(agent=replace(load_agent("denial_triage"), max_cost_usd_per_run=0.0))
    g.price = {"input": 3.0, "output": 15.0}
    with pytest.raises(BudgetExceeded):
        g.chat("s", [{"role": "user", "content": "hi"}], [])


def test_gateway_blocks_raw_phi():
    g = gateway(phi=["Test Patient Alpha"])
    with pytest.raises(PHIBlocked):
        g.chat("s", [{"role": "user", "content": "Patient: test patient alpha"}], [])
    assert not Path(settings.MODEL_CALL_LOG).exists()     # nothing was sent / logged as sent


def test_openai_message_conversion():
    msgs = [{"role": "user", "content": "letter"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "search_sop",
                                               "input": {"query": "q"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "{}"}]}]
    out, fns = OpenAICompatBackend.to_openai("sys", msgs, [{"name": "search_sop", "description": "d",
                                                            "input_schema": {"type": "object"}}])
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "search_sop"
    assert out[3]["tool_call_id"] == "1"
    assert fns[0]["function"]["parameters"] == {"type": "object"}


# ---- provider errors: bounded retries, quota stops fast ---------------------------
from forge.llm_backends import QuotaExhausted, RetryableError, _classify


def test_error_classification():
    per_min = "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    per_day = "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    no_tier = "429 Quota exceeded for metric generate_content_free_tier_requests, limit: 0"
    assert isinstance(_classify(Exception(per_min)), RetryableError)
    assert isinstance(_classify(Exception(per_day)), QuotaExhausted)
    assert isinstance(_classify(Exception(no_tier)), QuotaExhausted)


class FlakyBackend:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def complete(self, *a):
        self.calls += 1
        raise self.exc


def test_quota_exhausted_is_not_retried():
    b = FlakyBackend(QuotaExhausted("per day"))
    with pytest.raises(QuotaExhausted):
        gateway(backend=b).chat("s", [{"role": "user", "content": "hi"}], [])
    assert b.calls == 1


def test_rate_limit_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("forge.gateway.time.sleep", lambda s: None)
    b = FlakyBackend(RetryableError("429 per minute"))
    with pytest.raises(RuntimeError):
        gateway(backend=b).chat("s", [{"role": "user", "content": "hi"}], [])
    assert b.calls == 3
