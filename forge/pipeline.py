"""Process one letter end to end: tokenize -> (fail closed?) -> agent -> result record."""
from __future__ import annotations
import hashlib
import uuid
from pathlib import Path

from forge import settings
from forge.agent import run_agent
from forge.audit import audit
from forge.catalogue import AgentSpec
from forge.gateway import BudgetExceeded, ModelGateway, PHIBlocked
from forge.llm_backends import QuotaExhausted
from forge.phi import tokenize
from forge.policy import PolicyEngine
from forge.sop_index import SOPIndex
from forge.tools import RunContext, ToolExecutor, parse_billed_amount
from forge.vault import Vault


def system_flag(executor: ToolExecutor, reason: str, category: str = "unknown"):
    """Fail-safe: the SYSTEM (not the model) routes the letter to human review.
    Still goes through the catalogue + policy check like any other call."""
    executor.handle("flag_for_human_review", {
        "reason": reason, "category": category, "recommended_action": "human_review"}, actor="system")


def process_document(path: Path, *, run_id: str, agent: AgentSpec, policy: PolicyEngine,
                     sops: SOPIndex, vault: Vault, provider=None, model=None,
                     correlation_id: str | None = None) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    document_id = path.stem
    correlation_id = correlation_id or str(uuid.uuid4())
    ids = dict(run_id=run_id, correlation_id=correlation_id, document_id=document_id)
    audit("document_received", **ids, sha256=hashlib.sha256(raw.encode()).hexdigest(), chars=len(raw))

    # 1. tokenize + vault ---------------------------------------------------------------
    tok = tokenize(raw)
    vault.store(correlation_id, document_id, tok.mapping)
    audit("tokenize", **ids, tokens=len(tok.mapping),
          field_types=sorted({t for t, _ in tok.mapping.values()}),
          residual_phi=tok.residual, safe=tok.safe)

    ctx = RunContext(run_id=run_id, correlation_id=correlation_id, document_id=document_id,
                     fields=tok.fields, billed_amount=parse_billed_amount(raw), agent=agent,
                     policy=policy, sops=sops, vault=vault)
    executor = ToolExecutor(ctx)
    audit("facts_extracted", **ids, billed_amount=ctx.billed_amount)

    gateway = None
    status = "completed"
    stop_reason = None
    if not tok.safe:
        # 2a. FAIL CLOSED: PHI may remain -> no model call at all
        status = "fail_closed_phi"
        audit("fail_closed", **ids, reason="possible residual PHI after tokenization",
              details=tok.residual)
        system_flag(executor, "PHI could not be fully removed; letter not sent to the model.")
    else:
        # 2b. agent run through the gateway
        gateway = ModelGateway(agent, run_id=run_id, correlation_id=correlation_id,
                               document_id=document_id, phi_values=vault.values_for(correlation_id),
                               provider=provider, model=model)
        try:
            run_agent(gateway, executor, document_id, tok.text)
        except BudgetExceeded as e:
            status, stop_reason = "budget_exceeded", str(e)
            audit("run_stopped", **ids, reason=str(e))
        except PHIBlocked as e:
            status, stop_reason = "fail_closed_phi_guard", str(e)
            audit("run_stopped", **ids, reason=str(e))
        except QuotaExhausted as e:
            status, stop_reason = "quota_exhausted", str(e)[:200]
            audit("run_stopped", **ids, reason=f"provider quota exhausted: {str(e)[:200]}")
        except Exception as e:  # provider/API error: never leave a letter un-routed
            status, stop_reason = "error", f"{type(e).__name__}: {str(e)[:300]}"
            audit("run_stopped", **ids, reason=stop_reason)
        if not ctx.done:
            if status == "completed":
                status = "no_terminal_action"
            system_flag(executor, f"Agent run ended without a decision ({status}).")

    d = ctx.decision or {}
    result = {
        "document_id": document_id, "correlation_id": correlation_id, "run_id": run_id,
        "status": status, "error": stop_reason,
        "category": d.get("category"), "action": d.get("action"),
        "human_review": d.get("human_review", True), "decided_by": d.get("actor"),
        "sop_reference": d.get("sop_reference"),
        "tool_calls": ctx.executed, "policy_denials": ctx.policy_denials,
        "model_steps": gateway.steps if gateway else 0,
        "tokens": gateway.tokens_used if gateway else 0,
        "cost_usd": round(gateway.cost_usd, 6) if gateway else 0.0,
    }
    audit("run_complete", **ids, **{k: v for k, v in result.items()
                                    if k not in ("document_id", "correlation_id", "run_id")})
    return result
