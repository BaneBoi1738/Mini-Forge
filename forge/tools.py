"""Agent tools + the guarded executor.

Every tool call - from the model or from the system's own fail-safe - goes through
ToolExecutor.handle(), in this order:

  1. catalogue check  - tool must be in the agent's allowed_tools (agents.yaml)
  2. policy check     - PolicyEngine.evaluate() against policies.yaml, deny by default.
                        Denied -> logged, nothing de-tokenized, nothing written.
  3. validation       - decisions must cite an approved SOP that search_sop actually
                        returned in this run, and the action must be that SOP's action
  4. execute          - de-tokenize ONLY the fields the policy rule lists, write one
                        JSON file per action to outbox/<run_id>/

What goes back to the model is a short status. It never contains de-tokenized values.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from forge import settings
from forge.audit import audit, now_iso
from forge.catalogue import AgentSpec
from forge.phi import TOKEN_RE
from forge.policy import PolicyEngine
from forge.sop_index import SOPIndex
from forge.vault import Vault

CATEGORIES = ["missing_or_invalid_info", "authorization_missing", "timely_filing",
              "medical_necessity", "duplicate_claim", "coverage_terminated", "unreadable", "unknown"]
ROUTE_ACTIONS = ["correct_and_resubmit", "request_retro_auth", "timely_filing_review",
                 "appeal_with_records", "close_no_action", "verify_eligibility"]
TERMINAL_TOOLS = {"route_to_work_queue", "flag_for_human_review"}

TOOL_SCHEMAS = {
    "search_sop": {
        "name": "search_sop",
        "description": "Search the approved SOP library. Describe the denial reason in your own "
                       "words and include the reason code (CARC) if the letter states one. "
                       "Returns the best-matching SOPs with their category and action.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Denial reason, e.g. 'CARC 50 not medically necessary'"}},
            "required": ["query"]},
    },
    "route_to_work_queue": {
        "name": "route_to_work_queue",
        "description": "FINAL step for a decided case. Sends the claim to the work queue for the "
                       "SOP's action. The claim number and date of service are attached automatically.",
        "input_schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ROUTE_ACTIONS},
            "category": {"type": "string", "enum": CATEGORIES[:6]},
            "sop_id": {"type": "string", "description": "The SOP this decision is based on, e.g. SOP-101"},
            "note": {"type": "string", "description": "One or two sentences for the work queue (what to fix / why)."}},
            "required": ["action", "category", "sop_id", "note"]},
    },
    "draft_appeal": {
        "name": "draft_appeal",
        "description": "Draft an appeal letter (only where the SOP says to appeal). You may refer to "
                       "privacy tokens such as <CLAIM_NUMBER_...> verbatim; they are filled in later "
                       "only if policy allows.",
        "input_schema": {"type": "object", "properties": {
            "sop_id": {"type": "string"},
            "appeal_letter": {"type": "string"},
            "records_to_attach": {"type": "array", "items": {"type": "string"}}},
            "required": ["sop_id", "appeal_letter", "records_to_attach"]},
    },
    "flag_for_human_review": {
        "name": "flag_for_human_review",
        "description": "FINAL step when a human must decide: unreadable letter, undeterminable "
                       "reason, an SOP rule requiring review, or a denied tool call.",
        "input_schema": {"type": "object", "properties": {
            "reason": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORIES},
            "recommended_action": {"type": "string", "enum": ROUTE_ACTIONS + ["human_review"]},
            "sop_id": {"type": "string", "description": "SOP the flag is based on, if any"}},
            "required": ["reason", "category", "recommended_action"]},
    },
}


@dataclass
class RunContext:
    run_id: str
    correlation_id: str
    document_id: str
    fields: dict                  # field_type -> token (from tokenizer)
    billed_amount: float | None   # parsed by code from the raw letter (not PHI)
    agent: AgentSpec
    policy: PolicyEngine
    sops: SOPIndex
    vault: Vault
    retrieved: set = field(default_factory=set)       # SOP ids returned by search_sop
    executed: list = field(default_factory=list)      # tool names that actually ran
    decision: dict | None = None                      # the terminal decision
    policy_denials: int = 0
    drafted_appeal: bool = False
    seq: int = 0

    @property
    def done(self) -> bool:
        return self.decision is not None


def detokenize(text: str, allowed: list[str], ctx: RunContext) -> tuple[str, set, set]:
    """Replace tokens whose field type is in `allowed`; redact every other token."""
    released, redacted = set(), set()

    def sub(m):
        hit = ctx.vault.resolve(ctx.correlation_id, m.group(0))
        if hit is None:
            redacted.add("unresolved")
            return "[UNRESOLVED-TOKEN]"
        ftype, value = hit
        if ftype in allowed:
            released.add(ftype)
            return value
        redacted.add(ftype)
        return f"[REDACTED:{ftype}]"
    return TOKEN_RE.sub(sub, text or ""), released, redacted


class ToolExecutor:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def _ids(self):
        c = self.ctx
        return dict(run_id=c.run_id, correlation_id=c.correlation_id, document_id=c.document_id)

    def handle(self, name: str, args: dict, actor: str = "agent") -> dict:
        c = self.ctx
        args = args if isinstance(args, dict) else {}
        audit("tool_requested", **self._ids(), actor=actor, tool=name, args=args)

        # 1. catalogue ---------------------------------------------------------------
        if name not in c.agent.allowed_tools:
            return self._deny(name, actor, f"tool '{name}' is not in agent '{c.agent.name}' catalogue entry")

        # 2. policy (trusted context only) ---------------------------------------------
        sop = c.sops.get(args.get("sop_id", "")) if args.get("sop_id") in c.retrieved else None
        context = {"category": sop.category if sop else None, "billed_amount": c.billed_amount}
        decision = c.policy.evaluate(name, context)
        audit("policy_decision", **self._ids(), actor=actor, tool=name, allowed=decision.allowed,
              reason=decision.reason, context=context, detokenize=decision.detokenize)
        if not decision.allowed:
            c.policy_denials += 1
            return {"status": "denied", "reason": decision.reason}

        # 3. validation ------------------------------------------------------------------
        err = self._validate(name, args, actor)
        if err:
            audit("tool_rejected", **self._ids(), actor=actor, tool=name, error=err)
            return {"status": "error", "reason": err}

        # 4. execute ---------------------------------------------------------------------
        result = getattr(self, f"_run_{name}")(args, decision.detokenize, actor)
        c.executed.append(name)
        return result

    def _deny(self, name, actor, reason):
        self.ctx.policy_denials += 1
        audit("policy_decision", **self._ids(), actor=actor, tool=name, allowed=False, reason=reason)
        return {"status": "denied", "reason": reason}

    def _validate(self, name, args, actor) -> str | None:
        c = self.ctx
        schema = TOOL_SCHEMAS[name]["input_schema"]
        for req in schema.get("required", []):
            if args.get(req) in (None, "", []):
                return f"missing required argument '{req}'"
        for key, prop in schema["properties"].items():
            if "enum" in prop and key in args and args[key] not in prop["enum"]:
                return f"'{key}' must be one of {prop['enum']}"
        if name == "search_sop" or actor == "system":
            return None
        sop_id = args.get("sop_id")
        if sop_id and sop_id not in c.retrieved:
            return f"{sop_id} was not returned by search_sop in this run - search first and cite a retrieved SOP"
        if name == "route_to_work_queue":
            sop = c.sops.get(sop_id)
            if sop.action != args["action"]:
                return f"action '{args['action']}' does not match {sop_id} action '{sop.action}'"
            if args["action"] == "appeal_with_records" and not c.drafted_appeal:
                return f"{sop_id} requires draft_appeal before routing to Appeals"
        return None

    # ---- tool implementations ---------------------------------------------------------
    def _run_search_sop(self, args, _allowed, actor):
        hits = self.ctx.sops.search(str(args["query"]))
        self.ctx.retrieved.update(h["sop_id"] for h in hits)
        audit("tool_call", **self._ids(), actor=actor, tool="search_sop",
              results=[(h["sop_id"], h["score"]) for h in hits])
        return {"status": "ok", "results": hits}

    def _run_route_to_work_queue(self, args, allowed, actor):
        queue = self.ctx.policy.work_queues.get(args["action"], "Human Review")
        out = self._write_outbox("route_to_work_queue", allowed, actor, {
            "queue": queue, "action": args["action"], "category": args["category"],
            "sop_reference": args["sop_id"], "note": args["note"]}, text_fields=["note"])
        self.ctx.decision = {"tool": "route_to_work_queue", "category": args["category"],
                             "action": args["action"], "human_review": False,
                             "sop_reference": args["sop_id"], "actor": actor}
        return {"status": "ok", "queue": queue, "outbox_file": out}

    def _run_draft_appeal(self, args, allowed, actor):
        out = self._write_outbox("draft_appeal", allowed, actor, {
            "queue": self.ctx.policy.work_queues.get("appeal_with_records", "Appeals"),
            "type": "appeal_draft", "sop_reference": args["sop_id"],
            "appeal_letter": args["appeal_letter"],
            "records_to_attach": args.get("records_to_attach", [])}, text_fields=["appeal_letter"])
        self.ctx.drafted_appeal = True
        return {"status": "ok", "outbox_file": out,
                "next": "route_to_work_queue with action appeal_with_records"}

    def _run_flag_for_human_review(self, args, allowed, actor):
        out = self._write_outbox("flag_for_human_review", allowed, actor, {
            "queue": self.ctx.policy.work_queues.get("human_review", "Human Review"),
            "category": args["category"], "recommended_action": args["recommended_action"],
            "sop_reference": args.get("sop_id"), "reason": args["reason"]}, text_fields=["reason"])
        self.ctx.decision = {"tool": "flag_for_human_review", "category": args["category"],
                             "action": args["recommended_action"], "human_review": True,
                             "sop_reference": args.get("sop_id"), "actor": actor}
        return {"status": "ok", "queue": "Human Review", "outbox_file": out}

    def _write_outbox(self, tool, allowed, actor, body: dict, text_fields: list[str]) -> str:
        c = self.ctx
        released, redacted = set(), set()
        # identifiers attached by code: exactly the fields this tool's policy rule allows
        identifiers = {}
        for f in allowed:
            tok = c.fields.get(f)
            val, rel, red = detokenize(tok, allowed, c) if tok else (None, set(), set())
            identifiers[f] = val
            released |= rel
            redacted |= red
        for f in text_fields:
            body[f], rel, red = detokenize(str(body.get(f, "")), allowed, c)
            released |= rel
            redacted |= red

        c.seq += 1
        record = {"correlation_id": c.correlation_id, "run_id": c.run_id,
                  "document_id": c.document_id, "action_type": tool, "actor": actor,
                  "created_at": now_iso(), "identifiers": identifiers, **body}
        path = Path(settings.OUTBOX_DIR) / c.run_id / f"{c.document_id}__{c.seq:02d}_{tool}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        rel_path = str(path.relative_to(settings.DATA_DIR)) if path.is_relative_to(settings.DATA_DIR) else str(path)
        # audit records WHICH fields were released / redacted - never the values
        audit("tool_call", **self._ids(), actor=actor, tool=tool, outbox_file=rel_path)
        audit("output", **self._ids(), actor=actor, tool=tool, outbox_file=rel_path,
              fields_released=sorted(released), fields_redacted=sorted(redacted))
        return rel_path


def tool_schemas_for(agent: AgentSpec) -> list[dict]:
    """The model is only shown tools that the catalogue allows for this agent."""
    return [TOOL_SCHEMAS[t] for t in agent.allowed_tools if t in TOOL_SCHEMAS]


def parse_billed_amount(raw_text: str) -> float | None:
    m = re.search(r"(?i)(?:billed\s+amount|amount\s+billed|total\s+charges?|billed)\s*:?\s*"
                  r"\$\s?([\d,]+(?:\.\d{1,2})?)", raw_text)
    return float(m.group(1).replace(",", "")) if m else None
