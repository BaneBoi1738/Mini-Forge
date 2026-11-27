"""Deny-by-default action policy, enforced in code before every tool call.

A tool call is allowed only if a rule in config/policies.yaml says `effect: allow`
AND all of its `when` conditions hold. Anything else -> deny.

The context passed in is built by the runner from TRUSTED sources (the category of the
SOP the agent actually retrieved, the billed amount parsed by code from the letter),
not from free text the model wrote.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import yaml

from forge import settings


@dataclass
class Decision:
    allowed: bool
    reason: str
    detokenize: list = field(default_factory=list)


class PolicyEngine:
    def __init__(self, path=None):
        cfg = yaml.safe_load(open(path or settings.CONFIG_DIR / "policies.yaml"))
        if cfg.get("default", "deny") != "deny":
            raise ValueError("policies.yaml must be deny-by-default")
        self.rules = {r["tool"]: r for r in cfg.get("rules", [])}
        self.work_queues: dict = cfg.get("work_queues", {})

    def evaluate(self, tool: str, context: dict) -> Decision:
        rule = self.rules.get(tool)
        if rule is None:
            return Decision(False, f"no rule for '{tool}' (default deny)")
        if rule.get("effect") != "allow":
            return Decision(False, f"rule for '{tool}' is effect: {rule.get('effect')}")
        for key, expected in (rule.get("when") or {}).items():
            ok, why = self._check(key, expected, context)
            if not ok:
                return Decision(False, f"condition failed: {why}")
        return Decision(True, "allowed by rule", list(rule.get("detokenize") or []))

    @staticmethod
    def _check(key: str, expected, ctx: dict) -> tuple[bool, str]:
        if key.endswith("_max"):
            name = key[:-4]
            val = ctx.get(name)
            if val is None:
                return False, f"{name} unknown (required <= {expected})"
            return (val <= expected), f"{name}={val} must be <= {expected}"
        if key.endswith("_min"):
            name = key[:-4]
            val = ctx.get(name)
            if val is None:
                return False, f"{name} unknown (required >= {expected})"
            return (val >= expected), f"{name}={val} must be >= {expected}"
        val = ctx.get(key)
        allowed = expected if isinstance(expected, list) else [expected]
        return (val in allowed), f"{key}={val!r} must be one of {allowed}"
