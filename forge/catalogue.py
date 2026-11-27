"""Pre-approved agent catalogue. Anything not in config/agents.yaml cannot run."""
from __future__ import annotations
from dataclasses import dataclass

import yaml

from forge import settings


class AgentNotApproved(Exception):
    pass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    allowed_tools: tuple[str, ...]
    max_steps: int
    max_tokens_per_run: int
    max_cost_usd_per_run: float


def load_agent(name: str, path=None) -> AgentSpec:
    path = path or settings.CONFIG_DIR / "agents.yaml"
    catalogue = (yaml.safe_load(open(path)) or {}).get("agents", {})
    if name not in catalogue:
        raise AgentNotApproved(
            f"agent '{name}' is not in the approved catalogue ({', '.join(catalogue) or 'empty'})")
    a = catalogue[name]
    return AgentSpec(
        name=name,
        description=a.get("description", ""),
        allowed_tools=tuple(a["allowed_tools"]),
        max_steps=int(a["max_steps"]),
        max_tokens_per_run=int(a["max_tokens_per_run"]),
        max_cost_usd_per_run=float(a["max_cost_usd_per_run"]),
    )
