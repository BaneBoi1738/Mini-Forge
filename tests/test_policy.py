from __future__ import annotations
import pytest
import yaml

from forge.catalogue import AgentNotApproved, load_agent
from forge.policy import PolicyEngine
from forge.tools import tool_schemas_for


@pytest.fixture
def policy():
    return PolicyEngine()


def test_unknown_tool_is_denied_by_default(policy):
    d = policy.evaluate("delete_everything", {})
    assert not d.allowed and "default deny" in d.reason


@pytest.mark.parametrize("tool", ["submit_to_payer", "write_off"])
def test_explicit_deny_rules(policy, tool):
    assert not policy.evaluate(tool, {"category": "medical_necessity", "billed_amount": 1}).allowed


@pytest.mark.parametrize("category,amount,allowed", [
    ("medical_necessity", 2300, True),
    ("medical_necessity", 5000, True),       # boundary: "above $5,000" is blocked, 5,000 is not
    ("medical_necessity", 5000.01, False),
    ("medical_necessity", 12800, False),
    ("medical_necessity", None, False),      # unknown amount -> deny (fail closed)
    ("missing_or_invalid_info", 100, False),
    (None, 100, False),
])
def test_draft_appeal_conditions(policy, category, amount, allowed):
    assert policy.evaluate("draft_appeal", {"category": category, "billed_amount": amount}).allowed is allowed


def test_detokenize_scopes(policy):
    assert policy.evaluate("route_to_work_queue", {}).detokenize == ["claim_number", "date_of_service"]
    assert policy.evaluate("flag_for_human_review", {}).detokenize == ["claim_number"]
    assert policy.evaluate("search_sop", {}).detokenize == []
    assert "patient_name" not in policy.evaluate("route_to_work_queue", {}).detokenize


def test_policy_file_must_be_deny_by_default(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"default": "allow", "rules": []}))
    with pytest.raises(ValueError):
        PolicyEngine(p)


def test_catalogue_refuses_unlisted_agent():
    with pytest.raises(AgentNotApproved):
        load_agent("rogue_agent")


def test_model_is_only_shown_catalogue_tools():
    names = {t["name"] for t in tool_schemas_for(load_agent("denial_triage"))}
    assert names == {"search_sop", "route_to_work_queue", "draft_appeal", "flag_for_human_review"}
