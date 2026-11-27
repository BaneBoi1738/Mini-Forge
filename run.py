"""Batch runner - processes every letter unattended. No chat interface.

    python run.py --input documents/
    python run.py --input documents/DOC-010.txt
    python run.py --input documents/ --agent some_other_agent   # refused: not in catalogue
"""
import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from forge import settings
from forge.audit import audit
from forge.catalogue import AgentNotApproved, load_agent
from forge.phi import ner_status
from forge.pipeline import process_document
from forge.policy import PolicyEngine
from forge.sop_index import SOPIndex
from forge.vault import Vault


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mini-Forge batch runner")
    ap.add_argument("--input", required=True, help="folder of .txt letters, or a single letter")
    ap.add_argument("--agent", default="denial_triage", help="agent name from config/agents.yaml")
    ap.add_argument("--provider", help="override LLM_PROVIDER (mock|anthropic|bedrock|openai_compat)")
    ap.add_argument("--model", help="override LLM_MODEL")
    args = ap.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:4]
    try:
        agent = load_agent(args.agent)
    except AgentNotApproved as e:
        audit("agent_refused", run_id=run_id, correlation_id=None, agent=args.agent, reason=str(e))
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    src = Path(args.input)
    files = sorted(src.glob("*.txt")) if src.is_dir() else [src]
    if not files:
        print(f"no .txt letters found in {src}", file=sys.stderr)
        return 1

    provider = args.provider or settings.LLM_PROVIDER
    model = args.model or settings.LLM_MODEL
    policy, sops, vault = PolicyEngine(), SOPIndex(), Vault()
    audit("run_started", run_id=run_id, correlation_id=None, agent=agent.name, provider=provider,
          model=model, documents=len(files), phi_ner=ner_status(),
          sops_indexed=list(sops.sops), sops_skipped=sops.skipped)

    print(f"Mini-Forge {run_id}  agent={agent.name}  provider={provider}  model={model}  "
          f"PHI-NER={ner_status()}")
    print(f"SOPs indexed: {', '.join(sops.sops)}  |  skipped (not approved): "
          f"{', '.join(f'{s} ({st})' for s, st in sops.skipped) or 'none'}\n")
    print(f"{'doc':<9}{'status':<17}{'category':<25}{'action':<22}{'review':<7}{'denied':<7}"
          f"{'steps':<6}{'cost $':>9}  tools")

    results_path = Path(settings.RESULTS_DIR) / f"{run_id}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, f in enumerate(files):
        print(f"   ... {f.stem} ({i + 1}/{len(files)})", file=sys.stderr, flush=True)
        if i and settings.SLEEP_BETWEEN_DOCS:
            time.sleep(settings.SLEEP_BETWEEN_DOCS)
        r = process_document(f, run_id=run_id, agent=agent, policy=policy, sops=sops, vault=vault,
                             provider=provider, model=model)
        results.append(r)
        with open(results_path, "a") as out:
            out.write(json.dumps(r) + "\n")
        print(f"{r['document_id']:<9}{r['status']:<17}{str(r['category']):<25}{str(r['action']):<22}"
              f"{'yes' if r['human_review'] else 'no':<7}{r['policy_denials']:<7}{r['model_steps']:<6}"
              f"{r['cost_usd']:>9.5f}  {' > '.join(r['tool_calls'])}")
        if r.get("error"):
            print(f"         reason: {r['error'][:150]}")
        if r["status"] == "quota_exhausted":
            print("\nProvider quota is used up - stopping the batch (the remaining letters would "
                  "fail the same way).\nSwitch LLM_MODEL / provider in .env, or wait for the quota "
                  "to reset, then re-run.", file=sys.stderr)
            break

    total = sum(r["cost_usd"] for r in results)
    (Path(settings.RESULTS_DIR) / "LATEST").write_text(run_id)
    audit("run_finished", run_id=run_id, correlation_id=None, documents=len(results),
          total_cost_usd=round(total, 6))
    print(f"\n{len(results)} letters processed. Total LLM cost: ${total:.5f}")
    print(f"results: {results_path}\noutbox:  {Path(settings.OUTBOX_DIR) / run_id}\n"
          f"audit:   {settings.AUDIT_LOG}\nNext: python eval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
