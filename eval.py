"""Evaluate a run against labels.json and prove no PHI reached the model.

    python eval.py                 # latest run
    python eval.py --run-id run-...
    python eval.py --all-logs      # leak-scan every model request ever logged, not just this run

Prints: category / action / human-review accuracy, tool-sequence match, policy denials,
total LLM cost, and the PHI leak count from scanning every logged model request
(and the audit log) for every PHI value in labels.json. Exit code 1 if any leak.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from forge import settings


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def phi_variants(value: str) -> set[str]:
    """The exact value plus common re-formattings a model might produce (dates)."""
    out = {value.lower()}
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
        out |= {d.strftime("%m/%d/%Y"), d.strftime("%B %d, %Y").lower(),
                d.strftime("%b %d, %Y").lower(), f"{d.month}/{d.day}/{d.year}"}
    except ValueError:
        pass
    return out


def leak_scan(records: list[dict], labels: list[dict], keys=("request", "response")) -> list[dict]:
    """Return one hit per (record, phi field). Hits name the field, never the value."""
    needles = []
    for lab in labels:
        for field, value in lab["phi"].items():
            for v in phi_variants(str(value)):
                needles.append((lab["document_id"], field, v))
    hits = []
    for rec in records:
        blob = json.dumps({k: rec.get(k) for k in keys} if keys else rec).lower()
        for doc, field, v in needles:
            if v in blob:
                hits.append({"record_doc": rec.get("document_id"), "step": rec.get("step"),
                             "phi_of": doc, "field": field, "event": rec.get("event")})
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--all-logs", action="store_true")
    args = ap.parse_args(argv)

    latest = Path(settings.RESULTS_DIR) / "LATEST"
    run_id = args.run_id or (latest.read_text().strip() if latest.exists() else None)
    if not run_id:
        print("No run found. Run `python run.py --input documents/` first.")
        return 1
    results = {r["document_id"]: r for r in read_jsonl(Path(settings.RESULTS_DIR) / f"{run_id}.jsonl")}
    labels = json.loads(settings.LABELS_PATH.read_text())["labels"]

    print(f"Evaluation of {run_id}\n")
    print(f"{'doc':<9}{'cat':<5}{'act':<5}{'rev':<5}{'tools':<7}{'predicted (category / action / review)':<62}expected")
    n = cat_ok = act_ok = rev_ok = tools_ok = 0
    failures = []
    for lab in labels:
        doc = lab["document_id"]
        r = results.get(doc)
        if not r:
            failures.append(f"{doc}: not processed in this run")
            continue
        n += 1
        c = r["category"] == lab["expected_category"]
        a = r["action"] == lab["expected_action"]
        h = bool(r["human_review"]) == lab["expected_human_review"]
        executed = [t for t in r["tool_calls"] if t != "search_sop"]
        t = executed == lab["expected_tool_calls"]
        cat_ok, act_ok, rev_ok, tools_ok = cat_ok + c, act_ok + a, rev_ok + h, tools_ok + t
        mark = lambda ok: "ok" if ok else "X"
        pred = f"{r['category']} / {r['action']} / {'yes' if r['human_review'] else 'no'}"
        exp = f"{lab['expected_category']} / {lab['expected_action']} / {'yes' if lab['expected_human_review'] else 'no'}"
        print(f"{doc:<9}{mark(c):<5}{mark(a):<5}{mark(h):<5}{mark(t):<7}{pred:<62}{exp}")
        if not (c and a and h and t):
            failures.append(f"{doc}: status={r['status']}, decided_by={r['decided_by']}, "
                            f"tools={executed}, expected tools={lab['expected_tool_calls']}")

    audit_rows = [a for a in read_jsonl(settings.AUDIT_LOG) if a.get("run_id") == run_id]
    calls = read_jsonl(settings.MODEL_CALL_LOG)
    run_calls = calls if args.all_logs else [c for c in calls if c.get("run_id") == run_id]
    denials = [a for a in audit_rows if a["event"] == "policy_decision" and not a["allowed"]]
    model_leaks = leak_scan(run_calls, labels)
    audit_leaks = leak_scan(audit_rows, labels, keys=None)
    total_cost = sum(c.get("cost_usd", 0) for c in calls if c.get("run_id") == run_id)
    tokens = sum(c["usage"]["input_tokens"] + c["usage"]["output_tokens"]
                 for c in calls if c.get("run_id") == run_id)
    model = next((f"{c['provider']}/{c['model']}" for c in calls if c.get("run_id") == run_id), "n/a")

    pct = lambda k: f"{k}/{n} = {100 * k / n:.1f}%" if n else "n/a"
    print("\n=== Summary ===")
    print(f"Model                     : {model}")
    print(f"Category accuracy         : {pct(cat_ok)}")
    print(f"Action accuracy           : {pct(act_ok)}")
    print(f"Human-review accuracy     : {pct(rev_ok)}")
    print(f"Tool-sequence match       : {pct(tools_ok)}")
    print(f"Policy denials            : {len(denials)}"
          + "".join(f"\n   - {d['document_id']}: {d['tool']} ({d['reason']})" for d in denials))
    print(f"Fail-closed / system flags: {sum(1 for r in results.values() if r['decided_by'] == 'system')}")
    print(f"Model calls / tokens      : {len(run_calls) if not args.all_logs else sum(1 for c in calls if c.get('run_id') == run_id)} / {tokens}")
    print(f"Total LLM cost            : ${total_cost:.5f}")
    print(f"PHI leak scan             : {len(run_calls)} model requests+responses scanned "
          f"against {sum(len(l['phi']) for l in labels)} PHI values (+ date variants)")
    print(f"PHI LEAKS (model)         : {len(model_leaks)}   {'<-- target is 0' if model_leaks else '(target 0)'}")
    print(f"PHI leaks (audit log)     : {len(audit_leaks)}")
    for hit in (model_leaks + audit_leaks)[:20]:
        print(f"   ! {hit}")
    if not run_calls:
        print("   note: no model calls were logged for this run, so the leak scan is vacuous.")
    if failures:
        print("\nMismatches:")
        for f in failures:
            print(f"   - {f}")

    report = {"run_id": run_id, "n": n, "category_accuracy": cat_ok / n if n else None,
              "action_accuracy": act_ok / n if n else None, "human_review_accuracy": rev_ok / n if n else None,
              "tool_sequence_match": tools_ok / n if n else None, "policy_denials": len(denials),
              "total_cost_usd": round(total_cost, 6), "tokens": tokens, "model": model,
              "model_requests_scanned": len(run_calls), "phi_leaks_model": len(model_leaks),
              "phi_leaks_audit": len(audit_leaks), "failures": failures}
    (Path(settings.RESULTS_DIR) / f"{run_id}.eval.json").write_text(json.dumps(report, indent=2))
    return 1 if (model_leaks or audit_leaks) else 0


if __name__ == "__main__":
    sys.exit(main())
