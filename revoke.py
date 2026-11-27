"""Consent revocation: destroy one letter's vault entries and prove its tokens no longer resolve.

    python revoke.py --document DOC-004
"""
from __future__ import annotations
import argparse
import sys

from forge.audit import audit
from forge.vault import Vault


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True, help="document id, e.g. DOC-004")
    ap.add_argument("--reason", default="patient consent revoked")
    args = ap.parse_args(argv)

    vault = Vault()
    rows = vault.tokens_for_document(args.document)
    if not rows:
        print(f"No vault entries for {args.document}.")
        return 1

    print(f"{args.document}: {len(rows)} token(s) across {len({r[0] for r in rows})} run(s). Before:")
    for cid, token, ftype in rows[:6]:
        print(f"  {token:<30} -> resolves: {vault.resolve(cid, token) is not None}")

    deleted = vault.delete_document(args.document)
    for cid in sorted({r[0] for r in rows}):
        audit("consent_revoked", run_id="revocation", correlation_id=cid, document_id=args.document,
              tokens_destroyed=sum(1 for r in rows if r[0] == cid), reason=args.reason)

    print(f"\nDeleted {deleted} vault row(s). After:")
    still = sum(vault.resolve(cid, token) is not None for cid, token, _ in rows)
    for cid, token, ftype in rows[:6]:
        print(f"  {token:<30} -> resolves: {vault.resolve(cid, token) is not None}")
    print(f"\n{len(rows) - still}/{len(rows)} tokens no longer resolve. "
          f"Audit entry 'consent_revoked' written. Outbox files already delivered to work "
          f"queues are outside the vault and must be purged by those systems.")
    return 0 if still == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
