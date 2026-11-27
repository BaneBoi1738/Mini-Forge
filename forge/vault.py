"""Local token vault (SQLite).

token -> real value mapping lives ONLY here. Rows are scoped to a correlation_id
(one letter in one run), so revoking consent for a letter = deleting its rows.
"""
from __future__ import annotations
import sqlite3
from contextlib import closing

from forge import settings
from forge.audit import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS phi_tokens (
    correlation_id TEXT NOT NULL,
    document_id    TEXT NOT NULL,
    token          TEXT NOT NULL,
    field_type     TEXT NOT NULL,
    value          TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (correlation_id, token)
);
CREATE INDEX IF NOT EXISTS idx_phi_doc ON phi_tokens(document_id);
"""


class Vault:
    def __init__(self, path=None):
        self.path = path or settings.VAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def store(self, correlation_id: str, document_id: str, mapping: dict[str, tuple[str, str]]) -> None:
        """mapping: token -> (field_type, value)"""
        rows = [(correlation_id, document_id, tok, ftype, val, now_iso())
                for tok, (ftype, val) in mapping.items()]
        with closing(self._conn()) as c, c:
            c.executemany("INSERT OR REPLACE INTO phi_tokens VALUES (?,?,?,?,?,?)", rows)

    def resolve(self, correlation_id: str, token: str) -> tuple[str, str] | None:
        """Return (field_type, value) or None if the token is unknown / revoked."""
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT field_type, value FROM phi_tokens WHERE correlation_id=? AND token=?",
                (correlation_id, token)).fetchone()
        return tuple(row) if row else None

    def values_for(self, correlation_id: str) -> list[str]:
        with closing(self._conn()) as c:
            return [r[0] for r in c.execute(
                "SELECT value FROM phi_tokens WHERE correlation_id=?", (correlation_id,))]

    def tokens_for_document(self, document_id: str) -> list[tuple[str, str, str]]:
        """(correlation_id, token, field_type) for every vault row of a document."""
        with closing(self._conn()) as c:
            return list(c.execute(
                "SELECT correlation_id, token, field_type FROM phi_tokens WHERE document_id=?",
                (document_id,)))

    def delete_document(self, document_id: str) -> int:
        with closing(self._conn()) as c, c:
            cur = c.execute("DELETE FROM phi_tokens WHERE document_id=?", (document_id,))
            return cur.rowcount
