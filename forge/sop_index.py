"""SOP library search (BM25 keyword search, no external services).

Only SOPs whose front matter says `status: approved` are indexed. Drafts (e.g. SOP-110,
the "write off anything under $1,000" proposal) are skipped at index time, so the agent
cannot retrieve them no matter what it searches for.

Why keyword search and not embeddings: 7 short documents, the vocabulary of denial
letters overlaps heavily with the SOP wording ("authorization number on file",
"documentation does not support"), and it is deterministic and explainable.
A reason code in the query (e.g. "16", "CARC 197") gives the matching SOP a strong boost.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from forge import settings

STOP = set("a an the and or of to for in on is are was were be by with as at this that it "
           "its from not no any if then than these those has have had do does".split())


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


@dataclass
class SOP:
    sop_id: str
    title: str
    status: str
    category: str
    action: str | None
    carc: str | None
    body: str
    path: str


def parse_sop(path: Path) -> SOP:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if m:
        meta, body = yaml.safe_load(m.group(1)) or {}, m.group(2)
    sop_id = re.match(r"(SOP-\d+)", path.name).group(1)
    category = re.sub(r"^SOP-\d+_", "", path.stem).lower()
    title = next((l.lstrip("# ").strip() for l in body.splitlines() if l.startswith("#")), sop_id)
    action = re.search(r"^Action:\s*(\w+)", body, re.M)
    carc = re.search(r"CARC\s+(\d+)", title)
    return SOP(sop_id, title, str(meta.get("status", "unknown")).lower(), category,
               action.group(1) if action else None, carc.group(1) if carc else None,
               body.strip(), str(path))


class SOPIndex:
    k1, b = 1.5, 0.75

    def __init__(self, sop_dir=None):
        self.sops: dict[str, SOP] = {}
        self.skipped: list[tuple[str, str]] = []   # (sop_id, status) not indexed
        for p in sorted(Path(sop_dir or settings.SOP_DIR).glob("*.md")):
            sop = parse_sop(p)
            if sop.status == "approved":
                self.sops[sop.sop_id] = sop
            else:
                self.skipped.append((sop.sop_id, sop.status))
        self._docs = {sid: Counter(_terms(s.title + " " + s.body)) for sid, s in self.sops.items()}
        n = len(self._docs) or 1
        self._avgdl = sum(sum(c.values()) for c in self._docs.values()) / n
        df = Counter(t for c in self._docs.values() for t in c)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _terms(query)
        scores = {}
        for sid, tf in self._docs.items():
            dl = sum(tf.values())
            s = sum(self._idf.get(t, 0) * tf[t] * (self.k1 + 1) /
                    (tf[t] + self.k1 * (1 - self.b + self.b * dl / self._avgdl)) for t in q if t in tf)
            carc = self.sops[sid].carc
            if carc and re.search(rf"(?<!\d){carc}(?!\d)", query):
                s += 10.0          # an explicit reason code is the strongest signal
            scores[sid] = s
        ranked = sorted((sid for sid in scores if scores[sid] > 0), key=lambda x: -scores[x])[:k]
        return [{"sop_id": sid, "title": self.sops[sid].title, "category": self.sops[sid].category,
                 "action": self.sops[sid].action, "score": round(scores[sid], 2),
                 "text": self.sops[sid].body} for sid in ranked]

    def get(self, sop_id: str) -> SOP | None:
        return self.sops.get(sop_id)
