"""PHI detection, tokenization and the fail-closed residual check.

Pipeline for one letter:
  1. detect_known_values()  - labelled fields ("Patient:", "DOB:", "Claim #"...) and
                              name patterns in prose ("Mr. Smith", "Jane Doe, born ...").
  2. tokenize()             - replace every occurrence of those values (anywhere in the
                              text, incl. inside sentences), then every generic pattern
                              (dates, IDs, phone, email, SSN), then stray name parts.
  3. find_residual_phi()    - an independent second pass over the TOKENIZED text. If it
                              finds anything that still looks like PHI, the caller must
                              fail closed (no model call, human review).

Design choice: over-redaction is acceptable, under-redaction is not. All dates are
tokenized (HIPAA treats dates tied to a person as identifiers) - the model doesn't need
them to triage a denial. Billed amounts and reason codes are NOT PHI and are kept.
"""
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime

# Policy field names (must match the `detokenize` lists in config/policies.yaml)
PATIENT_NAME = "patient_name"
DATE_OF_BIRTH = "date_of_birth"
MEMBER_ID = "member_id"
CLAIM_NUMBER = "claim_number"
DATE_OF_SERVICE = "date_of_service"
DATE = "date"            # any other date (notice date, received date...)
PHONE = "phone"
EMAIL = "email"
SSN = "ssn"
ADDRESS = "address"
OTHER_ID = "other_id"

TOKEN_RE = re.compile(r"<([A-Z_]+)_([0-9a-f]{6})>")

# ---- labelled fields -------------------------------------------------------
LABELS = {
    PATIENT_NAME: r"patient(?:'s)?(?:\s+full)?(?:\s+name)?|member\s+name|subscriber(?:\s+name)?|"
                  r"insured(?:\s+name)?|beneficiary(?:\s+name)?|enrollee(?:\s+name)?|name",
    DATE_OF_BIRTH: r"date\s+of\s+birth|birth\s*date|d\.?o\.?b\.?",
    MEMBER_ID: r"member\s*(?:id|#|number|no\.?)|subscriber\s*(?:id|#|number|no\.?)|"
               r"policy\s*(?:id|#|number|no\.?)|insurance\s*id|medicaid\s*(?:id|#)|"
               r"medicare\s*(?:id|#)|mbi|patient\s*(?:id|#|account(?:\s*(?:number|#))?)|mrn",
    CLAIM_NUMBER: r"claim\s*(?:number|#|no\.?|id)|icn|dcn|reference\s*(?:number|#|no\.?)",
    DATE_OF_SERVICE: r"dates?\s*(?:\(s\)\s*)?of\s+service|service\s+dates?|d\.?o\.?s\.?|"
                     r"admission\s+date|admit\s+date|discharge\s+date",
    ADDRESS: r"(?:patient\s+|member\s+|home\s+|mailing\s+)?address",
    PHONE: r"(?:patient\s+|member\s+|home\s+)?(?:phone|telephone|tel\.?)",
    SSN: r"ssn|social\s+security(?:\s+number)?",
}
# A label, then ":" or "#", then the value up to end of line / " | " / ";" / 2+ spaces.
_LABEL_RE = {
    ftype: re.compile(rf"(?im)(?:^|(?<=\s{{2}})|(?<=\|)|(?<=;))\s*(?:{pat})\s*[:#]\s*"
                      rf"(?P<val>[^\n|;]+?)\s*(?=$|\s{{2,}}|\s*[|;])")
    for ftype, pat in LABELS.items()
}

# ---- generic patterns ------------------------------------------------------
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
    re.compile(rf"(?i)\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"),
    re.compile(rf"(?i)\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\.?,?\s+\d{{4}}\b"),
]
GENERIC_PATTERNS = [
    (EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    (SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (PHONE, re.compile(r"(?<!\w)\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    # IDs: mixed letters+digits (>=6 chars, >=3 digits), or long pure digit runs
    (OTHER_ID, re.compile(r"\b(?=[A-Z0-9-]*\d[A-Z0-9-]*\d[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])"
                          r"[A-Z0-9][A-Z0-9-]{5,}\b")),
    (OTHER_ID, re.compile(r"\b\d{7,}\b")),
]

# ---- names in prose (no label) ----------------------------------------------
_NAME_WORD = r"[A-Z][a-z][a-zA-Z'\-]*"
_NAME = rf"{_NAME_WORD}(?:\s+(?:[A-Z]\.\s+)?{_NAME_WORD}){{0,3}}"
PROSE_NAME_PATTERNS = [
    re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Mx)\.?\s+(?P<name>{_NAME})"),
    re.compile(rf"\b(?:patient|member|subscriber|beneficiary|enrollee|insured)\s+"
               rf"(?P<name>{_NAME_WORD}\s+{_NAME_WORD}(?:\s+{_NAME_WORD})?)"),
    re.compile(rf"(?P<name>{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,3}}),?\s+(?:born|DOB|d\.o\.b\.|date of birth)\b"),
]
# words that may appear inside a name match but are not identifying on their own
NAME_STOPWORDS = {"test", "patient", "member", "the", "and", "our", "your", "this", "that",
                  "dear", "plan", "health", "care", "services", "provider", "claim", "claims",
                  "department", "program", "records", "show", "mr", "mrs", "ms", "miss", "dr"}


@dataclass
class TokenizedDoc:
    text: str                                   # what the model is allowed to see
    mapping: dict = field(default_factory=dict)  # token -> (field_type, value) -> vault
    fields: dict = field(default_factory=dict)   # field_type -> primary token
    residual: list = field(default_factory=list)  # reasons PHI may remain (no values!)

    @property
    def safe(self) -> bool:
        return not self.residual


def normalize_date(s: str) -> str | None:
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", s.strip(), flags=re.I).replace(",", "")
    s = re.sub(r"\s+of\s+", " ", s, flags=re.I)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m.%d.%Y",
                "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s.replace(".", "") if "%b" in fmt or "%B" in fmt else s,
                                     fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---- optional NER layer (Microsoft Presidio + spaCy) ------------------------
# Regex can't spot a bare name with no cue ("Re: Jane Doe."). If Presidio is installed
# it adds PERSON detection to both tokenization and the residual check.
# Disable with PHI_NER=off. Without it, the regex layer + fail-closed rules still apply.
_analyzer = None


def _ner_enabled() -> bool:
    return os.getenv("PHI_NER", "auto").lower() != "off"


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": os.getenv("SPACY_MODEL", "en_core_web_sm")}],
        }).create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    return _analyzer


def ner_person_names(text: str) -> list[str]:
    """PERSON entities found by Presidio, trimmed to one line, stopwords ignored."""
    if not _ner_enabled():
        return []
    try:
        analyzer = _get_analyzer()
    except Exception:          # Presidio / spaCy model not installed -> regex-only mode
        return []
    names = []
    for r in analyzer.analyze(text=text, language="en", entities=["PERSON"], score_threshold=0.6):
        span = text[r.start:r.end].split("\n")[0]
        # keep only the leading run of Name-like words: "Maria Lopez   DOB" -> "Maria Lopez"
        m = re.match(rf"\s*({_NAME_WORD}(?:\s+{_NAME_WORD})*)", span)
        if not m or "REDACTED" in m.group(1):
            continue
        name = m.group(1)
        if any(w.lower() not in NAME_STOPWORDS for w in name.split()):
            names.append(name)
    return names


def ner_status() -> str:
    if not _ner_enabled():
        return "off"
    try:
        _get_analyzer()
        return "presidio"
    except Exception:
        return "unavailable (regex only)"


def _trim_value(val: str) -> str:
    """A labelled value ends at a sentence break: 'Dates of Service: 09/01/2026. Ms. X ...'"""
    val = re.split(r"(?:(?<=\w{2})|(?<=>))\.\s+", val, maxsplit=1)[0]
    return val.strip().rstrip(".,")


def detect_known_values(text: str) -> list[tuple[str, str]]:
    """Return [(field_type, value)] found via labels, prose name patterns and NER."""
    found: list[tuple[str, str]] = []
    for ftype, rx in _LABEL_RE.items():
        for m in rx.finditer(text):
            val = _trim_value(m.group("val"))
            if val and not TOKEN_RE.fullmatch(val):
                found.append((ftype, val))
    for rx in PROSE_NAME_PATTERNS:
        for m in rx.finditer(text):
            name = m.group("name").strip()
            words = [w for w in name.split() if w.lower().strip(".") not in NAME_STOPWORDS]
            if words:
                found.append((PATIENT_NAME, name))
    for name in ner_person_names(text):
        found.append((PATIENT_NAME, name))
    # de-duplicate, keep first type seen for a value
    seen, out = set(), []
    for ftype, val in found:
        key = re.sub(r"\s+", " ", val).lower()
        if key not in seen:
            seen.add(key)
            out.append((ftype, val))
    return out


def _value_regex(value: str) -> re.Pattern:
    parts = [re.escape(p) for p in value.split()]
    return re.compile(r"(?<![\w-])" + r"\s+".join(parts) + r"(?![\w-])", re.I)


def _sub_outside_tokens(text: str, rx: re.Pattern, repl) -> str:
    """Apply rx only to the parts of text that are not already tokens."""
    pieces, last = [], 0
    for m in TOKEN_RE.finditer(text):
        pieces.append(rx.sub(repl, text[last:m.start()]))
        pieces.append(m.group(0))
        last = m.end()
    pieces.append(rx.sub(repl, text[last:]))
    return "".join(pieces)


def _name_parts(values: list[tuple[str, str]]) -> set[str]:
    parts = set()
    for ftype, val in values:
        if ftype == PATIENT_NAME:
            for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", val):
                if len(w) >= 3 and w.lower() not in NAME_STOPWORDS:
                    parts.add(w)
    return parts


def tokenize(text: str) -> TokenizedDoc:
    doc = TokenizedDoc(text=text)
    by_value: dict[tuple[str, str], str] = {}   # (type, normalized value) -> token

    def token_for(ftype: str, value: str) -> str:
        key = (ftype, re.sub(r"\s+", " ", value).lower())
        if key not in by_value:
            tok = f"<{ftype.upper()}_{secrets.token_hex(3)}>"
            by_value[key] = tok
            doc.mapping[tok] = (ftype, value)
            doc.fields.setdefault(ftype, tok)
        return by_value[key]

    known = detect_known_values(text)
    # remember which dates are DOB / DOS so the same date in any format gets the right type
    special_dates = {normalize_date(v): t for t, v in known
                     if t in (DATE_OF_BIRTH, DATE_OF_SERVICE) and normalize_date(v)}

    out = text
    # 1) known values, longest first so "Jane Q Doe" wins over "Jane"
    for ftype, val in sorted(known, key=lambda kv: -len(kv[1])):
        tok = token_for(ftype, val)
        out = _sub_outside_tokens(out, _value_regex(val), lambda m, t=tok: t)

    # 2) generic patterns: dates first, then IDs / contact details
    def date_repl(m):
        iso = normalize_date(m.group(0))
        ftype = special_dates.get(iso, DATE)
        return token_for(ftype, m.group(0))
    for rx in DATE_PATTERNS:
        out = _sub_outside_tokens(out, rx, date_repl)
    for ftype, rx in GENERIC_PATTERNS:
        out = _sub_outside_tokens(out, rx, lambda m, f=ftype: token_for(f, m.group(0)))

    # 3) stray parts of a known name ("... Mr. Lima ..." after "Patient: Test Patient Lima")
    for part in sorted(_name_parts(known), key=len, reverse=True):
        out = _sub_outside_tokens(out, _value_regex(part),
                                  lambda m: token_for(PATIENT_NAME, m.group(0)))

    doc.text = out
    doc.residual = find_residual_phi(out, [v for _, v in known] + list(_name_parts(known)),
                                     found_types={t for t, _ in known})
    return doc


def find_residual_phi(tokenized: str, known_values: list[str], found_types: set[str]) -> list[str]:
    """Independent check on text that is about to leave for the model.

    Returns human-readable reasons WITHOUT the offending values (they go to the audit log).
    Empty list == safe to send.
    """
    reasons = []
    flat = re.sub(r"\s+", " ", TOKEN_RE.sub(" ", tokenized)).lower()

    for i, val in enumerate(known_values):
        v = re.sub(r"\s+", " ", val).lower().strip()
        if v and re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", flat):
            reasons.append(f"known PHI value #{i} still present")

    stripped = TOKEN_RE.sub(" ", tokenized)
    for rx in DATE_PATTERNS:
        if rx.search(stripped):
            reasons.append("date pattern still present")
    for ftype, rx in GENERIC_PATTERNS:
        if rx.search(stripped):
            reasons.append(f"{ftype} pattern still present")
    for ftype, rx in _LABEL_RE.items():
        for m in rx.finditer(tokenized):
            leftover = TOKEN_RE.sub("", _trim_value(m.group("val")))
            if re.search(r"[A-Za-z0-9]", leftover):
                reasons.append(f"labelled {ftype} field has untokenized content")
    for rx in PROSE_NAME_PATTERNS:
        for m in rx.finditer(stripped):
            words = [w for w in m.group("name").split() if w.lower() not in NAME_STOPWORDS]
            if words:
                reasons.append("possible person name in prose")
    # tokens are swapped for a fixed placeholder so NER output doesn't depend on random hex
    if ner_person_names(TOKEN_RE.sub("[REDACTED]", tokenized)):
        reasons.append("NER found a possible person name")

    # If we could not find WHO the letter is about, we cannot prove we removed it.
    if not ({PATIENT_NAME, MEMBER_ID} & found_types):
        reasons.append("no patient identity field found - cannot confirm PHI was located")
    return sorted(set(reasons))
