from __future__ import annotations
import json
from pathlib import Path

import pytest

from forge.phi import TOKEN_RE, find_residual_phi, tokenize

ROOT = Path(__file__).resolve().parent.parent
LABELS = {l["document_id"]: l for l in json.loads((ROOT / "labels.json").read_text())["labels"]}


@pytest.mark.parametrize("doc_id", sorted(LABELS))
def test_no_label_phi_survives_tokenization(doc_id):
    raw = (ROOT / "documents" / f"{doc_id}.txt").read_text()
    t = tokenize(raw)
    assert t.safe, t.residual
    for field, value in LABELS[doc_id]["phi"].items():
        assert value.lower() not in t.text.lower(), f"{field} leaked"


def test_mapping_round_trips_to_original_values():
    t = tokenize((ROOT / "documents" / "DOC-001.txt").read_text())
    primary = {ftype: t.mapping[tok][1] for ftype, tok in t.fields.items()}
    assert primary["patient_name"] == "Test Patient Alpha"
    assert primary["claim_number"] == "TEST-CLM-2601"
    assert primary["date_of_birth"] == "1962-02-11"
    assert primary["date_of_service"] == "2026-08-14"


def test_phi_inside_a_sentence_is_tokenized_with_the_same_token():
    t = tokenize((ROOT / "documents" / "DOC-012.txt").read_text())
    name_tok = t.fields["patient_name"]
    dob_tok = t.fields["date_of_birth"]
    assert f"that {name_tok}, born {dob_tok}" in t.text


def test_non_phi_needed_for_triage_is_kept():
    t = tokenize((ROOT / "documents" / "DOC-005.txt").read_text())
    assert "$12,800.00" in t.text
    assert "reason code: 50" in t.text


def test_unseen_formats():
    raw = ("Patient Name: Maria Lopez   DOB: 03/15/1971 | Member #: ABC1234567 ; Claim No.: CLM-998877\n"
           "Dates of Service: 09/01/2026. Ms. Lopez was admitted on September 1st, 2026.\n"
           "Call 555-123-4567, email m.lopez@mail.com, SSN 123-45-6789.")
    t = tokenize(raw)
    for secret in ["Maria", "Lopez", "03/15/1971", "ABC1234567", "CLM-998877", "09/01/2026",
                   "September 1st, 2026", "555-123-4567", "m.lopez@mail.com", "123-45-6789"]:
        assert secret not in t.text, secret
    assert t.safe, t.residual
    assert "was admitted on" in t.text          # the sentence itself survives


def test_fail_closed_when_no_patient_identity_found():
    t = tokenize("The claim was denied for a missing modifier. Please resubmit.")
    assert not t.safe
    assert any("identity" in r for r in t.residual)


def test_residual_check_catches_leftover_value_without_printing_it():
    reasons = find_residual_phi("Patient: <PATIENT_NAME_abcdef>. Jane Roe called.",
                                ["Jane Roe"], {"patient_name"})
    assert reasons and all("Jane" not in r for r in reasons)


def test_tokens_have_expected_shape():
    t = tokenize((ROOT / "documents" / "DOC-003.txt").read_text())
    assert all(TOKEN_RE.fullmatch(tok) for tok in t.mapping)
