"""Every test gets its own temp folder for vault / logs / outbox / results."""
import pytest

from forge import settings


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "VAULT_PATH", tmp_path / "vault.db")
    monkeypatch.setattr(settings, "OUTBOX_DIR", tmp_path / "outbox")
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(settings, "AUDIT_LOG", tmp_path / "logs" / "audit.jsonl")
    monkeypatch.setattr(settings, "MODEL_CALL_LOG", tmp_path / "logs" / "model_calls.jsonl")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "SLEEP_BETWEEN_DOCS", 0)
    return tmp_path
