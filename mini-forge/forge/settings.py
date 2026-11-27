"""Paths and environment settings in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"
SOP_DIR = ROOT / "sops"
LABELS_PATH = ROOT / "labels.json"

# Runtime folders can be redirected (tests point these at a temp dir).
DATA_DIR = Path(os.getenv("FORGE_DATA_DIR", ROOT))
VAULT_PATH = DATA_DIR / "vault.db"
OUTBOX_DIR = DATA_DIR / "outbox"
LOG_DIR = DATA_DIR / "logs"
RESULTS_DIR = DATA_DIR / "results"
AUDIT_LOG = LOG_DIR / "audit.jsonl"
MODEL_CALL_LOG = LOG_DIR / "model_calls.jsonl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "mock-triage-v1").strip()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
SLEEP_BETWEEN_DOCS = float(os.getenv("SLEEP_BETWEEN_DOCS", "0") or 0)
