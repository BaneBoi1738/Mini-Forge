"""The ONE door to the LLM.

Every model call in Mini-Forge goes through ModelGateway.chat(), which:
  1. enforces the per-run limits from agents.yaml (max_steps, max_tokens_per_run,
     max_cost_usd_per_run) - checked BEFORE the call with an estimate, and AFTER with
     the real usage;
  2. runs a last-line PHI guard: if any vault value for this letter appears in the
     outgoing payload, the call is blocked (defence in depth - tokenization should
     already have removed it);
  3. logs model, tokens, cost, latency and correlation ID to the audit log, and the
     full (tokenized) request/response to logs/model_calls.jsonl for the leak scan.
"""
import json
import sys
import time

import yaml

from forge import settings
from forge.audit import append_jsonl, audit, now_iso
from forge.catalogue import AgentSpec
from forge.llm_backends import (AnthropicBackend, LLMResponse, MockBackend,
                                OpenAICompatBackend, QuotaExhausted, RetryableError)


class BudgetExceeded(Exception):
    pass


class PHIBlocked(Exception):
    pass


def make_backend(provider: str, model: str):
    if provider == "mock":
        return MockBackend()
    if provider == "anthropic":
        return AnthropicBackend(model)
    if provider == "bedrock":
        return AnthropicBackend(model, bedrock=True)
    if provider == "openai_compat":
        return OpenAICompatBackend(model)
    raise ValueError(f"unknown LLM_PROVIDER '{provider}'")


def load_prices() -> dict:
    cfg = yaml.safe_load(open(settings.CONFIG_DIR / "pricing.yaml"))
    return cfg


class ModelGateway:
    def __init__(self, spec: AgentSpec, *, run_id: str, correlation_id: str, document_id: str,
                 phi_values: list[str], provider: str | None = None, model: str | None = None,
                 backend=None):
        self.spec = spec
        self.run_id, self.correlation_id, self.document_id = run_id, correlation_id, document_id
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL
        if self.provider == "mock":
            self.model = MockBackend.model
        self.backend = backend or make_backend(self.provider, self.model)
        prices = load_prices()
        self.price_configured = self.model in prices["models"]
        self.price = prices["models"].get(self.model, prices["default"])
        self.phi_values = [v.lower() for v in phi_values if len(v) >= 3]
        self.steps = 0
        self.tokens_used = 0
        self.cost_usd = 0.0

    def _cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok * self.price["input"] + out_tok * self.price["output"]) / 1_000_000

    def _ids(self):
        return dict(run_id=self.run_id, correlation_id=self.correlation_id,
                    document_id=self.document_id)

    def chat(self, system: str, messages: list, tools: list) -> LLMResponse:
        request = {"system": system, "messages": messages, "tools": tools}
        payload = json.dumps(request)

        # --- 1. budget pre-check (estimate ~4 chars/token, assume full output) -----------
        est_in = len(payload) // 4
        max_out = settings.MAX_OUTPUT_TOKENS
        problems = []
        if self.steps >= self.spec.max_steps:
            problems.append(f"max_steps {self.spec.max_steps} reached")
        if self.tokens_used + est_in + max_out > self.spec.max_tokens_per_run:
            problems.append(f"next call could exceed max_tokens_per_run "
                            f"({self.tokens_used}+~{est_in + max_out} > {self.spec.max_tokens_per_run})")
        if self.cost_usd + self._cost(est_in, max_out) > self.spec.max_cost_usd_per_run:
            problems.append(f"next call could exceed max_cost_usd_per_run {self.spec.max_cost_usd_per_run}")
        if problems:
            audit("budget_block", **self._ids(), step=self.steps, reasons=problems)
            raise BudgetExceeded("; ".join(problems))

        # --- 2. PHI guard on the exact bytes leaving the process -------------------------
        low = payload.lower()
        hits = sum(1 for v in self.phi_values if v in low)
        if hits:
            audit("phi_guard_block", **self._ids(), step=self.steps, matches=hits)
            raise PHIBlocked(f"{hits} vault value(s) found in outgoing model request")

        # --- 3. call with bounded retry on rate limits / timeouts -------------------------
        # Worst case per call: 3 attempts x 60 s timeout + 10 + 20 s back-off. Never hangs.
        t0 = time.time()
        for attempt in range(3):
            try:
                resp = self.backend.complete(system, messages, tools, max_out)
                break
            except QuotaExhausted as e:
                audit("model_quota_exhausted", **self._ids(), error=str(e)[:300])
                raise
            except RetryableError as e:
                wait = 10 * 2 ** attempt
                audit("model_retry", **self._ids(), attempt=attempt + 1, wait_s=wait,
                      error=str(e)[:200])
                if attempt < 2:
                    print(f"   [{self.document_id}] model busy/rate-limited, retry {attempt + 1}/2 "
                          f"in {wait}s: {str(e)[:120]}", file=sys.stderr, flush=True)
                    time.sleep(wait)
        else:
            raise RuntimeError("model call failed after 3 attempts (rate limit / timeout)")
        latency_ms = int((time.time() - t0) * 1000)

        self.steps += 1
        cost = self._cost(resp.input_tokens, resp.output_tokens)
        self.tokens_used += resp.input_tokens + resp.output_tokens
        self.cost_usd += cost

        append_jsonl(settings.MODEL_CALL_LOG, {
            "ts": now_iso(), **self._ids(), "step": self.steps,
            "provider": self.provider, "model": self.model,
            "request": request,
            "response": {"text": resp.text, "tool_calls": [tc.__dict__ for tc in resp.tool_calls],
                         "stop_reason": resp.stop_reason},
            "usage": {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            "cost_usd": round(cost, 6), "latency_ms": latency_ms,
            "price_configured": self.price_configured,
        })
        audit("model_call", **self._ids(), step=self.steps, provider=self.provider,
              model=self.model, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
              cost_usd=round(cost, 6), run_tokens=self.tokens_used,
              run_cost_usd=round(self.cost_usd, 6), latency_ms=latency_ms,
              tool_calls=[tc.name for tc in resp.tool_calls])

        # --- 4. budget post-check with real usage ------------------------------------------
        if self.tokens_used > self.spec.max_tokens_per_run or self.cost_usd > self.spec.max_cost_usd_per_run:
            audit("budget_block", **self._ids(), step=self.steps,
                  reasons=["actual usage exceeded run budget; response discarded"])
            raise BudgetExceeded("actual usage exceeded run budget")
        return resp
