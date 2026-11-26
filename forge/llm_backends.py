"""LLM backends behind the gateway. Only forge/gateway.py may call these.

Internally we keep one message format (Anthropic-style content blocks):
  user:      "text"  or  [{"type": "tool_result", "tool_use_id", "content"}]
  assistant: [{"type": "text", "text"}, {"type": "tool_use", "id", "name", "input"}]
OpenAI-compatible providers are converted to/from that format here.
"""
import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    def as_assistant_blocks(self) -> list:
        blocks = [{"type": "text", "text": self.text}] if self.text else []
        blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                   for tc in self.tool_calls]
        return blocks


class RetryableError(Exception):
    """Rate limit / transient server error - the gateway backs off and retries."""


# ---------------------------------------------------------------- Anthropic / Bedrock
class AnthropicBackend:
    def __init__(self, model: str, bedrock: bool = False):
        import anthropic
        self._anthropic = anthropic
        self.model = model
        self.client = (anthropic.AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
                       if bedrock else anthropic.Anthropic())

    def complete(self, system, messages, tools, max_tokens) -> LLMResponse:
        try:
            r = self.client.messages.create(model=self.model, system=system, messages=messages,
                                            tools=tools, max_tokens=max_tokens, temperature=0)
        except (self._anthropic.RateLimitError, self._anthropic.InternalServerError,
                self._anthropic.APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return LLMResponse(text, calls, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)


# ---------------------------------------------------------------- OpenAI-compatible
class OpenAICompatBackend:
    """Gemini (free tier), Groq, OpenRouter, OpenAI... anything speaking /chat/completions."""

    def __init__(self, model: str):
        import openai
        self._openai = openai
        self.model = model
        self.client = openai.OpenAI(base_url=os.getenv("OPENAI_COMPAT_BASE_URL") or None,
                                    api_key=os.getenv("OPENAI_COMPAT_API_KEY"))

    @staticmethod
    def to_openai(system, messages, tools):
        out = [{"role": "system", "content": system}]
        for m in messages:
            if isinstance(m["content"], str):
                out.append({"role": m["role"], "content": m["content"]})
            elif m["role"] == "assistant":
                text = "".join(b["text"] for b in m["content"] if b["type"] == "text")
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in m["content"] if b["type"] == "tool_use"]
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:  # user message carrying tool results
                for b in m["content"]:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                    "content": b["content"]})
                    elif b["type"] == "text":
                        out.append({"role": "user", "content": b["text"]})
        fns = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                 "parameters": t["input_schema"]}} for t in tools]
        return out, fns

    def complete(self, system, messages, tools, max_tokens) -> LLMResponse:
        msgs, fns = self.to_openai(system, messages, tools)
        try:
            r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=fns,
                                                    tool_choice="auto", temperature=0,
                                                    max_tokens=max_tokens)
        except (self._openai.RateLimitError, self._openai.InternalServerError,
                self._openai.APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        choice = r.choices[0]
        calls = []
        for i, tc in enumerate(choice.message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": tc.function.arguments}
            calls.append(ToolCall(tc.id or f"call_{i}", tc.function.name, args))
        usage = r.usage
        return LLMResponse(choice.message.content or "", calls,
                           getattr(usage, "prompt_tokens", 0) or 0,
                           getattr(usage, "completion_tokens", 0) or 0, choice.finish_reason or "")


# ---------------------------------------------------------------- Mock
class MockBackend:
    """Deterministic scripted stand-in for an LLM, for running without an API key.

    It sees exactly what a real model would see (the tokenized letter + tool results via
    the gateway) and emits real tool calls, so the whole pipeline - policy, vault, scoped
    de-tokenization, audit, eval - is exercised. Its "reasoning" is a few rules; it
    measures the plumbing, not LLM quality.
    """
    model = "mock-triage-v1"

    def complete(self, system, messages, tools, max_tokens) -> LLMResponse:
        letter = re.search(r"<letter>(.*)</letter>", messages[0]["content"], re.S).group(1)
        results = [b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                   for b in m["content"] if b["type"] == "tool_result"]
        calls_made = [b for m in messages if m["role"] == "assistant"
                      for b in m["content"] if b["type"] == "tool_use"]
        n = len(calls_made)

        def call(name, args, text=""):
            payload = json.dumps(messages) + json.dumps(args)
            return LLMResponse(text, [ToolCall(f"mock_{n}", name, args)],
                               input_tokens=len(payload) // 4, output_tokens=len(json.dumps(args)) // 4,
                               stop_reason="tool_use")

        body = "\n".join(l for l in letter.splitlines()
                         if l.strip() and ":" not in l and "fictional" not in l
                         and "questions" not in l and "Department" not in l)
        garbled = len(re.findall(r"[#@%]|\b\w*\d\w*[a-z]\w*\b", body))
        unreadable = "unreadable" in body.lower() or garbled > 8
        carc = re.search(r"reason code:\s*(\d+)", letter, re.I)
        billed = re.search(r"Billed amount:\s*\$([\d,]+(?:\.\d+)?)", letter)
        billed = float(billed.group(1).replace(",", "")) if billed else None

        if not calls_made:
            q = ("unreadable reason cannot be determined" if unreadable
                 else (f"CARC {carc.group(1)} " if carc else "") + body)
            return call("search_sop", {"query": q[:300]})

        last = json.loads(results[-1]["content"]) if results else {}
        if last.get("status") in ("denied", "error"):
            return call("flag_for_human_review", {
                "reason": f"Previous tool call was {last.get('status')}: {last.get('reason', '')}",
                "category": "unknown", "recommended_action": "human_review"})

        search = next(json.loads(r["content"]) for r in results if "results" in json.loads(r["content"]))
        top = (search.get("results") or [None])[0]
        if unreadable or top is None or top["action"] is None:
            return call("flag_for_human_review", {
                "reason": "Letter is unreadable or the denial reason cannot be determined (SOP-199 rule 1).",
                "category": "unreadable", "recommended_action": "human_review", "sop_id": "SOP-199"})

        cat, action, sop = top["category"], top["action"], top["sop_id"]
        if cat == "medical_necessity":
            if billed is None or billed > 5000:
                return call("flag_for_human_review", {
                    "reason": f"Medical necessity denial with billed amount above $5,000 ({sop} step 3).",
                    "category": cat, "recommended_action": action, "sop_id": sop})
            if not any(c["name"] == "draft_appeal" for c in calls_made):
                tok = {t: f"<{t}_{h}>" for t, h in re.findall(r"<([A-Z_]+)_([0-9a-f]{6})>", letter)}
                return call("draft_appeal", {
                    "sop_id": sop,
                    # uses the privacy tokens it was given; the DOB token is included on purpose
                    # to show that draft_appeal's scope does NOT release date_of_birth
                    "appeal_letter": (
                        f"Re: claim {tok.get('CLAIM_NUMBER')} for {tok.get('PATIENT_NAME')} "
                        f"(member {tok.get('MEMBER_ID')}, DOB {tok.get('DATE_OF_BIRTH')}), date of "
                        f"service {tok.get('DATE_OF_SERVICE')}. We request reconsideration of the "
                        "medical necessity denial. Supporting clinical records are attached."),
                    "records_to_attach": ["clinical notes", "physician orders", "test results"]})
        return call("route_to_work_queue", {
            "action": action, "category": cat, "sop_id": sop,
            "note": f"Denial matches {sop}; action {action}."})
