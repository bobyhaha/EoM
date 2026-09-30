"""Restart-safe accounting and bounded retries for a time-limited experiment."""

import json
import math
import time
from pathlib import Path

import requests
from hayekmas.utils.llm import LLMClient


class CampaignStop(BaseException):
    """Not an Exception: upstream agent fallbacks must not swallow run limits."""


def append(path, value):
    import os

    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def canonical_decision(text, kind):
    """Extract the last complete decision, without generating or editing its values."""
    required = {
        "formation": "action",
        "contribute": "contribution",
        "negotiate": "contribution",
        "assess_bid": "contribution",
        "draft": ("candidate", "abstain"),
        "independent_check": ("candidate", "abstain"),
        "review": "checks",
        "revise": ("candidate", "abstain"),
        "discuss": "message",
        "vote": "candidate_id",
        "reflect": "reflect",
        "inspect": "analysis",
        "improve": "strategy",
    }
    key = required.get(kind)
    if key is None:
        return text
    decoder = json.JSONDecoder()
    candidates = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            cursor = start + 1
            continue
        cursor = start + length
        if isinstance(value, dict) and any(k in value for k in (key if isinstance(key, tuple) else (key,))):
            candidates.append(value)
    return json.dumps(candidates[-1], ensure_ascii=False, allow_nan=False) if candidates else text


class CampaignClient(LLMClient):
    api_name = "openrouter"
    MAX_EMPTY_RETRIES = 1
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, key, out, deadline, cap=25, tick=None):
        super().__init__("openai/gpt-6-luna", max_tokens=8192, temperature=None)
        self.key, self.out, self.deadline = key, Path(out), deadline
        self.cap, self.phase_cap, self.tick = cap, cap, tick
        self.context = {}
        self.records = {}
        self.ledger = self.out / "api_usage.jsonl"
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                row = json.loads(line)
                self.records[row["request"]] = row

    def usage(self):
        rows = list(self.records.values())
        return {
            "cost_usd": math.fsum(r.get("cost_usd", 0) for r in rows),
            "unconfirmed_usd": math.fsum(r["reserved_usd"] for r in rows if "cost_usd" not in r),
            "requests": len(rows),
            "input_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("completion_tokens", 0) for r in rows),
            "cap_usd": self.cap,
        }

    def record(self, row):
        self.records[row["request"]] = dict(row)
        append(self.ledger, row)
        if self.tick:
            self.tick()

    def _generate_impl(
        self,
        prompt,
        system_prompt=None,
        max_tokens=None,
        temperature=None,
        stop=None,
        reasoning_effort="medium",
        kind="solve",
        json_mode=False,
        **kwargs,
    ):
        if kwargs:
            raise CampaignStop("Unsupported model parameters")
        cap = max_tokens or self.default_max_tokens
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        for attempt in range(3):
            if time.time() >= self.deadline - 5 or (self.out.parent / "STOP").exists():
                raise CampaignStop("Deadline reached or STOP requested")
            # Retry a reasoning-only response with more room; visible protocol limits remain unchanged.
            native_cap = cap * (2**attempt)
            reserve = ((len(json.dumps(messages, ensure_ascii=False).encode()) + 2048) * 0.15 + native_cap * 0.6) / 1e6
            usage = self.usage()
            if usage["cost_usd"] + usage["unconfirmed_usd"] + reserve > min(self.cap, self.phase_cap):
                raise CampaignStop("Phase or arm spending ceiling reached")
            number = max(self.records, default=0) + 1
            row = {
                "request": number,
                "transport_revision": 2,
                "cost_handling_revision": 2,
                "status": "reserved",
                "reserved_usd": reserve,
                "time": time.time(),
                "kind": kind,
                "attempt": attempt + 1,
                "reasoning_effort": reasoning_effort,
                "max_tokens": native_cap,
                **self.context,
            }
            self.record(row)
            append(self.out / "requests.jsonl", {**row, "prompt": prompt, "system": system_prompt})
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": native_cap,
                "reasoning": {"effort": reasoning_effort},
                "stream": False,
                "provider": {
                    "require_parameters": True,
                    "allow_fallbacks": False,
                    "max_price": {"prompt": 0.15, "completion": 0.6, "request": 0},
                },
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if stop:
                payload["stop"] = stop
            retry = None
            try:
                response = requests.post(
                    self.URL,
                    headers={"Authorization": "Bearer " + self.key},
                    json=payload,
                    timeout=min(120, max(1, self.deadline - time.time())),
                    allow_redirects=False,
                )
                row["http_status"] = response.status_code
                if response.status_code != 200:
                    row["status"] = "uncertain_http_error"
                    self.record(row)
                    if response.status_code in (429, 500, 502, 503, 504):
                        retry = "HTTP " + str(response.status_code)
                    else:
                        raise CampaignStop("OpenRouter HTTP " + str(response.status_code))
                else:
                    data = response.json()
                    usage = data.get("usage") or {}
                    cost = usage.get("cost")
                    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                        # Never use an unpriced answer. Keep the full reservation
                        # and retry within the existing request/episode limits.
                        # Log only bounded receipt metadata, with credentials removed.
                        def safe(value):
                            return str(value).replace(self.key, "[redacted]")[:600] if value is not None else None

                        choices = data.get("choices")
                        first = (
                            choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                        )
                        row.update(
                            status="uncertain_missing_cost",
                            generation_id=safe(data.get("id")),
                            model=safe(data.get("model")),
                            provider=safe(data.get("provider")),
                            finish_reason=safe(first.get("finish_reason")),
                        )
                        self.record(row)
                        append(
                            self.out / "transport_errors.jsonl",
                            {
                                "request": number,
                                **self.context,
                                "kind": kind,
                                "http_status": response.status_code,
                                "generation_id": row["generation_id"],
                                "finish_reason": row["finish_reason"],
                                "cost_type": type(cost).__name__,
                                "provider_error": safe(data.get("error")),
                                "reason": "Missing or invalid usage.cost; answer discarded and reservation retained",
                            },
                        )
                        if attempt == 2:
                            raise CampaignStop("Retries exhausted: Missing/invalid API cost")
                        time.sleep(min(10 * (attempt + 1), max(0, self.deadline - time.time())))
                        continue
                    choice = (data.get("choices") or [{}])[0]
                    content = (choice.get("message") or {}).get("content")
                    row.update(
                        status="charged",
                        cost_usd=cost,
                        generation_id=data.get("id"),
                        model=data.get("model"),
                        provider=data.get("provider"),
                        finish_reason=choice.get("finish_reason"),
                        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    )
                    for name in ("prompt_tokens", "completion_tokens"):
                        value = usage.get(name)
                        if type(value) is int and value >= 0:
                            row[name] = value
                    self.record(row)
                    if cost > reserve + 1e-9:
                        raise CampaignStop("Reported cost exceeded conservative reservation")
                    if choice.get("finish_reason") == "error":
                        row["status"] = "charged_provider_error"
                        self.record(row)
                        append(
                            self.out / "responses.jsonl",
                            {
                                "request": number,
                                **self.context,
                                "kind": kind,
                                "text": content,
                                "invalid": "provider reported finish_reason=error",
                            },
                        )
                        retry = "Provider generation error"
                    elif isinstance(content, str) and content.strip():
                        append(
                            self.out / "responses.jsonl",
                            {"request": number, **self.context, "kind": kind, "text": content},
                        )
                        if json_mode:
                            normalized = canonical_decision(content, kind)
                            if normalized != content:
                                append(
                                    self.out / "format_normalizations.jsonl",
                                    {
                                        "request": number,
                                        **self.context,
                                        "kind": kind,
                                        "normalization": "last complete JSON decision",
                                        "decision": normalized,
                                    },
                                )
                            return normalized
                        return content
                    else:
                        row["status"] = "charged_empty"
                        self.record(row)
                        retry = "Empty response"
            except (requests.RequestException, ValueError) as exc:
                row.update(status="uncertain_transport_error", error_type=type(exc).__name__)
                self.record(row)
                retry = type(exc).__name__
            if attempt == 2:
                raise CampaignStop("Retries exhausted: " + str(retry))
            time.sleep(min(10 * (attempt + 1), max(0, self.deadline - time.time())))
        raise CampaignStop("No model response")
