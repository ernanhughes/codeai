from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .adapters import CallResult, CognitionAdapter
from .domain import CallSpec


class ProviderError(RuntimeError):
    pass


class MissingCredentialsError(ProviderError):
    pass


# Versioned pricing table. Unknown models -> cost None (never pretend).
PRICING_VERSION = "2026-09-01"
# (input USD per 1M tokens, output USD per 1M tokens) by exact model id prefix.
PRICING_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku": (0.25, 1.25),
    "claude-opus-4": (15.00, 75.00),
}


def estimate_cost_usd(model_id: str | None, input_tokens: int, output_tokens: int) -> float | None:
    if not model_id:
        return None
    for prefix, (inp, out) in PRICING_TABLE.items():
        if model_id.startswith(prefix):
            return input_tokens / 1_000_000 * inp + output_tokens / 1_000_000 * out
    return None


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:  # network/HTTP errors become failed calls upstream
        raise ProviderError(f"provider request failed: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned non-JSON response: {exc}") from exc
    if isinstance(parsed, dict) and parsed.get("error"):
        raise ProviderError(f"provider error: {parsed['error']}")
    if not isinstance(parsed, dict):
        raise ProviderError("provider returned unexpected response shape")
    return parsed


def _params(spec: CallSpec, key: str, default: Any = None) -> Any:
    if key in spec.parameters:
        return spec.parameters[key]
    variant_value = getattr(spec.variant, key, None)
    return variant_value if variant_value is not None else default


def _prompt_messages(spec: CallSpec) -> list[dict[str, str]]:
    instruction = spec.instruction or ""
    prompt = spec.context.prompt or ""
    user_text = f"{instruction}\n\n{prompt}".strip() or prompt or instruction
    return [{"role": "user", "content": user_text}]


@dataclass
class OpenAIAdapter(CognitionAdapter):
    """OpenAI chat-completions adapter (also usable against compatible clouds)."""

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 120.0
    http_post: Callable[..., dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")

    def _require_key(self) -> str:
        if not self.api_key:
            raise MissingCredentialsError(
                "OpenAI credentials absent: set OPENAI_API_KEY or pass api_key"
            )
        return self.api_key

    def invoke(self, spec: CallSpec) -> CallResult:
        started = time.perf_counter()
        started_at = _now_iso()
        try:
            key = self._require_key()
            body: dict[str, Any] = {
                "model": self.model,
                "messages": _prompt_messages(spec),
            }
            temperature = _params(spec, "temperature")
            if temperature is not None:
                body["temperature"] = temperature
            seed = _params(spec, "seed")
            if seed is not None:
                try:
                    body["seed"] = int(seed)
                except (TypeError, ValueError):
                    pass
            post = self.http_post or _post_json
            parsed = post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                body,
                {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                self.timeout,
            )
            text = _openai_text(parsed)
            usage = parsed.get("usage", {}) if isinstance(parsed.get("usage"), dict) else {}
            in_tokens = int(usage.get("prompt_tokens", 0) or 0)
            out_tokens = int(usage.get("completion_tokens", 0) or 0)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=estimate_cost_usd(self.model, in_tokens, out_tokens),
                latency_ms=latency_ms,
                provider="openai",
                model=self.model,
                model_version=str(parsed.get("model", self.model)),
                fingerprint=parsed.get("system_fingerprint"),
                provider_call_id=parsed.get("id"),
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
            )
        except (MissingCredentialsError, ProviderError) as exc:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                status="failed",
                error=str(exc),
                provider="openai",
                model=self.model,
                started_at=started_at,
                completed_at=_now_iso(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


def _openai_text(parsed: dict[str, Any]) -> str:
    choices = parsed.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        raise ProviderError("OpenAI response contained no choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):  # structured content parts
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content or "")


@dataclass
class AnthropicAdapter(CognitionAdapter):
    """Anthropic messages-API adapter."""

    model: str = "claude-haiku-4-5-20251001"
    api_key: str | None = None
    base_url: str = "https://api.anthropic.com/v1"
    timeout: float = 120.0
    http_post: Callable[..., dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def invoke(self, spec: CallSpec) -> CallResult:
        started = time.perf_counter()
        started_at = _now_iso()
        try:
            if not self.api_key:
                raise MissingCredentialsError(
                    "Anthropic credentials absent: set ANTHROPIC_API_KEY or pass api_key"
                )
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": int(_params(spec, "max_tokens", 1024)),
                "messages": _prompt_messages(spec),
            }
            temperature = _params(spec, "temperature")
            if temperature is not None:
                body["temperature"] = temperature
            post = self.http_post or _post_json
            parsed = post(
                f"{self.base_url.rstrip('/')}/messages",
                body,
                {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                self.timeout,
            )
            if parsed.get("type") == "error":
                raise ProviderError(f"Anthropic error: {parsed.get('error')}")
            blocks = parsed.get("content", [])
            text = "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            )
            usage = parsed.get("usage", {}) if isinstance(parsed.get("usage"), dict) else {}
            in_tokens = int(usage.get("input_tokens", 0) or 0)
            out_tokens = int(usage.get("output_tokens", 0) or 0)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=estimate_cost_usd(self.model, in_tokens, out_tokens),
                latency_ms=latency_ms,
                provider="anthropic",
                model=self.model,
                model_version=str(parsed.get("model", self.model)),
                provider_call_id=parsed.get("id"),
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
            )
        except (MissingCredentialsError, ProviderError) as exc:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                status="failed",
                error=str(exc),
                provider="anthropic",
                model=self.model,
                started_at=started_at,
                completed_at=_now_iso(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


@dataclass
class OpenAICompatibleAdapter(CognitionAdapter):
    """Local/self-hosted OpenAI-compatible endpoint (e.g. Ollama /v1). No key required."""

    model: str = "qwen"
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    provider_name: str = "openai-compatible"
    timeout: float = 300.0
    http_post: Callable[..., dict[str, Any]] | None = None

    def invoke(self, spec: CallSpec) -> CallResult:
        started = time.perf_counter()
        started_at = _now_iso()
        try:
            if not self.base_url:
                raise ProviderError("OpenAI-compatible base_url is required")
            body: dict[str, Any] = {"model": self.model, "messages": _prompt_messages(spec)}
            temperature = _params(spec, "temperature")
            if temperature is not None:
                body["temperature"] = temperature
            headers = {"Content-Type": "application/json"}
            key = self.api_key or os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            post = self.http_post or _post_json
            parsed = post(
                f"{self.base_url.rstrip('/')}/chat/completions", body, headers, self.timeout
            )
            text = _openai_text(parsed)
            usage = parsed.get("usage", {}) if isinstance(parsed.get("usage"), dict) else {}
            in_tokens = int(usage.get("prompt_tokens", 0) or 0)
            out_tokens = int(usage.get("completion_tokens", 0) or 0)
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=estimate_cost_usd(self.model, in_tokens, out_tokens),
                latency_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider_name,
                model=self.model,
                model_version=str(parsed.get("model", self.model)),
                provider_call_id=parsed.get("id"),
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
            )
        except ProviderError as exc:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                status="failed",
                error=str(exc),
                provider=self.provider_name,
                model=self.model,
                started_at=started_at,
                completed_at=_now_iso(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
