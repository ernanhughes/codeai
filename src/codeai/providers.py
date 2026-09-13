from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .adapters import (
    NORMALIZER_VERSION,
    CallResult,
    CognitionAdapter,
    TransportObservation,
    TransportOutcome,
    is_credential_key,
    sanitize_effective_params,
)
from .domain import CallSpec


class ProviderError(RuntimeError):
    pass


class ProviderHttpError(ProviderError):
    """HTTP-layer failure carrying its status code for error classification."""

    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


class TransportFailure(ProviderError):
    """No HTTP response exists (timeout, DNS, connection reset, ...).

    Distinct from an HTTP error response: there is no status, no headers,
    and no body to preserve. The underlying exception type is retained.
    """

    def __init__(self, exception_type: str, message: str) -> None:
        super().__init__(message)
        self.exception_type = exception_type


class MissingCredentialsError(ProviderError):
    pass


# Canonical OpenCode Zen gateway. OPENAI_API_KEY is deliberately NOT consulted
# here: OpenAI-compatible protocol != OpenAI provider, and the Zen credential
# must never be silently sourced from an unrelated provider's key.
OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/go"
OPENCODE_ZEN_API_KEY_ENV = "OPENCODE_ZEN_API_KEY"


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


def estimate_cost_usd(
    model_id: str | None, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimate cost in USD. Unknown model or unknown usage -> None (never 0)."""
    if not model_id:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    for prefix, (inp, out) in PRICING_TABLE.items():
        if model_id.startswith(prefix):
            return input_tokens / 1_000_000 * inp + output_tokens / 1_000_000 * out
    return None


def _extract_usage(usage: Mapping[str, Any] | None, *keys: str) -> tuple[int | None, bool]:
    """Return (tokens, reported). Absent/non-numeric usage -> (None, False).

    A present-but-zero value is measured zero, distinct from unavailable.
    """
    if not isinstance(usage, dict):
        return None, False
    for key in keys:
        if key in usage and usage[key] is not None:
            try:
                return int(usage[key]), True  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None, False
    return None, False


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


def _post_json_with_status(
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POST JSON, preserving HTTP status for error classification.

    Shares _post_json's contract for successes; failures raise
    ProviderHttpError with the observed status (None for transport errors).
    """
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            details = exc.read().decode("utf-8", errors="replace")
        except (OSError, ValueError):
            details = ""
        raise ProviderHttpError(exc.code, f"provider HTTP {exc.code}: {details[:500]}") from exc
    except Exception as exc:  # network/timeout/transport failures
        raise ProviderHttpError(None, f"provider request failed: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderHttpError(None, f"provider returned non-JSON response: {exc}") from exc
    if isinstance(parsed, dict) and parsed.get("error"):
        raise ProviderHttpError(None, f"provider error: {parsed['error']}")
    if not isinstance(parsed, dict):
        raise ProviderHttpError(None, "provider returned unexpected response shape")
    return parsed


def _scrub_payload(value: Any) -> Any:
    """Recursively redact credential-like keys from a decoded provider payload."""
    if isinstance(value, Mapping):
        return {
            str(key): ("[redacted]" if is_credential_key(str(key)) else _scrub_payload(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_payload(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Exact bytes of one HTTP response, before any JSON parsing.

    Used by the OpenCode gateway path so the runtime can preserve provider
    bodies verbatim. Never truncated, never decoded here.
    """

    status: int
    headers: Mapping[str, str]
    body: bytes
    content_type: str | None = None


def _content_type_of(headers: Mapping[str, str]) -> str | None:
    for key, value in headers.items():
        if str(key).lower() == "content-type":
            return str(value).split(";")[0].strip() or None
    return None


def _request_bytes(
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> HttpResponse:
    """POST JSON and return the exact HTTP response (status, headers, bytes).

    HTTP error statuses do NOT raise: an error body is evidence and is
    returned intact. Only the absence of any HTTP response raises, as
    TransportFailure carrying the underlying exception type.
    """
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_headers = {str(k): str(v) for k, v in response.headers.items()}
            return HttpResponse(
                status=int(response.status),
                headers=raw_headers,
                body=response.read(),
                content_type=_content_type_of(raw_headers),
            )
    except urllib.error.HTTPError as exc:
        try:
            raw_headers = (
                {str(k): str(v) for k, v in exc.headers.items()} if exc.headers else {}
            )
        except (OSError, ValueError, AttributeError):
            raw_headers = {}
        try:
            body = exc.read()
        except (OSError, ValueError):
            body = b""
        return HttpResponse(
            status=int(exc.code),
            headers=raw_headers,
            body=body,
            content_type=_content_type_of(raw_headers),
        )
    except Exception as exc:  # network/timeout/transport failures: no response exists
        raise TransportFailure(type(exc).__name__, f"provider request failed: {exc}") from exc


def _parse_http_response(reply: HttpResponse) -> dict[str, Any]:
    """Decode a preserved HTTP response, with the legacy error contract.

    Success and error-message shapes are identical to _post_json_with_status
    so existing classifications do not change; only the evidence path is new.
    """
    if reply.status != 200:
        text = reply.body.decode("utf-8", errors="replace")
        raise ProviderHttpError(reply.status, f"provider HTTP {reply.status}: {text[:500]}")
    try:
        text = reply.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Legacy contract: undecodable 200 bodies surface as transport
        # failures, not parse failures. Preserved verbatim.
        raise ProviderHttpError(None, f"provider request failed: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderHttpError(None, f"provider returned non-JSON response: {exc}") from exc
    if isinstance(parsed, dict) and parsed.get("error"):
        raise ProviderHttpError(None, f"provider error: {parsed['error']}")
    if not isinstance(parsed, dict):
        raise ProviderHttpError(None, "provider returned unexpected response shape")
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

    def effective_request(self, spec: CallSpec) -> dict[str, Any]:
        """Sanitized actually-sent controls for the OpenAI body (no secrets)."""
        effective: dict[str, Any] = {"model": self.model, "endpoint": "chat/completions"}
        temperature = _params(spec, "temperature")
        if temperature is not None:
            effective["temperature"] = temperature
        seed = _params(spec, "seed")
        if seed is not None:
            try:
                effective["seed"] = int(seed)
            except (TypeError, ValueError):
                pass
        return sanitize_effective_params(effective)

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
            raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
            in_tokens, in_reported = _extract_usage(raw_usage, "prompt_tokens")
            out_tokens, out_reported = _extract_usage(raw_usage, "completion_tokens")
            usage_source = "measured" if (in_reported or out_reported) else "unavailable"
            cost = estimate_cost_usd(self.model, in_tokens, out_tokens)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                provider="openai",
                model=self.model,
                model_version=str(parsed.get("model", self.model)),
                fingerprint=parsed.get("system_fingerprint"),
                provider_call_id=parsed.get("id"),
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
                usage_source=usage_source,
                pricing_version=PRICING_VERSION,
                cost_source="estimated" if cost is not None else "unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
            )
        except (MissingCredentialsError, ProviderError) as exc:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                input_tokens=None,
                output_tokens=None,
                usage_source="unavailable",
                error_kind="missing_credentials"
                if isinstance(exc, MissingCredentialsError)
                else "provider_error",
                pricing_version=PRICING_VERSION,
                cost_source="unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
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

    def effective_request(self, spec: CallSpec) -> dict[str, Any]:
        """Sanitized actually-sent controls for the Anthropic body (no secrets)."""
        effective: dict[str, Any] = {
            "model": self.model,
            "endpoint": "messages",
            "max_tokens": int(_params(spec, "max_tokens", 1024)),
        }
        temperature = _params(spec, "temperature")
        if temperature is not None:
            effective["temperature"] = temperature
        return sanitize_effective_params(effective)

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
            raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
            in_tokens, in_reported = _extract_usage(raw_usage, "input_tokens")
            out_tokens, out_reported = _extract_usage(raw_usage, "output_tokens")
            usage_source = "measured" if (in_reported or out_reported) else "unavailable"
            cost = estimate_cost_usd(self.model, in_tokens, out_tokens)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                provider="anthropic",
                model=self.model,
                model_version=str(parsed.get("model", self.model)),
                provider_call_id=parsed.get("id"),
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
                usage_source=usage_source,
                pricing_version=PRICING_VERSION,
                cost_source="estimated" if cost is not None else "unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
            )
        except (MissingCredentialsError, ProviderError) as exc:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                input_tokens=None,
                output_tokens=None,
                usage_source="unavailable",
                error_kind="missing_credentials"
                if isinstance(exc, MissingCredentialsError)
                else "provider_error",
                pricing_version=PRICING_VERSION,
                cost_source="unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
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

    def effective_request(self, spec: CallSpec) -> dict[str, Any]:
        """Sanitized actually-sent controls for the compatible body (no secrets)."""
        effective: dict[str, Any] = {"model": self.model, "endpoint": "chat/completions"}
        temperature = _params(spec, "temperature")
        if temperature is not None:
            effective["temperature"] = temperature
        return sanitize_effective_params(effective)

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
            raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
            in_tokens, in_reported = _extract_usage(raw_usage, "prompt_tokens")
            out_tokens, out_reported = _extract_usage(raw_usage, "completion_tokens")
            usage_source = "measured" if (in_reported or out_reported) else "unavailable"
            cost = estimate_cost_usd(self.model, in_tokens, out_tokens)
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                latency_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider_name,
                model=self.model,
                model_version=str(parsed.get("model", self.model)),
                provider_call_id=parsed.get("id"),
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
                usage_source=usage_source,
                pricing_version=PRICING_VERSION,
                cost_source="estimated" if cost is not None else "unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
            )
        except ProviderError as exc:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                input_tokens=None,
                output_tokens=None,
                usage_source="unavailable",
                error_kind="provider_error",
                pricing_version=PRICING_VERSION,
                cost_source="unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
                status="failed",
                error=str(exc),
                provider=self.provider_name,
                model=self.model,
                started_at=started_at,
                completed_at=_now_iso(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


def _opencode_input_text(spec: CallSpec) -> str:
    instruction = spec.instruction or ""
    prompt = spec.context.prompt or ""
    return f"{instruction}\n\n{prompt}".strip() or prompt or instruction


def _responses_text(parsed: dict[str, Any]) -> str:
    """Extract canonical text from a Responses-API payload.

    Raises ProviderError on malformed output (never silently empty).
    """
    output = parsed.get("output")
    if not isinstance(output, list):
        raise ProviderError("OpenCode Responses payload contained no output items")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            continue  # thinking trace: not model output, skip without failing
        for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                texts.append(str(block.get("text", "")))
    if not texts:
        raise ProviderError("OpenCode Responses payload contained no output text")
    return "".join(texts)


def _classify_opencode_error(exc: ProviderError) -> str:
    """Map transport/HTTP failures to stable error kinds (never model text)."""
    if isinstance(exc, MissingCredentialsError):
        return "missing_credentials"
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    status = exc.status if isinstance(exc, ProviderHttpError) else None
    if status in (401, 403):
        return "authentication_error"
    if status == 429:
        return "rate_limited"
    if status is not None and status >= 500:
        return "provider_error"
    if "non-json" in message or "unexpected response shape" in message:
        return "malformed_response"
    return "provider_error"


# Concrete routes through the OpenCode gateway. Protocol is a property of the
# resolved model route, not of the gateway: e.g. mimo-v2.5 is served on Chat
# Completions, while other occupants use Responses or Messages.
OPENCODE_ENDPOINTS = {
    "responses": "/v1/responses",
    "chat_completions": "/v1/chat/completions",
}


def _chat_text(parsed: dict[str, Any]) -> str:
    """Extract canonical text from a Chat-Completions payload.

    Raises ProviderError on malformed output (never silently empty).
    """
    choices = parsed.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        raise ProviderError("OpenCode Chat Completions payload contained no choices")
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        raise ProviderError("OpenCode Chat Completions payload contained no message")
    content = message.get("content", "")
    if isinstance(content, list):  # structured content parts
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not text:
            raise ProviderError("OpenCode Chat Completions payload contained no output text")
        return text
    if not content:
        raise ProviderError("OpenCode Chat Completions payload contained no output text")
    return str(content)


@dataclass
class OpenCodeCognitionAdapter(CognitionAdapter):
    """OpenCode Zen gateway as a first-class cognition adapter.

    Gateway identity is always ``opencode``; the wire dialect is carried
    separately in ``protocol`` (``responses`` or ``chat_completions``) and
    comes from the resolved model route — it is never inferred from the
    model name. An OpenAI-compatible endpoint on this gateway must NOT be
    recorded as provider ``openai``.

    Credentials come from ``OPENCODE_ZEN_API_KEY`` (or an explicit api_key)
    only. ``OPENAI_API_KEY`` is never consulted here.
    """

    GATEWAY = "opencode"

    model: str = "mimo-v2.5"
    api_key: str | None = None
    base_url: str = OPENCODE_ZEN_BASE_URL
    protocol: str = "responses"
    timeout: float = 120.0
    # Test seam: may return a parsed dict (legacy fixture shape, no transport
    # evidence) or an HttpResponse (exact bytes preserved as evidence).
    http_post: Callable[..., dict[str, Any] | HttpResponse] | None = None
    user_agent: str = "codeai/0.1.0"
    # Go-gateway routing token sent as x-opencode-session. Explicit when set;
    # otherwise generated per invoke and recorded on the attempt (transport
    # routing, not a secret and not a generation control).
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv(OPENCODE_ZEN_API_KEY_ENV)
        if self.protocol not in OPENCODE_ENDPOINTS:
            supported = sorted(OPENCODE_ENDPOINTS)
            raise ValueError(
                f"unsupported OpenCode protocol '{self.protocol}'; currently supported: {supported}"
            )

    def _require_key(self) -> str:
        if not self.api_key:
            raise MissingCredentialsError(
                f"OpenCode credentials absent: set {OPENCODE_ZEN_API_KEY_ENV} or pass api_key"
            )
        return self.api_key

    def _endpoint(self) -> str:
        return OPENCODE_ENDPOINTS[self.protocol]

    def _resolve_session(self) -> str:
        if self.session_id:
            return self.session_id
        import uuid as _uuid

        return f"codeai-{_uuid.uuid4().hex[:16]}"

    def _request_body(self, spec: CallSpec) -> dict[str, Any]:
        """Build the protocol-specific request body for the resolved route."""
        if self.protocol == "chat_completions":
            body: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": _opencode_input_text(spec)}],
            }
            # Chat route controls: only what this dialect supports. A
            # Responses-style reasoning.effort is never sent here.
            max_tokens = _params(spec, "max_tokens", _params(spec, "max_output_tokens"))
            if max_tokens is not None:
                try:
                    body["max_tokens"] = int(max_tokens)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    pass
            temperature = _params(spec, "temperature")
            if temperature is not None:
                body["temperature"] = temperature
            return body
        body = {"model": self.model, "input": _opencode_input_text(spec)}
        reasoning_effort = _params(spec, "reasoning_effort")
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        max_output = _params(spec, "max_output_tokens", _params(spec, "max_tokens"))
        if max_output is not None:
            try:
                body["max_output_tokens"] = int(max_output)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        temperature = _params(spec, "temperature")
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def effective_request(self, spec: CallSpec) -> dict[str, Any]:
        """Sanitized actually-sent controls for the resolved route (no secrets)."""
        effective: dict[str, Any] = {
            "gateway": self.GATEWAY,
            "protocol": self.protocol,
            "endpoint": self._endpoint(),
            "model": self.model,
        }
        if self.session_id:
            effective["session_id"] = self.session_id
        if self.protocol == "chat_completions":
            max_tokens = _params(spec, "max_tokens", _params(spec, "max_output_tokens"))
            if max_tokens is not None:
                effective["max_tokens"] = max_tokens
            temperature = _params(spec, "temperature")
            if temperature is not None:
                effective["temperature"] = temperature
            return sanitize_effective_params(effective)
        reasoning_effort = _params(spec, "reasoning_effort")
        if reasoning_effort is not None:
            effective["reasoning_effort"] = reasoning_effort
        max_output = _params(spec, "max_output_tokens", _params(spec, "max_tokens"))
        if max_output is not None:
            effective["max_output_tokens"] = max_output
        temperature = _params(spec, "temperature")
        if temperature is not None:
            effective["temperature"] = temperature
        return sanitize_effective_params(effective)

    def invoke(self, spec: CallSpec) -> CallResult:
        started = time.perf_counter()
        started_at = _now_iso()
        transport: TransportObservation | None = None
        try:
            key = self._require_key()
            if not self.base_url:
                raise ProviderError("OpenCode base_url is required")
            body = self._request_body(spec)
            session_id = self._resolve_session()
            post = self.http_post or _request_bytes
            reply = post(
                f"{self.base_url.rstrip('/')}{self._endpoint()}",
                body,
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    "User-Agent": self.user_agent,
                    "x-opencode-session": session_id,
                },
                self.timeout,
            )
            if isinstance(reply, HttpResponse):
                transport = TransportObservation(
                    outcome=(
                        TransportOutcome.RESPONSE_RECEIVED.value
                        if reply.status == 200
                        else TransportOutcome.HTTP_ERROR.value
                    ),
                    status_code=reply.status,
                    body=reply.body,
                    headers=dict(reply.headers),
                    content_type=reply.content_type,
                    endpoint=self._endpoint(),
                    observed_at=_now_iso(),
                )
                parsed = _parse_http_response(reply)
            else:
                # Legacy fixture shape: parsed JSON only, no transport bytes.
                parsed = reply
            if self.protocol == "chat_completions":
                text = _chat_text(parsed)
                raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
                in_tokens, in_reported = _extract_usage(raw_usage, "prompt_tokens")
                out_tokens, out_reported = _extract_usage(raw_usage, "completion_tokens")
            else:
                text = _responses_text(parsed)
                raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
                in_tokens, in_reported = _extract_usage(raw_usage, "input_tokens")
                out_tokens, out_reported = _extract_usage(raw_usage, "output_tokens")
            usage_source = "measured" if (in_reported or out_reported) else "unavailable"
            cost = estimate_cost_usd(self.model, in_tokens, out_tokens)
            reported_model = parsed.get("model")
            effective = self.effective_request(spec)
            effective["session_id"] = session_id  # actual routing token used
            return CallResult(
                call_id=spec.call_id,
                raw_output=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                latency_ms=int((time.perf_counter() - started) * 1000),
                provider=self.GATEWAY,
                model=self.model,
                model_version=str(reported_model) if reported_model is not None else None,
                provider_call_id=parsed.get("id") if isinstance(parsed.get("id"), str) else None,
                started_at=started_at,
                completed_at=_now_iso(),
                status="succeeded",
                usage_source=usage_source,
                pricing_version=PRICING_VERSION,
                cost_source="estimated" if cost is not None else "unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=effective,
                protocol=self.protocol,
                raw_observation_kind="decoded_json",
                raw_payload=_scrub_payload(parsed),
                transport=transport,
            )
        except ProviderError as exc:
            if transport is None and isinstance(exc, TransportFailure):
                transport = TransportObservation(
                    outcome=TransportOutcome.NO_RESPONSE.value,
                    status_code=None,
                    body=None,
                    headers={},
                    content_type=None,
                    endpoint=self._endpoint(),
                    exception_type=exc.exception_type,
                    observed_at=_now_iso(),
                )
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                input_tokens=None,
                output_tokens=None,
                usage_source="unavailable",
                error_kind=_classify_opencode_error(exc),
                pricing_version=PRICING_VERSION,
                cost_source="unknown",
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
                protocol=self.protocol,
                raw_observation_kind="decoded_json",
                transport=transport,
                status="failed",
                error=str(exc),
                provider=self.GATEWAY,
                model=self.model,
                started_at=started_at,
                completed_at=_now_iso(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
