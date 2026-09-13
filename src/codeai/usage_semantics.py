"""Versioned usage semantics: what reported token counts mean.

Observation != interpretation != accounting. This module interprets the
``usage`` member of a preserved provider response under a named, versioned
rule. It never changes recorded usage, decisions or bytes, and it never
produces a bill.

``usage-semantics-v1`` reproduces the historical adapter view: input and
output counts only, extracted exactly as the adapters extract them.

``usage-semantics-v2`` interprets component by component under documented
dialect semantics:

- equivalent quantities are normalized into named components;
- related but non-equivalent quantities are preserved separately, with an
  explicit relation (subset of input, additive to input, subset of output);
- fields without a rule are preserved as unrecognized paths, never guessed;
- a provider-reported total is kept as reported and compared with the
  derived total; neither replaces the other;
- anything that cannot be derived is UNKNOWN (``None``), optionally with a
  known lower bound, and never zero.

Dialect rules describe what the API family documents. They do not establish
that a particular route behind a gateway honours them, so every v2 result
records ``route_conformance = "unverified"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

USAGE_SEMANTICS_V1 = "usage-semantics-v1"
USAGE_SEMANTICS_V2 = "usage-semantics-v2"

COMPONENT_NAMES = ("input", "output", "cache_read", "cache_write", "reasoning", "total_reported")
DERIVED_NAMES = ("total_input", "fresh_input", "processed_total")


class ComponentStatus(StrEnum):
    REPORTED = "reported"
    NOT_REPORTED = "not_reported"
    INVALID = "invalid"


class Relation(StrEnum):
    PRIMARY = "primary"
    SUBSET_OF_INPUT = "subset_of_input"
    ADDITIVE_TO_INPUT = "additive_to_input"
    SUBSET_OF_OUTPUT = "subset_of_output"
    REPORTED_TOTAL = "reported_total"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Component:
    """One reported quantity, its JSON path, validity and declared relation."""

    name: str
    value: int | None
    status: str
    path: str | None
    relation: str = Relation.UNKNOWN.value


@dataclass(frozen=True, slots=True)
class Derived:
    """A quantity computed from components under a named rule, or UNKNOWN."""

    name: str
    value: int | None
    lower_bound: int | None
    rule: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class UsageInterpretation:
    version: str
    protocol: str | None
    rule_id: str | None
    rule_source: str | None
    route_conformance: str
    usage_present: bool
    components: tuple[Component, ...] = ()
    derived: tuple[Derived, ...] = ()
    conflicts: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    unrecognized_paths: tuple[str, ...] = ()
    provider_cost: Mapping[str, Any] = field(default_factory=dict)
    estimate: Mapping[str, Any] | None = None

    def component(self, name: str) -> Component | None:
        return next((c for c in self.components if c.name == name), None)

    def derived_value(self, name: str) -> Derived | None:
        return next((d for d in self.derived if d.name == name), None)


# (container, key, relation); container None means a top-level usage key.
_Spec = tuple[str | None, str, str] | None

_RULES: dict[str, dict[str, Any]] = {
    "responses": {
        "rule_id": "openai-responses-usage-v1",
        "rule_source": (
            "OpenAI API docs, prompt caching and reasoning guides: cached_tokens and "
            "cache_write_tokens are components of input_tokens; reasoning_tokens are "
            "billed within output_tokens"
        ),
        "input_includes_cache": True,
        "input": (None, "input_tokens", Relation.PRIMARY.value),
        "output": (None, "output_tokens", Relation.PRIMARY.value),
        "total_reported": (None, "total_tokens", Relation.REPORTED_TOTAL.value),
        "cache_read": ("input_tokens_details", "cached_tokens", Relation.SUBSET_OF_INPUT.value),
        "cache_write": (
            "input_tokens_details",
            "cache_write_tokens",
            Relation.SUBSET_OF_INPUT.value,
        ),
        "reasoning": ("output_tokens_details", "reasoning_tokens", Relation.SUBSET_OF_OUTPUT.value),
    },
    "chat_completions": {
        "rule_id": "openai-chat-usage-v1",
        "rule_source": (
            "OpenAI Responses usage semantics applied by analogy to Chat "
            "prompt_tokens_details / completion_tokens_details (not separately verified)"
        ),
        "input_includes_cache": True,
        "input": (None, "prompt_tokens", Relation.PRIMARY.value),
        "output": (None, "completion_tokens", Relation.PRIMARY.value),
        "total_reported": (None, "total_tokens", Relation.REPORTED_TOTAL.value),
        "cache_read": ("prompt_tokens_details", "cached_tokens", Relation.SUBSET_OF_INPUT.value),
        "cache_write": (
            "prompt_tokens_details",
            "cache_write_tokens",
            Relation.SUBSET_OF_INPUT.value,
        ),
        "reasoning": (
            "completion_tokens_details",
            "reasoning_tokens",
            Relation.SUBSET_OF_OUTPUT.value,
        ),
    },
    "messages": {
        "rule_id": "anthropic-messages-usage-v1",
        "rule_source": (
            "Anthropic API docs, prompt caching and extended thinking: input_tokens excludes "
            "cache reads and writes; total input = cache_read_input_tokens + "
            "cache_creation_input_tokens + input_tokens; thinking_tokens are billed within "
            "output_tokens"
        ),
        "input_includes_cache": False,
        "input": (None, "input_tokens", Relation.PRIMARY.value),
        "output": (None, "output_tokens", Relation.PRIMARY.value),
        "total_reported": None,
        "cache_read": (None, "cache_read_input_tokens", Relation.ADDITIVE_TO_INPUT.value),
        "cache_write": (None, "cache_creation_input_tokens", Relation.ADDITIVE_TO_INPUT.value),
        "reasoning": ("output_tokens_details", "thinking_tokens", Relation.SUBSET_OF_OUTPUT.value),
    },
}


def _path(container: str | None, key: str) -> str:
    return f"usage.{container}.{key}" if container else f"usage.{key}"


def _read_component(
    usage: Mapping[str, Any] | None, name: str, spec: _Spec, diagnostics: list[str]
) -> Component:
    if spec is None:
        return Component(name, None, ComponentStatus.NOT_REPORTED.value, None)
    container, key, relation = spec
    path = _path(container, key)
    source: Any = usage
    if container is not None and isinstance(usage, Mapping):
        source = usage.get(container)
        if source is not None and not isinstance(source, Mapping):
            diagnostics.append(f"invalid_container:usage.{container}")
            return Component(name, None, ComponentStatus.INVALID.value, path, relation)
    if not isinstance(source, Mapping) or key not in source:
        return Component(name, None, ComponentStatus.NOT_REPORTED.value, path, relation)
    raw = source[key]
    if raw is None:
        diagnostics.append(f"null_value:{path}")
        return Component(name, None, ComponentStatus.NOT_REPORTED.value, path, relation)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        diagnostics.append(f"invalid_value:{path}")
        return Component(name, None, ComponentStatus.INVALID.value, path, relation)
    return Component(name, raw, ComponentStatus.REPORTED.value, path, relation)


def _leaf_paths(usage: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key, value in usage.items():
        if isinstance(value, Mapping):
            paths.extend(f"usage.{key}.{inner}" for inner in value)
        else:
            paths.append(f"usage.{key}")
    return sorted(paths)


def _provider_cost(parsed: Mapping[str, Any]) -> dict[str, Any]:
    if "cost" not in parsed:
        return {}
    return {"path": "cost", "raw": parsed["cost"], "accounting": "unknown"}


def _reasoning_content_present(protocol: str | None, parsed: Mapping[str, Any]) -> bool:
    if protocol == "chat_completions":
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping):
                return any(
                    isinstance(message.get(key), str) and bool(message.get(key))
                    for key in ("reasoning", "reasoning_content")
                )
        return False
    if protocol == "messages":
        content = parsed.get("content")
        return isinstance(content, list) and any(
            isinstance(block, Mapping)
            and block.get("type") == "thinking"
            and bool(block.get("thinking"))
            for block in content
        )
    if protocol == "responses":
        output = parsed.get("output")
        return isinstance(output, list) and any(
            isinstance(item, Mapping)
            and item.get("type") == "reasoning"
            and bool(item.get("summary") or item.get("content"))
            for item in output
        )
    return False


def _unknown(name: str, rule: str, reason: str, lower_bound: int | None = None) -> Derived:
    return Derived(name, None, lower_bound, rule, reason)


def _sum_known(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _interpret_v1(protocol: str | None, parsed: Mapping[str, Any]) -> UsageInterpretation:
    """Reproduce the adapters' historical extraction: input/output, int-coerced."""
    usage = parsed.get("usage") if isinstance(parsed, Mapping) else None
    present = isinstance(usage, Mapping)
    in_key, out_key = (
        ("prompt_tokens", "completion_tokens")
        if protocol == "chat_completions"
        else ("input_tokens", "output_tokens")
    )
    components = []
    for name, key in (("input", in_key), ("output", out_key)):
        value: int | None = None
        status = ComponentStatus.NOT_REPORTED.value
        if present and usage.get(key) is not None:
            try:
                value = int(usage[key])
                status = ComponentStatus.REPORTED.value
            except (TypeError, ValueError):
                value = None
        components.append(Component(name, value, status, _path(None, key), Relation.PRIMARY.value))
    return UsageInterpretation(
        version=USAGE_SEMANTICS_V1,
        protocol=protocol,
        rule_id="adapter-extraction-v1",
        rule_source="CodeAI adapter _extract_usage (historical view)",
        route_conformance="not_assessed",
        usage_present=present,
        components=tuple(components),
    )


def _interpret_v2(protocol: str | None, parsed: Mapping[str, Any]) -> UsageInterpretation:
    usage_raw = parsed.get("usage") if isinstance(parsed, Mapping) else None
    present = isinstance(usage_raw, Mapping)
    usage: Mapping[str, Any] | None = usage_raw if present else None
    diagnostics: list[str] = []
    conflicts: list[str] = []
    if usage_raw is None:
        diagnostics.append("usage_not_reported")
    elif not present:
        diagnostics.append("invalid_container:usage")
    cost = _provider_cost(parsed) if isinstance(parsed, Mapping) else {}
    rule = _RULES.get(protocol or "")

    if rule is None:
        return UsageInterpretation(
            version=USAGE_SEMANTICS_V2,
            protocol=protocol,
            rule_id=None,
            rule_source=None,
            route_conformance="unverified",
            usage_present=present,
            derived=tuple(
                _unknown(name, "none", "no semantic rule for this protocol")
                for name in DERIVED_NAMES
            ),
            diagnostics=tuple(diagnostics + ["no_semantic_rule_for_protocol"]),
            unrecognized_paths=tuple(_leaf_paths(usage)) if usage is not None else (),
            provider_cost=cost,
        )

    comps = {name: _read_component(usage, name, rule[name], diagnostics) for name in COMPONENT_NAMES}
    usable = {name: c.status == ComponentStatus.REPORTED.value for name, c in comps.items()}

    def value(name: str) -> int | None:
        return comps[name].value if usable[name] else None

    # Relationship checks: a declared subset may not exceed its parent.
    if rule["input_includes_cache"]:
        parts = [v for v in (value("cache_read"), value("cache_write")) if v is not None]
        if value("input") is not None and parts and sum(parts) > comps["input"].value:
            conflicts.append("cache_components_exceed_input")
            usable["cache_read"] = usable["cache_write"] = False
    if (
        value("reasoning") is not None
        and value("output") is not None
        and comps["reasoning"].value > comps["output"].value
    ):
        conflicts.append("reasoning_exceeds_output")
        usable["reasoning"] = False

    i, o, r, w, t = (value(n) for n in ("input", "output", "cache_read", "cache_write", "total_reported"))
    derived: list[Derived] = []
    if rule["input_includes_cache"]:
        subset_rule = "input includes cache reads and writes; output includes reasoning"
        derived.append(
            Derived("total_input", i, i, "total_input = input")
            if i is not None
            else _unknown("total_input", "total_input = input", "input not reported or invalid")
        )
        if i is not None and r is not None and w is not None:
            derived.append(
                Derived("fresh_input", i - r - w, i - r - w, "fresh_input = input - cache_read - cache_write")
            )
        else:
            derived.append(
                _unknown(
                    "fresh_input",
                    "fresh_input = input - cache_read - cache_write",
                    "input or a cache component not reported or invalid",
                )
            )
        if i is not None and o is not None:
            derived.append(Derived("processed_total", i + o, i + o, f"input + output ({subset_rule})"))
        else:
            derived.append(
                _unknown(
                    "processed_total",
                    f"input + output ({subset_rule})",
                    "input or output not reported or invalid",
                    _sum_known([i, o]),
                )
            )
    else:
        additive_rule = "input excludes cache; cache reads and writes are added to input"
        if i is not None and r is not None and w is not None:
            derived.append(
                Derived("total_input", i + r + w, i + r + w, "total_input = input + cache_read + cache_write")
            )
        else:
            derived.append(
                _unknown(
                    "total_input",
                    "total_input = input + cache_read + cache_write",
                    "a cache component was not reported; absence is not zero",
                    _sum_known([i, r, w]),
                )
            )
        derived.append(
            Derived("fresh_input", i, i, "fresh_input = input (input excludes cache by definition)")
            if i is not None
            else _unknown("fresh_input", "fresh_input = input", "input not reported or invalid")
        )
        if None not in (i, r, w, o):
            derived.append(
                Derived("processed_total", i + r + w + o, i + r + w + o, f"input + cache + output ({additive_rule})")
            )
        else:
            derived.append(
                _unknown(
                    "processed_total",
                    f"input + cache + output ({additive_rule})",
                    "a component was not reported; absence is not zero",
                    _sum_known([i, r, w, o]),
                )
            )

    processed = next(d for d in derived if d.name == "processed_total")
    if t is not None and processed.value is not None and t != processed.value:
        conflicts.append("reported_total_differs_from_derived")

    if _reasoning_content_present(protocol, parsed):
        if value("reasoning") == 0:
            diagnostics.append("reasoning_tokens_zero_with_reasoning_content")
        elif comps["reasoning"].status == ComponentStatus.NOT_REPORTED.value:
            diagnostics.append("reasoning_content_present_reasoning_tokens_not_reported")

    recognized = {c.path for c in comps.values() if c.path is not None}
    unrecognized = [p for p in _leaf_paths(usage) if p not in recognized] if usage is not None else []

    return UsageInterpretation(
        version=USAGE_SEMANTICS_V2,
        protocol=protocol,
        rule_id=rule["rule_id"],
        rule_source=rule["rule_source"],
        route_conformance="unverified",
        usage_present=present,
        components=tuple(comps[name] for name in COMPONENT_NAMES),
        derived=tuple(derived),
        conflicts=tuple(conflicts),
        diagnostics=tuple(diagnostics),
        unrecognized_paths=tuple(unrecognized),
        provider_cost=cost,
    )


def interpret_usage(
    protocol: str | None,
    parsed: Mapping[str, Any],
    *,
    version: str = USAGE_SEMANTICS_V2,
    estimate: Mapping[str, Any] | None = None,
) -> UsageInterpretation:
    """Interpret the usage member of one decoded provider response.

    ``estimate`` is carried verbatim for inspection; it never fills a
    reported component or a derived value. Unknown versions fail loudly.
    """
    if not isinstance(parsed, Mapping):
        parsed = {}
    if version == USAGE_SEMANTICS_V1:
        result = _interpret_v1(protocol, parsed)
    elif version == USAGE_SEMANTICS_V2:
        result = _interpret_v2(protocol, parsed)
    else:
        raise ValueError(f"unknown usage semantics version: {version!r}")
    if estimate is None:
        return result
    return UsageInterpretation(
        **{**{f: getattr(result, f) for f in result.__dataclass_fields__}, "estimate": dict(estimate)}
    )
