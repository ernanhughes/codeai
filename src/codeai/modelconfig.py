from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import (
    OPENCODE_ZEN_API_KEY_ENV,
    OPENCODE_ZEN_BASE_URL,
    AnthropicAdapter,
    OpenAIAdapter,
    OpenAICompatibleAdapter,
    OpenCodeCognitionAdapter,
)


@dataclass(frozen=True, slots=True)
class ModelMapping:
    logical_name: str
    adapter: str  # "openai" | "anthropic" | "openai-compatible" | "opencode" | "fake"
    model: str
    base_url: str | None = None
    provider: str | None = None
    # Wire dialect for gateways that support several, e.g. OpenCode Zen:
    # "responses" | "chat_completions" | "messages". None = adapter default.
    protocol: str | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    models: dict[str, ModelMapping] = field(default_factory=dict)

    def resolve(self, logical_name: str) -> ModelMapping:
        if logical_name not in self.models:
            raise KeyError(
                f"unknown model '{logical_name}'; configured: {sorted(self.models) or 'none'}"
            )
        return self.models[logical_name]

    def resolve_chamber(self, chamber: str) -> ModelMapping:
        """Resolve a chamber (logical job name) to its concrete occupant.

        A chamber is named by its job (e.g. "deep-review"), not by its model.
        This reuses the single logical-model mapping; there is no second
        registry. Raises KeyError for unconfigured chambers.
        """
        return self.resolve(chamber)

    def build_adapter(self, logical_name: str) -> Any:
        """Build a CognitionAdapter for a logical model name. Secrets from env only."""
        mapping = self.resolve(logical_name)
        if mapping.adapter == "openai":
            return OpenAIAdapter(
                model=mapping.model,
                base_url=mapping.base_url or "https://api.openai.com/v1",
            )
        if mapping.adapter == "anthropic":
            return AnthropicAdapter(
                model=mapping.model,
                base_url=mapping.base_url or "https://api.anthropic.com/v1",
            )
        if mapping.adapter == "openai-compatible":
            return OpenAICompatibleAdapter(
                model=mapping.model,
                base_url=mapping.base_url or "http://localhost:11434/v1",
                provider_name=mapping.provider or "openai-compatible",
            )
        if mapping.adapter == "opencode":
            return OpenCodeCognitionAdapter(
                model=mapping.model,
                base_url=mapping.base_url or OPENCODE_ZEN_BASE_URL,
                protocol=mapping.protocol or "responses",
            )
        if mapping.adapter == "fake":
            from .adapters import FakeCognitionAdapter

            return FakeCognitionAdapter(model=mapping.model)
        raise ValueError(f"unknown adapter '{mapping.adapter}' for model '{logical_name}'")


def default_config_path(cwd: str | Path | None = None) -> Path:
    return Path(cwd or Path.cwd()) / ".codeai" / "config.toml"


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    """Load logical model mappings. Missing file -> empty config (fake only)."""
    resolved = Path(path) if path else default_config_path()
    if not resolved.exists():
        return ModelConfig()
    with resolved.open("rb") as handle:
        data = tomllib.load(handle)
    models: dict[str, ModelMapping] = {}
    raw_models = data.get("models", {})
    if not isinstance(raw_models, dict):
        raise TypeError(".codeai/config.toml: [models.*] must be a table")
    for logical_name, entry in raw_models.items():
        if not isinstance(entry, dict):
            raise TypeError(f".codeai/config.toml: models.{logical_name} must be a table")
        models[str(logical_name)] = ModelMapping(
            logical_name=str(logical_name),
            adapter=str(entry.get("adapter", "fake")),
            model=str(entry.get("model", logical_name)),
            base_url=entry.get("base_url"),
            provider=entry.get("provider"),
            protocol=entry.get("protocol"),
        )
    return ModelConfig(models=models)


EXAMPLE_CONFIG = """\
# Logical model names for experiments. No secrets here: keys come from env.
# OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_COMPAT_API_KEY, OPENCODE_ZEN_API_KEY
[models.deep-review]
adapter = "opencode"
model = "mimo-v2.5"
protocol = "responses"

[models.qwen]
adapter = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen"

[models.claude]
adapter = "anthropic"
model = "claude-haiku-4-5-20251001"

[models.gpt]
adapter = "openai"
model = "gpt-4o-mini"
"""


def missing_credentials(mapping: ModelMapping) -> str | None:
    """Return a human-readable missing-credential hint, or None if configured."""
    if mapping.adapter == "openai" and not os.getenv("OPENAI_API_KEY"):
        return "set OPENAI_API_KEY"
    if mapping.adapter == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        return "set ANTHROPIC_API_KEY"
    if mapping.adapter == "opencode" and not os.getenv(OPENCODE_ZEN_API_KEY_ENV):
        return f"set {OPENCODE_ZEN_API_KEY_ENV}"
    return None
