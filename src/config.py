"""Configuration loading for replaceable LLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"
    suffix: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    model: str = "mock-baduanjin-prescriber"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 60
    json_parse_retries: int = 2
    max_regenerations: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "")
        return ""


def load_llm_config(path: str | None = None) -> LLMConfig:
    """Load LLM settings from JSON.

    The config file is intentionally provider-light: any OpenAI-compatible
    endpoint can be used by changing base_url, api key environment variable,
    model, and temperature.
    """

    if path is None:
        path = os.getenv("BADUANJIN_LLM_CONFIG", "src/config/llm_config.example.json")

    with open(path, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    allowed = set(LLMConfig.__dataclass_fields__.keys())
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return LLMConfig(**filtered)
