"""Small JSON helpers used by CLI, tools, and LLM parsing."""

from __future__ import annotations

import json
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize data as stable UTF-8 JSON for prompts and logs."""

    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


# patient_profile 区块里只有 data 是工具/LLM 要读的；raw（原始行，审计用）、field_index
# 与 data_quality 不进运行时决策，且 raw 往往占 profile 体积的 ~90%。渲染进 prompt 时剔除，
# 避免无谓撑大上下文。
_PROFILE_PROMPT_DROP = ("raw", "field_index", "data_quality")


def prompt_context(context: Any) -> Any:
    """Return a prompt-facing copy of a generation context with the non-decision,
    bulky parts of patient_profile (raw rows / field_index / data_quality) removed."""

    if not isinstance(context, dict):
        return context
    profile = context.get("patient_profile")
    if not isinstance(profile, dict) or not any(key in profile for key in _PROFILE_PROMPT_DROP):
        return context
    slim_profile = {key: value for key, value in profile.items() if key not in _PROFILE_PROMPT_DROP}
    return {**context, "patient_profile": slim_profile}


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def extract_json_object(text: str) -> Any:
    """Parse a JSON object from raw model text.

    The model is instructed to return only JSON, but this function tolerates
    fenced blocks or short prose around the object so regeneration can be
    handled by guardrails instead of crashing the whole pipeline.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])

