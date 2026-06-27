"""Prompt Skill scripts.

Each skill is an independent Python script and uses Claude-style XML annotation
blocks inside the prompt.
"""

from skills.full_skill import (
    build_final_messages as build_full_final_messages,
    build_form_messages as build_full_form_messages,
    build_messages as build_full_skill_messages,
)
from skills.tools_kb_skill import (
    build_final_messages as build_tools_kb_final_messages,
    build_form_messages as build_tools_kb_form_messages,
    build_messages as build_tools_kb_messages,
)
from skills.tools_only_skill import (
    build_final_messages as build_tools_only_final_messages,
    build_form_messages as build_tools_only_form_messages,
    build_messages as build_tools_only_messages,
)

SKILL_BUILDERS = {
    "full": {
        "legacy": build_full_skill_messages,
        "form": build_full_form_messages,
        "final": build_full_final_messages,
    },
    "full_skill": {
        "legacy": build_full_skill_messages,
        "form": build_full_form_messages,
        "final": build_full_final_messages,
    },
    "tools_kb": {
        "legacy": build_tools_kb_messages,
        "form": build_tools_kb_form_messages,
        "final": build_tools_kb_final_messages,
    },
    "tools+kb": {
        "legacy": build_tools_kb_messages,
        "form": build_tools_kb_form_messages,
        "final": build_tools_kb_final_messages,
    },
    "tools_only": {
        "legacy": build_tools_only_messages,
        "form": build_tools_only_form_messages,
        "final": build_tools_only_final_messages,
    },
    "tools-only": {
        "legacy": build_tools_only_messages,
        "form": build_tools_only_form_messages,
        "final": build_tools_only_final_messages,
    },
}

__all__ = ["SKILL_BUILDERS"]
