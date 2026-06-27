"""Tool 4: action matching — project Tool 3 risks onto per-form and global parameters.

Tool 4 is a pure matcher (KB4 查表): it consumes Tool 3's enumerated risk list and,
for each risk, looks up the KB4 mapping table to emit per-form parameter exclusions
(hard_constraints), soft preferences, and global prescription caps (global_constraints).
It does NOT detect risks from the raw profile — that is Tool 3's job (评估 vs 匹配).
"""

from __future__ import annotations

import json
import os

from typing import Any

ALL = "all"

# KB4 mapping: risk_id -> list of projections.
#   hard: per-form parameter value exclusion (enters the feasible-region intersection)
#   soft: per-form preference (advisory, not a hard exclusion)
#   dose: global prescription cap (cycle_decrement / frequency_per_week /
#         single_session_max_minutes)
#   note: decision note that must be visible to downstream generation/guardrail,
#         without mechanically changing a numeric parameter.
def _load_match_table() -> dict[str, list[dict[str, Any]]]:
    """Load risk->projection rules from the externalised JSON (single source of
    truth; Tool 5's restricted-action enforcement reads the same file). Human-
    maintained source: data/知识库/带 CRF 字段的受限表格.xlsx — 改表后需重新生成 JSON。"""

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "restricted_action_rules.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["rules"]


MATCH_TABLE: dict[str, list[dict[str, Any]]] = _load_match_table()

# Risks whose data is incomplete for fine-grained matching -> recorded as unresolved.
UNRESOLVED: dict[str, dict[str, Any]] = {
    "body_pain": {
        "signal": "pain_location_missing",
        "detail": "疼痛只有总分，缺少肩/颈/腰/膝部位，不能逐部位精细匹配。",
        "affected_forms": ALL,
    },
}


def build_action_limitation_profile(risk_profile: dict[str, Any]) -> dict[str, Any]:
    risks = risk_profile.get("risks") or []

    hard_constraints: list[dict[str, Any]] = []
    soft_preferences: list[dict[str, Any]] = []
    global_constraints: dict[str, Any] = {}
    global_notes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for risk in risks:
        key = risk.get("risk_id")
        for proj in MATCH_TABLE.get(key, []):
            kind = proj["kind"]
            if kind == "hard":
                hard_constraints.append(
                    {
                        "source": key,
                        "forms": proj["forms"],
                        "parameter": proj["parameter"],
                        "disallow": proj["disallow"],
                        "reason": proj["reason"],
                    }
                )
            elif kind == "soft":
                soft_preferences.append(
                    {
                        "source": key,
                        "forms": proj["forms"],
                        "parameter": proj["parameter"],
                        "prefer": proj["prefer"],
                        "reason": proj["reason"],
                    }
                )
            elif kind == "dose":
                _apply_global_constraint(global_constraints, proj["dose"], proj["value"])
                global_notes.append(
                    {"type": "global_safety_cap", "detail": proj["reason"], "affected": "global"}
                )
            elif kind == "note":
                # Advisory note (no movement/dose change). Use the projection's own
                # reason as the canonical text — the patient-specific value stays
                # traceable via the Tool 3 risk (source), so no concatenation here.
                global_notes.append(
                    {"type": proj["type"], "detail": proj["reason"], "affected": "global"}
                )
        if key in UNRESOLVED:
            unresolved.append(UNRESOLVED[key])

    return {
        "hard_constraints": hard_constraints,
        "soft_preferences": soft_preferences,
        "global_constraints": global_constraints,
        "annotations": global_notes,
        "unresolved": unresolved,
    }


def _apply_global_constraint(global_constraints: dict[str, Any], key: str, value: Any) -> None:
    """Merge a global cap, keeping the most conservative when several risks collide."""

    if key == "cycle_decrement":
        global_constraints[key] = max(int(global_constraints.get(key, 0)), int(value))
    elif key == "single_session_max_minutes":
        existing = global_constraints.get(key)
        global_constraints[key] = int(value) if existing is None else min(int(existing), int(value))
    elif key == "frequency_per_week":
        existing = global_constraints.get(key)
        if existing is None:
            global_constraints[key] = list(value)
        else:
            global_constraints[key] = [item for item in existing if item in set(value)] or list(value)
    else:
        global_constraints[key] = value
