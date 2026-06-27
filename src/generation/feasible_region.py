"""Deterministic feasible-region integration (constrained-generation layer).

Merges Tool 1/2/3/4 constraints into the per-form feasible region the LLM must
generate within. This is part of the constrained-generation scaffolding (see
generation/__init__.py), not a numbered Tool 0-5.
"""

from __future__ import annotations

from typing import Any

from constants import (
    BADUANJIN_FORMS,
    BASE_AMPLITUDE_LEVELS,
    MAX_CYCLES,
    REST_LEVELS,
    TEMPO_LEVELS,
)


def build_feasible_region(
    eligibility: dict[str, Any],
    function_layer: dict[str, Any],
    risk_constraints: dict[str, Any],
    action_profile: dict[str, Any],
) -> dict[str, Any]:
    forms: dict[str, dict[str, Any]] = {}
    excluded_reasons: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    global_preferences: dict[str, Any] = {}

    # All risk projections (per-form 幅度/节奏/休息 exclusions + global caps) now come
    # from Tool 4, which matches Tool 3's risk list against the KB4 table. Tool 3
    # itself only enumerates risks; it no longer projects onto parameters.
    global_constraints = action_profile.get("global_constraints") or action_profile.get("dose_constraints") or {}

    for form in BADUANJIN_FORMS:
        form_id = form["form_id"]
        standard_cycles = form["standard_cycles"]
        cycle_upper = _cycle_upper_bound(form_id, standard_cycles, eligibility, function_layer, global_constraints)
        forms[str(form_id)] = {
            "form_id": form_id,
            "name": form["name"],
            "cycles": list(range(1, cycle_upper + 1)),
            "amplitude": list(BASE_AMPLITUDE_LEVELS),
            "tempo": list(TEMPO_LEVELS),
            "rest": list(REST_LEVELS),
            "standard_cycles": standard_cycles,
        }
        if global_constraints.get("cycle_decrement"):
            excluded_reasons.append(
                {
                    "source": "global_cycle_decrement",
                    "form_id": form_id,
                    "parameter": "cycles",
                    "reason": f"全局安全限制：每式循环上界在原候选基础上下调 {int(global_constraints['cycle_decrement'])}。",
                }
            )

    # Hard constraints from Tool 4 carry amplitude/tempo AND rest exclusions
    # (e.g. rest disallow 标准 == 休息至少延长), applied uniformly via _apply_exclusion.
    for constraint in action_profile.get("hard_constraints") or []:
        _apply_exclusion(forms, constraint, excluded_reasons)

    if eligibility.get("data_incomplete"):
        for form in forms.values():
            _apply_rest_minimum(form, "延长")
        global_preferences.update(
            {
                "amplitude": "优先简化，但若该式无明确动作受限且可行域允许，可由 LLM 选择标准幅度。",
                "tempo": "优先慢速，但若无通气/心律/症状限制且可行域允许，可由 LLM 选择标准节奏。",
                "rest": "至少延长休息。",
            }
        )
        annotations.append(
            {
                "type": "global_downshift",
                "detail": "data_incomplete 触发全局保守倾向：循环不上调（封顶在标准）、休息至少延长，幅度/节奏作为偏好而非全式硬排除。",
                "affected": "global",
            }
        )

    for form in forms.values():
        if not form["amplitude"]:
            form["amplitude"] = ["坐式"]
            form["cycles"] = [1]
            form["tempo"] = ["慢速"]
            form["rest"] = ["延长"]
            annotations.append(
                {
                    "type": "form_min_dose",
                    "detail": f"{form['name']} 幅度可行域为空，回退到坐式+慢速+cycle=1。",
                    "affected": form["form_id"],
                }
            )
        if not form["tempo"]:
            form["tempo"] = ["慢速"]
        if not form["rest"]:
            form["rest"] = ["延长"]

    annotations.extend(function_layer.get("decision_notes") or function_layer.get("annotations") or [])
    annotations.extend(risk_constraints.get("annotations") or [])
    annotations.extend(action_profile.get("annotations") or [])
    for item in action_profile.get("unresolved") or []:
        annotations.append(
            {
                "type": "unsupported_signal",
                "detail": item.get("detail", "存在未覆盖受限信号"),
                "affected": item.get("affected_forms", "unknown"),
            }
        )

    frequency_per_week = global_constraints.get("frequency_per_week") or [5, 6, 7]
    session_cap = _single_session_cap(function_layer)
    if global_constraints.get("single_session_max_minutes") is not None:
        session_cap = min(session_cap, int(global_constraints["single_session_max_minutes"]))

    return {
        "forms": forms,
        "global": {
            "sets_per_session": [int(function_layer.get("sets_per_session", 1) or 1)],
            "frequency_per_week": frequency_per_week,
            "times_per_day": [1, 2, 3, 4],
            "protocol_max_weekly_minutes": 300,
            "single_session_max_minutes": session_cap,
            "preferences": global_preferences,
        },
        "excluded_reasons": excluded_reasons,
        "annotations": annotations,
    }


def _cycle_upper_bound(
    form_id: int,
    standard_cycles: int,
    eligibility: dict[str, Any],
    function_layer: dict[str, Any],
    global_constraints: dict[str, Any] | None = None,
) -> int:
    # Upward titration over the standard base, by Tool 2's per-form cycle
    # increment (keyed by form_id; tolerate int or str keys), capped at MAX_CYCLES.
    increment = function_layer.get("cycle_increment") or {}
    bump = increment.get(form_id)
    if bump is None:
        bump = increment.get(str(form_id), 0)
    upper = min(standard_cycles + (bump or 0), MAX_CYCLES)

    # Incomplete/uncertain data: no upward titration — cap at the standard base.
    if eligibility.get("data_incomplete"):
        upper = min(upper, standard_cycles)
    decrement = int((global_constraints or {}).get("cycle_decrement", 0) or 0)
    if decrement:
        upper = max(1, upper - decrement)
    return max(1, min(upper, MAX_CYCLES))


def _apply_exclusion(
    forms: dict[str, dict[str, Any]],
    constraint: dict[str, Any],
    excluded_reasons: list[dict[str, Any]],
) -> None:
    target_forms = constraint.get("forms")
    if target_forms == "all":
        form_ids = [int(key) for key in forms]
    else:
        form_ids = list(target_forms or [])

    parameter = constraint.get("parameter")
    disallowed = set(constraint.get("disallow") or [])
    for form_id in form_ids:
        form = forms.get(str(form_id))
        if not form or parameter not in form:
            continue
        before = list(form[parameter])
        form[parameter] = [value for value in form[parameter] if value not in disallowed]
        if before != form[parameter]:
            excluded_reasons.append(
                {
                    "source": constraint.get("source"),
                    "form_id": form_id,
                    "parameter": parameter,
                    "disallowed": sorted(disallowed),
                    "reason": constraint.get("reason"),
                }
            )


def _apply_rest_minimum(form: dict[str, Any], minimum: str) -> None:
    order = {"标准": 0, "延长": 1}
    min_rank = order[minimum]
    form["rest"] = [value for value in form["rest"] if order[value] >= min_rank]


def _single_session_cap(function_layer: dict[str, Any]) -> int:
    candidate_level = function_layer.get("candidate_level") or function_layer.get("candidate_class")
    if candidate_level in {"low", "运动能力低"}:
        return 20
    if candidate_level in {"medium", "运动能力中"}:
        return 30
    return 45
