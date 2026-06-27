"""Total-volume allocation for parameters 5-7.

A deterministic function in the constrained-generation layer (see
generation/__init__.py) — NOT a numbered Tool 0-5. It enumerates the legal
sets/frequency/times candidate combinations the LLM then selects one from.
"""

from __future__ import annotations

from typing import Any

from constants import SOP_VOLUME_LEVELS
from profile_access import data_bool, data_number


# Per-form time-weighting factors applied to the single-session estimate.
# 节奏档对应的相对速度（标准=1.0）：慢速 = 标准速度的 0.75 倍。
TEMPO_SPEED = {"标准": 1.0, "慢速": 0.75}
# 时间因子 = 1 / 速度：速度越慢，同样的动作耗时越长（慢速 ≈ ×1.333）。仅作用于该式动作时长。
TEMPO_FACTOR = {level: round(1.0 / speed, 4) for level, speed in TEMPO_SPEED.items()}
# 式间休息：延长 = 标准时长的 1.25 倍。仅作用于该式之后的式间休息。
REST_FACTOR = {"标准": 1.0, "延长": 1.25}

# 通用八段锦标准计时基线：来自一套约 12 分钟的逐式逐循环录制（标准节奏 + 标准幅度 + 标准式间休息）。
# movement_sec = 该式在 std_cycles 个循环下的动作总时长；rest_after_sec = 该式之后的式间休息。
# 每循环动作时长 = movement_sec / std_cycles，按所选循环数线性外推（幅度不计入时长）。
FORM_TIMING_BASELINE = {
    1: {"movement_sec": 65.0, "std_cycles": 4, "rest_after_sec": 6.0},
    2: {"movement_sec": 84.0, "std_cycles": 6, "rest_after_sec": 5.0},
    3: {"movement_sec": 83.0, "std_cycles": 6, "rest_after_sec": 7.0},
    4: {"movement_sec": 77.0, "std_cycles": 6, "rest_after_sec": 6.0},
    5: {"movement_sec": 101.0, "std_cycles": 6, "rest_after_sec": 7.0},
    6: {"movement_sec": 89.0, "std_cycles": 4, "rest_after_sec": 6.0},
    7: {"movement_sec": 63.0, "std_cycles": 6, "rest_after_sec": 5.0},
    8: {"movement_sec": 38.0, "std_cycles": 7, "rest_after_sec": 0.0},  # 第八式后直接接收势
}
# 每套固定环节（每套一次，不随循环数/节奏/幅度变化）。
PREP_SEC = 16.0          # 预备势
PREP_REST_SEC = 5.0      # 预备势后的式间休息
CLOSING_SEC = 53.0       # 收势


def allocate_volume_options(
    form_plan: dict[str, Any],
    profile: dict[str, Any],
    function_layer: dict[str, Any],
    eligibility: dict[str, Any],
    target_weekly_minutes: int | None = None,
) -> dict[str, Any]:
    # 参数⑤ sets-per-session is decided by Tool 2 (Borg-only case titrates it;
    # otherwise 1). It overrides the form-stage default for the time estimate and
    # is reported on every candidate so downstream stays consistent.
    sets_per_session = int(function_layer.get("sets_per_session", 1) or 1)
    single_session_minutes = estimate_single_session_minutes(form_plan, sets_override=sets_per_session)
    target = _target_minutes(profile, function_layer, eligibility, target_weekly_minutes)
    combinations = []

    for level in SOP_VOLUME_LEVELS:
        weekly_sessions = level["times_per_day"] * level["frequency_per_week"]
        weekly_minutes = round(single_session_minutes * weekly_sessions, 1)
        combinations.append(
            {
                "level": level["level"],
                "sets_per_session": sets_per_session,
                "frequency_per_week": level["frequency_per_week"],
                "times_per_day": level["times_per_day"],
                "weekly_sessions": weekly_sessions,
                "weekly_minutes": weekly_minutes,
                "target_delta": round(weekly_minutes - target, 1),
            }
        )

    if _has_cad_or_af(profile):
        combinations = [
            item
            for item in combinations
            if 3 <= item["frequency_per_week"] <= 5
        ]
    combinations = [item for item in combinations if item["weekly_minutes"] <= 300]

    high_risk = eligibility.get("risk_level") in {"high", "blocked"}
    combinations = _rank_combinations(combinations, target, high_risk)
    guidance = "目标约{}分钟/周；高危或低自我效能时优先选择不超过目标的候选。".format(target)

    return {
        "single_session_minutes": single_session_minutes,
        "target_weekly_minutes": target,
        "feasible_combinations": combinations[:5],
        "guidance": guidance,
    }


def estimate_single_session_minutes(form_plan: dict[str, Any], sets_override: int | None = None) -> float:
    """Additive per-form timing model grounded in the 12-min standard recording.

    单次时长 = 固定环节(预备势+预备休息+收势)
             + Σ式 [ 每循环动作秒 × 选定循环数 × 该式节奏因子 ]
             + Σ式 [ 该式后式间休息秒 × 该式休息因子 ]
    再整套 × sets_per_session（每套含预备/收势各一次）。
    """

    forms = form_plan.get("forms", [])
    if isinstance(forms, dict):
        forms = list(forms.values())

    if sets_override is not None:
        sets_per_session = sets_override
    else:
        sets_per_session = form_plan.get("global", {}).get("sets_per_session", 1)
    sets_per_session = int(sets_per_session or 1)

    # 固定环节：每套一次，不受逐式 tempo/rest/幅度影响。
    routine_sec = PREP_SEC + PREP_REST_SEC + CLOSING_SEC
    for item in forms:
        baseline = FORM_TIMING_BASELINE.get(_coerce_form_id(item.get("form_id")))
        if baseline is None:
            continue
        cycles = _coerce_cycles(item.get("cycles"), baseline["std_cycles"])
        per_cycle_sec = baseline["movement_sec"] / baseline["std_cycles"]
        tempo_factor = TEMPO_FACTOR.get(item.get("tempo", "标准"), 1.0)
        rest_factor = REST_FACTOR.get(item.get("rest", "标准"), 1.0)
        routine_sec += per_cycle_sec * cycles * tempo_factor
        routine_sec += baseline["rest_after_sec"] * rest_factor

    minutes = routine_sec * sets_per_session / 60.0
    return round(minutes, 1)


def _coerce_form_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_cycles(value: Any, default: int) -> int:
    try:
        cycles = int(value)
    except (TypeError, ValueError):
        return default
    return cycles if cycles > 0 else default


def _target_minutes(
    profile: dict[str, Any],
    function_layer: dict[str, Any],
    eligibility: dict[str, Any],
    target_weekly_minutes: int | None,
) -> int:
    if target_weekly_minutes is not None:
        target = target_weekly_minutes
    else:
        target = 150

    see = data_number(profile, "SEE", "seesum")
    if see is not None:
        if see >= 66:
            target += 30
        elif see <= 32:
            target -= 30

    candidate_level = function_layer.get("candidate_level") or function_layer.get("candidate_class")
    if candidate_level in {"low", "运动能力低"}:
        target = min(target, 150)
    if eligibility.get("data_incomplete"):
        target = min(target, 120)
    return int(max(120, min(300, target)))


def _rank_combinations(
    combinations: list[dict[str, Any]], target: int, high_risk: bool
) -> list[dict[str, Any]]:
    def score(item: dict[str, Any]) -> tuple[float, int, int]:
        delta = item["weekly_minutes"] - target
        over_penalty = abs(delta) + (15 if high_risk and delta > 0 else 0)
        frequency_preference = -item["frequency_per_week"]
        return (over_penalty, frequency_preference, item["times_per_day"])

    return sorted(combinations, key=score)


def _has_cad_or_af(profile: dict[str, Any]) -> bool:
    return data_bool(profile, "HIST", "cad") or data_bool(profile, "HIST", "af")
