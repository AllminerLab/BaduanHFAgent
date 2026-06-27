"""Tool 5: deterministic prescription guardrail."""

from __future__ import annotations

from typing import Any

from constants import (
    AMPLITUDE_LEVELS,
    FORM_IDS,
    FORM_NAME_BY_ID,
    MAX_CYCLES,
    REST_LEVELS,
    TEMPO_LEVELS,
)
from generation.volume_allocation import estimate_single_session_minutes
from profile_access import data_bool, data_number, data_value


# Full level sets per parameter — used to render the "allowed" set in restricted-action
# messages (allowed = all levels minus the values Tool 4 excluded for this patient).
_PARAM_ALL_LEVELS = {
    "amplitude": set(AMPLITUDE_LEVELS),
    "tempo": set(TEMPO_LEVELS),
    "rest": set(REST_LEVELS),
}


PARAM_5_RANGE = (1, 5)
PARAM_6_RANGE = (3, 7)
PARAM_7_RANGE = (1, 4)
WEEKLY_MINUTES_MAX = 300
SPECIAL_SESSION_MINUTES_MAX = 30
ALIAS_TO_DATA_FIELD: dict[str, tuple[str, str]] = {
    "nyha": ("CHECK", "nyha"),
    "bdsbp": ("CHECK", "sbp"),
    "sbp": ("CHECK", "sbp"),
    "bddbp": ("CHECK", "dbp"),
    "dbp": ("CHECK", "dbp"),
    "histarb3": ("HIST", "av_block_code"),
    "histcad": ("HIST", "cad"),
    "histara9": ("HIST", "af"),
    "histcvd6": ("HIST", "stroke"),
    "stroke": ("HIST", "stroke"),
    "histpad8": ("HIST", "pad"),
    "histcpd": ("HIST", "copd"),
    "pmyn": ("PACE", "device"),
    "edemalt": ("CHECK", "edema_left"),
    "edemart": ("CHECK", "edema_right"),
    "mobti": ("TEST", "mobility_time"),
    "mobcl": ("TEST", "mobility_class"),
    "eq5d4": ("EQ5D", "pain"),
    "pain": ("EQ5D", "pain"),
    "cpxslop": ("CPX", "ve_vco2_slope"),
    "dyspnea1": ("SYMPTOM", "dyspnea"),
    "ecgrtb": ("ECG", "arrhythmia"),
    "ecglbbb": ("ECG", "arrhythmia"),
    "ecgrbbb": ("ECG", "arrhythmia"),
    "ecolvefs2": ("ECHO", "lvef"),
    "lvef": ("ECHO", "lvef"),
    "cpxhrpk": ("CPX", "peak_hr"),
    "cpxhrrt": ("CPX", "rest_hr"),
    "bdhtr": ("CHECK", "rest_hr"),
    "cpxsbpk": ("CPX", "peak_sbp"),
    "ecoar": ("ECHO", "valve_ar"),
    "ecomr": ("ECHO", "valve_mr"),
    "ecoas": ("ECHO", "valve_as"),
    "ecoms": ("ECHO", "valve_ms"),
    "颈椎病史（1/0）": ("MSK", "cervical"),
    "cervical_spine_history": ("MSK", "cervical"),
    "neck_history": ("MSK", "cervical"),
    "肩关节病史（1/0）": ("MSK", "shoulder"),
    "shoulder_history": ("MSK", "shoulder"),
    "shoulder_joint_history": ("MSK", "shoulder"),
    "腰椎病史（1/0）": ("MSK", "lumbar"),
    "lumbar_spine_history": ("MSK", "lumbar"),
    "low_back_history": ("MSK", "lumbar"),
    "膝关节手术史（1/0）": ("MSK", "knee"),
    "膝关节病史（1/0）": ("MSK", "knee"),
    "knee_surgery_history": ("MSK", "knee"),
    "knee_history": ("MSK", "knee"),
}


def validate_prescription(
    prescription: dict[str, Any],
    feasible_region: dict[str, Any],
    volume_options: dict[str, Any] | None = None,
    eligibility: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the generated prescription.

    Patient-data rules are evidence-based: Tool 5 searches the available
    patient profile/source rows for each rule's variables. If the corresponding
    data are absent, that patient-specific rule is treated as not violated.
    Output-completeness and dose-range rules still fail when the generated
    prescription itself is incomplete or outside the allowed range.
    """

    violations: list[dict[str, Any]] = []

    # Safety block (last line of defense): a refuse/blocked eligibility state must
    # never yield an auto prescription. Upstream Tool 1 already gates these out,
    # but per framework 6.1 Tool 5 keeps a deterministic block -> refuse path.
    if eligibility is not None and (
        eligibility.get("eligibility_status") == "refuse"
        or eligibility.get("risk_level") == "blocked"
    ):
        violations.append(
            _violation(
                "safety_block",
                None,
                eligibility.get("eligibility_status"),
                "allow",
                severity="block",
            )
        )
        return _result(False, violations)

    if profile is not None:
        _validate_stop_red_lines(profile, violations)
        if any(item.get("severity") == "block" for item in violations):
            return _result(False, violations)

    status = prescription.get("status")
    if status != "generated":
        violations.append(_violation("invalid_generation_status", None, status, "generated"))
        return _result(False, violations)

    if "confidence" not in prescription:
        violations.append(_violation("confidence_missing", None, None, "required"))

    body = prescription.get("prescription")
    if not isinstance(body, dict):
        violations.append(_violation("missing_prescription", None, None, "object"))
        return _result(False, violations)

    forms = body.get("forms")
    if not isinstance(forms, list):
        violations.append(_violation("forms_not_list", None, type(forms).__name__, "list"))
        return _result(False, violations)

    ids = [item.get("form_id") for item in forms if isinstance(item, dict)]
    if sorted(ids) != FORM_IDS:
        violations.append(_violation("eight_forms_required", None, ids, FORM_IDS))

    feasible_forms = feasible_region.get("forms", {})
    for item in forms:
        if not isinstance(item, dict):
            violations.append(_violation("form_not_object", None, item, "object"))
            continue
        form_id = item.get("form_id")
        feasible = feasible_forms.get(str(form_id), {})
        _validate_form(item, feasible, violations)

    global_plan = body.get("global")
    if not isinstance(global_plan, dict):
        violations.append(_violation("global_not_object", None, type(global_plan).__name__, "object"))
        global_plan = {}
    _validate_global(body, global_plan, feasible_region, volume_options, violations)
    if profile is not None:
        _validate_restricted_action_table(feasible_region, forms, violations)
        _validate_special_population_rules(profile, prescription, forms, global_plan, violations)

    return _result(not violations, violations)


def _validate_form(
    item: dict[str, Any],
    feasible: dict[str, Any],
    violations: list[dict[str, Any]],
) -> None:
    form_id = item.get("form_id")
    expected_name = FORM_NAME_BY_ID.get(form_id)
    if expected_name and item.get("name") not in {None, expected_name}:
        violations.append(_violation("form_name_changed", form_id, item.get("name"), expected_name))

    cycles = item.get("cycles")
    if not isinstance(cycles, int):
        violations.append(_violation("cycles_not_int", form_id, cycles, "int"))
    else:
        if cycles < 1:
            violations.append(_violation("cycle_below_min", form_id, cycles, ">=1"))
        if cycles > MAX_CYCLES:
            violations.append(_violation("max_cycles_exceeded", form_id, cycles, MAX_CYCLES))
        if feasible and cycles not in feasible.get("cycles", []):
            violations.append(_violation("cycle_outside_feasible_region", form_id, cycles, feasible.get("cycles")))

    for side_key in ["left_cycles", "right_cycles"]:
        side_cycles = item.get(side_key)
        if side_cycles is not None and (not isinstance(side_cycles, int) or side_cycles < 1):
            violations.append(_violation(f"{side_key}_below_min", form_id, side_cycles, ">=1"))

    for parameter in ["amplitude", "tempo", "rest"]:
        value = item.get(parameter)
        allowed = feasible.get(parameter) if feasible else None
        if value is None:
            violations.append(_violation(f"{parameter}_missing", form_id, None, "required"))
        elif allowed is not None and value not in allowed:
            violations.append(_violation(f"{parameter}_outside_feasible_region", form_id, value, allowed))

    if not item.get("rationale"):
        violations.append(_violation("form_rationale_missing", form_id, item.get("rationale"), "non-empty"))


def _validate_global(
    body: dict[str, Any],
    global_plan: dict[str, Any],
    feasible_region: dict[str, Any],
    volume_options: dict[str, Any] | None,
    violations: list[dict[str, Any]],
) -> None:
    allowed_global = feasible_region.get("global", {})
    for key in ["sets_per_session", "frequency_per_week", "times_per_day"]:
        if key not in global_plan:
            violations.append(_violation(f"{key}_missing", None, None, "required"))
            continue
        if not _is_int(global_plan[key]):
            violations.append(_violation(f"{key}_not_int", None, global_plan[key], "int"))
            continue
        allowed = allowed_global.get(key)
        if allowed is not None and global_plan[key] not in allowed:
            violations.append(_violation(f"{key}_outside_feasible_region", None, global_plan[key], allowed))

    _validate_global_range("parameter_5_sets_per_session_out_of_range", global_plan.get("sets_per_session"), PARAM_5_RANGE, violations)
    _validate_global_range("parameter_6_frequency_per_week_out_of_range", global_plan.get("frequency_per_week"), PARAM_6_RANGE, violations)
    _validate_global_range("parameter_7_times_per_day_out_of_range", global_plan.get("times_per_day"), PARAM_7_RANGE, violations)

    weekly_minutes = global_plan.get("weekly_minutes")
    weekly_minutes_number = _to_number(weekly_minutes)
    protocol_max = min(allowed_global.get("protocol_max_weekly_minutes", WEEKLY_MINUTES_MAX), WEEKLY_MINUTES_MAX)
    if weekly_minutes is None:
        violations.append(_violation("weekly_minutes_missing", None, None, "required"))
    elif weekly_minutes_number is None:
        violations.append(_violation("weekly_minutes_not_number", None, weekly_minutes, "number"))
    elif weekly_minutes_number > protocol_max:
        violations.append(_violation("weekly_minutes_exceeded", None, weekly_minutes, protocol_max))

    session_minutes = global_plan.get("single_session_minutes")
    session_minutes_number = _to_number(session_minutes)
    session_max = allowed_global.get("single_session_max_minutes")
    if session_minutes is None:
        violations.append(_violation("single_session_minutes_missing", None, None, "required"))
    elif session_minutes_number is None:
        violations.append(_violation("single_session_minutes_not_number", None, session_minutes, "number"))
    elif session_max is not None and session_minutes_number > session_max:
        violations.append(_violation("single_session_minutes_exceeded", None, session_minutes, session_max))

    expected_session_minutes = estimate_single_session_minutes(body)
    if session_minutes_number is not None and abs(session_minutes_number - expected_session_minutes) > 0.2:
        violations.append(
            _violation(
                "single_session_minutes_mismatch",
                None,
                session_minutes,
                expected_session_minutes,
            )
        )

    frequency = global_plan.get("frequency_per_week")
    times_per_day = global_plan.get("times_per_day")
    if (
        weekly_minutes_number is not None
        and session_minutes_number is not None
        and isinstance(frequency, int)
        and isinstance(times_per_day, int)
    ):
        expected_weekly_minutes = round(expected_session_minutes * frequency * times_per_day, 1)
        if abs(weekly_minutes_number - expected_weekly_minutes) > 0.2:
            violations.append(
                _violation(
                    "weekly_minutes_mismatch",
                    None,
                    weekly_minutes,
                    expected_weekly_minutes,
                )
            )

    if volume_options:
        valid_pairs = {
            (
                item["sets_per_session"],
                item["frequency_per_week"],
                item["times_per_day"],
            )
            for item in volume_options.get("feasible_combinations", [])
        }
        selected = (
            global_plan.get("sets_per_session"),
            global_plan.get("frequency_per_week"),
            global_plan.get("times_per_day"),
        )
        if selected not in valid_pairs:
            violations.append(
                _violation("volume_combo_not_in_candidates", None, selected, sorted(valid_pairs))
            )


def _validate_global_range(
    rule: str,
    value: Any,
    allowed_range: tuple[int, int],
    violations: list[dict[str, Any]],
) -> None:
    if value is None or not _is_int(value):
        return
    low, high = allowed_range
    if not low <= value <= high:
        violations.append(_violation(rule, None, value, f"{low}-{high}"))


def _validate_stop_red_lines(profile: dict[str, Any], violations: list[dict[str, Any]]) -> None:
    acute_conditions: set[str] = set()

    nyha = _first_number(profile, ["nyha"])
    if nyha is not None and nyha >= 4:
        violations.append(_violation("stop_redline_nyha_iv", None, nyha, "NYHA < 4", severity="block"))

    sbp = _first_number(profile, ["bdsbp", "sbp"])
    dbp = _first_number(profile, ["bddbp", "dbp"])
    if sbp is not None and sbp > 220:
        violations.append(_violation("stop_redline_rest_sbp", None, sbp, "<=220", severity="block"))
    if dbp is not None and dbp > 110:
        violations.append(_violation("stop_redline_rest_dbp", None, dbp, "<=110", severity="block"))

    av_block = _first_number(profile, ["histarb3"])
    if av_block == 4:
        violations.append(_violation("stop_redline_high_grade_av_block", None, av_block, "histarb3 != 4", severity="block"))

    acute_map = {
        "stop_redline_early_acs": {"acute_coronary_syndrome_early", "acute_coronary_syndrome_2days", "acs_early", "acs_within_2_days"},
        "stop_redline_malignant_arrhythmia": {"malignant_arrhythmia", "severe_arrhythmia"},
        "stop_redline_acute_heart_failure": {"acute_heart_failure", "acute_decompensated_hf"},
        "stop_redline_acute_myocarditis": {"acute_myocarditis", "acute_myocarditis_pericarditis"},
        "stop_redline_acute_pericarditis": {"acute_pericarditis", "acute_myocarditis_pericarditis"},
        "stop_redline_acute_endocarditis": {"acute_endocarditis"},
        "stop_redline_intracardiac_thrombus": {"intracardiac_thrombus", "cardiac_thrombus"},
        "stop_redline_uncontrolled_diabetes": {"uncontrolled_diabetes"},
    }
    for rule, names in acute_map.items():
        if acute_conditions.intersection(names) or _any_truthy(profile, names):
            violations.append(_violation(rule, None, sorted(names), "absent", severity="block"))


def _validate_restricted_action_table(
    feasible_region: dict[str, Any],
    forms: list[Any],
    violations: list[dict[str, Any]],
) -> None:
    """禁忌动作校验：核对处方是否遵守了 Tool 4 给【这名患者】命中的每一条硬受限。

    依据是 Tool 4 实际写入可行域的 ``excluded_reasons``（每条含 form_id / 参数 / 被禁
    取值 / 来源风险 / 原因）——也就是"这名患者命中的受限表"。逐条核对、一条不漏；不同
    患者命中的受限不同，故以其各自的 excluded_reasons 为准，而非固定清单。这里只是核对
    Tool 4 的输出有没有被遵守，不在 Tool 5 里重新检测（避免与 Tool 4 判定分叉而漏检）。
    """

    form_by_id = {item.get("form_id"): item for item in forms if isinstance(item, dict)}
    for excluded in feasible_region.get("excluded_reasons") or []:
        parameter = excluded.get("parameter")
        disallowed = set(excluded.get("disallowed") or [])
        if parameter not in {"amplitude", "tempo", "rest"} or not disallowed:
            continue  # cycle/global caps are range-checked in _validate_form/_validate_global
        item = form_by_id.get(excluded.get("form_id"))
        if not isinstance(item, dict):
            continue
        if item.get(parameter) in disallowed:
            allowed = _PARAM_ALL_LEVELS.get(parameter, set()) - disallowed
            violations.append(
                _violation(
                    f"restricted_action_{excluded.get('source')}_{parameter}",
                    excluded.get("form_id"),
                    item.get(parameter),
                    sorted(allowed),
                )
            )


# 软（生理/症状性）相对禁忌不在 Tool 5 校验范围内（见 SZY Tool 5 处方安全校验器文档）：
# 它们由 Tool 4 soft_preferences 引导、并由动态知识卡承载，Tool 5 不再硬性回退它们。
# 对应的 _has_* 检测器与受限表的独立重检测一并移除——禁忌动作改为核对 Tool 4 的
# excluded_reasons（见 _validate_restricted_action_table），不在 Tool 5 重新判定。


def _validate_special_population_rules(
    profile: dict[str, Any],
    prescription: dict[str, Any],
    forms: list[Any],
    global_plan: dict[str, Any],
    violations: list[dict[str, Any]],
) -> None:
    if _has_cad_or_af(profile):
        frequency = global_plan.get("frequency_per_week")
        if _is_int(frequency) and not 3 <= frequency <= 5:
            violations.append(_violation("cad_af_frequency_outside_recommended_range", None, frequency, "3-5"))
        session_minutes = global_plan.get("single_session_minutes")
        session_minutes_number = _to_number(session_minutes)
        if session_minutes_number is not None and session_minutes_number > SPECIAL_SESSION_MINUTES_MAX:
            violations.append(
                _violation(
                    "cad_af_single_session_minutes_exceeded",
                    None,
                    session_minutes,
                    f"<={SPECIAL_SESSION_MINUTES_MAX}",
                )
            )

    severe_valves = _severe_valvular_findings(profile)
    if severe_valves and not _prescription_mentions_valvular_consideration(prescription, severe_valves):
        violations.append(
            _violation(
                "severe_valvular_condition_not_considered",
                None,
                severe_valves,
                "处方 rationale/annotations 需体现已考虑重度瓣膜反流/狭窄及中低强度/保守处理",
            )
        )

    # 自我效能低（SEESUM≤50）→ 第二/五/七式幅度简化，现为 Tool 4 的 hard 受限：它会进
    # 可行域 excluded_reasons，由禁忌动作校验（_validate_restricted_action_table）统一核对，
    # 故此处不再单独检查，避免与禁忌动作重复报。


def _has_cad_or_af(profile: dict[str, Any]) -> bool:
    return data_bool(profile, "HIST", "cad") or data_bool(profile, "HIST", "af")


def _severe_valvular_findings(profile: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "ecoar": "主动脉瓣重度返流",
        "ecomr": "二尖瓣重度返流",
        "ecoas": "主动脉瓣重度狭窄",
        "ecoms": "二尖瓣重度狭窄",
    }
    findings: list[dict[str, Any]] = []
    for field, label in labels.items():
        value = _first_number(profile, [field])
        if value == 3:
            findings.append({"field": field.upper(), "value": value, "label": label})
    return findings


def _prescription_mentions_valvular_consideration(
    prescription: dict[str, Any],
    severe_valves: list[dict[str, Any]],
) -> bool:
    text = _collect_text(prescription)
    if not text:
        return False
    text_lower = text.lower()
    field_or_label_present = any(
        item["field"].lower() in text_lower or item["label"] in text
        for item in severe_valves
    )
    generic_valve_present = (
        "重度瓣膜" in text
        or "瓣膜反流" in text
        or "瓣膜返流" in text
        or "瓣膜狭窄" in text
        or (("主动脉瓣" in text or "二尖瓣" in text) and ("反流" in text or "返流" in text or "狭窄" in text))
    )
    consideration_present = any(
        token in text
        for token in ["中低强度", "低强度", "保守", "考虑", "已考虑", "降低", "下调", "减少", "放慢", "延长", "简化"]
    )
    return (field_or_label_present or generic_valve_present) and consideration_present


def _collect_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            parts.append(_collect_text(item))
    elif isinstance(value, list):
        for item in value:
            parts.append(_collect_text(item))
    elif isinstance(value, str):
        parts.append(value)
    return " ".join(part for part in parts if part)


def _first_number(profile: dict[str, Any], aliases: list[str] | set[str], fallback: Any = None) -> float | None:
    if fallback is not None:
        return _to_number(fallback)
    value = _first_raw_value(profile, aliases)
    return _to_number(value)


def _any_truthy(profile: dict[str, Any], aliases: set[str]) -> bool:
    value = _first_raw_value(profile, aliases)
    return _is_truthy(value)


def _first_raw_value(profile: dict[str, Any], aliases: list[str] | set[str]) -> Any:
    for alias in aliases:
        key = str(alias).lower()
        target = ALIAS_TO_DATA_FIELD.get(key) or ALIAS_TO_DATA_FIELD.get(str(alias))
        if target is None:
            continue
        value = data_value(profile, target[0], target[1])
        if value is not None:
            return value
    return None


def _to_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_truthy(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"0", "false", "no", "否", "无", "none", "nan"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _result(passed: bool, violations: list[dict[str, Any]]) -> dict[str, Any]:
    if passed:
        return {"passed": True, "violations": [], "action": "accept", "regenerate_feedback": ""}

    block = [item for item in violations if item.get("severity") == "block"]
    hard = [item for item in violations if item.get("severity") == "hard"]

    # Framework 6.1: block -> refuse (safety stop), hard -> regenerate (feed the
    # violation list back to the LLM). After N failed regenerations the
    # orchestrator turns an unresolved "regenerate" into generation_failure.
    if block:
        action = "refuse"
        feedback_source = block
    else:
        action = "regenerate"
        feedback_source = hard

    return {
        "passed": False,
        "violations": violations,
        "action": action,
        "regenerate_feedback": _build_feedback(feedback_source, action),
    }


# Map parameter tokens (extracted from rule names) to Chinese labels.
_PARAM_LABELS = {
    "amplitude": "幅度档",
    "tempo": "节奏档",
    "rest": "式间休息",
    "cycles": "循环数",
    "confidence": "confidence",
    "sets_per_session": "每次套数",
    "frequency_per_week": "每周频率",
    "times_per_day": "每天次数",
    "weekly_minutes": "周总时长",
    "single_session_minutes": "单次时长",
}

# Exact natural-language remediation templates for specific rules. Form-scoped
# messages omit the form number (the group header 【第N式】 already carries it).
_MSG_TEMPLATES = {
    "confidence_missing": "缺少 confidence 字段（high/medium/low）",
    "form_name_changed": "名称应为「{limit}」，不得改为「{value}」",
    "cycles_not_int": "循环数必须为整数",
    "cycle_below_min": "循环数不得小于 1（不可跳式）",
    "max_cycles_exceeded": "循环数 {value} 超过上限 {limit}，请下调",
    "cycle_outside_feasible_region": "循环数 {value} 不在可行域允许集合 {limit} 内，请改选其中之一",
    "form_rationale_missing": "缺少 rationale，请补充简短临床依据",
    "left_cycles_below_min": "left_cycles 不得小于 1",
    "right_cycles_below_min": "right_cycles 不得小于 1",
    "eight_forms_required": "必须包含完整八式（form_id 1-8），当前为 {value}",
    "form_not_object": "存在非对象的式条目，请修正",
    "weekly_minutes_exceeded": "周总时长 {value} 超过上限 {limit}，请下调总量",
    "single_session_minutes_exceeded": "单次时长 {value} 超过上限 {limit}",
    "volume_combo_not_in_candidates": "总量三参数组合 {value} 不在候选 {limit} 中，请整组改选",
    "cad_af_frequency_outside_recommended_range": "合并冠心病/房颤：每周频率须在 {limit}（当前 {value}）",
    "cad_af_single_session_minutes_exceeded": "合并冠心病/房颤：单次时长 {value} 超过 {limit}",
    "severe_valvular_condition_not_considered": "未体现已考虑重度瓣膜情况：{limit}",
    # 安全红线（block -> refuse）
    "safety_block": "资格状态为「{value}」，按安全规则不允许自动生成处方",
    "stop_redline_nyha_iv": "NYHA {value} 命中 NYHA IV 停练红线",
    "stop_redline_rest_sbp": "静息收缩压 {value} 超过 220 mmHg 停练红线",
    "stop_redline_rest_dbp": "静息舒张压 {value} 超过 110 mmHg 停练红线",
    "stop_redline_high_grade_av_block": "高度房室传导阻滞 停练红线",
    "stop_redline_early_acs": "急性冠脉综合征（早期）停练红线",
    "stop_redline_malignant_arrhythmia": "恶性心律失常 停练红线",
    "stop_redline_acute_heart_failure": "急性/失代偿心衰 停练红线",
    "stop_redline_acute_myocarditis": "急性心肌炎 停练红线",
    "stop_redline_acute_pericarditis": "急性心包炎 停练红线",
    "stop_redline_acute_endocarditis": "急性心内膜炎 停练红线",
    "stop_redline_intracardiac_thrombus": "心腔内血栓 停练红线",
    "stop_redline_uncontrolled_diabetes": "未控制的糖尿病 停练红线",
}


def _build_feedback(items: list[dict[str, Any]], action: str) -> str:
    """Group violations by form (+ a global bucket) and render natural-language
    remediation lines. No cap — every violation is listed. The header depends on
    action: regenerate (fixable) vs refuse (safety stop, not regenerated)."""

    if not items:
        return ""

    grouped: dict[Any, tuple[int, str, list[str]]] = {}
    for item in items:
        form_id = item.get("form_id")
        if form_id is None:
            key, sort_key, label = ("__global__", 99, "全局")
        else:
            key, sort_key, label = (form_id, int(form_id), f"第{form_id}式")
        bucket = grouped.setdefault(key, (sort_key, label, []))
        message = _remediation(item)
        if message not in bucket[2]:
            bucket[2].append(message)

    if action == "refuse":
        header = "处方被 Tool 5 安全红线拦截，不予自动生成。命中以下红线："
    else:
        header = "上一版处方被 Tool 5 拦截，请针对以下全部问题逐条修正后重新生成（其余正确部分保持不变）："
    lines = [header]
    for _sort_key, label, messages in sorted(grouped.values(), key=lambda group: group[0]):
        lines.append(f"【{label}】" + "；".join(messages) + "。")
    return "\n".join(lines)


def _remediation(item: dict[str, Any]) -> str:
    rule = item.get("rule", "")
    value = item.get("value")
    limit = item.get("limit")

    template = _MSG_TEMPLATES.get(rule)
    if template is not None:
        return template.format(value=value, limit=limit)

    if rule.endswith("_outside_feasible_region"):
        return f"{_param_label(rule, '_outside_feasible_region')}「{value}」不在可行域允许集合 {limit} 内，请改选其中之一"
    if rule.endswith("_out_of_range"):
        return f"{_param_label(rule, '_out_of_range')} {value} 超出允许范围 {limit}"
    if rule.endswith("_mismatch"):
        return f"{_param_label(rule, '_mismatch')} 应为 {limit}（当前 {value}），请按 volume_options 提供的值/公式重算"
    if rule.endswith("_not_int"):
        return f"{_param_label(rule, '_not_int')} 必须为整数"
    if rule.endswith("_not_number"):
        return f"{_param_label(rule, '_not_number')} 必须为数值"
    if rule.endswith("_missing"):
        return f"缺少必填项 {_param_label(rule, '_missing')}"
    if rule.startswith("restricted_action_"):
        return f"该式取值须落在受限允许集合 {limit} 内（机械/结构性受限，必须降档）"
    if rule.endswith("_not_considered"):
        return f"未体现已考虑「{value}」：{limit}"
    return f"{rule}：当前 {value}，应满足 {limit}"


def _param_label(rule: str, suffix: str) -> str:
    token = rule[: -len(suffix)]
    # parameter_5_sets_per_session / parameter_6_frequency_per_week / ...
    for param, label in _PARAM_LABELS.items():
        if token.endswith(param):
            return label
    return token


def _violation(
    rule: str,
    form_id: int | None,
    value: Any,
    limit: Any,
    *,
    severity: str = "hard",
) -> dict[str, Any]:
    return {
        "rule": rule,
        "form_id": form_id,
        "value": value,
        "limit": limit,
        "severity": severity,
    }
