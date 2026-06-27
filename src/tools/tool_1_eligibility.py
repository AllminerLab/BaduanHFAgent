"""Tool 1: exercise eligibility screening."""

from __future__ import annotations

from typing import Any

from constants import ABSOLUTE_CONTRAINDICATIONS
from profile_access import data_section


def screen_exercise_eligibility(profile: dict[str, Any]) -> dict[str, Any]:
    """Return binary exercise eligibility (allow/refuse) with reasons.

    - refuse: absolute contraindications or high-risk functional thresholds.
    - allow: VO2peak/AT, 6MWD, and NYHA are all outside high-risk thresholds.
      ``data_incomplete`` flags the "allowed but key safety data missing/uncertain"
      case, which keeps the downstream feasible region / dose conservative.
    """

    refuse_reasons: list[str] = []
    annotations: list[dict[str, Any]] = []
    cpet = data_section(profile, "CPX")
    tests = data_section(profile, "TEST")
    check = data_section(profile, "CHECK")

    acute_conditions: set[str] = set()
    absolute_hits: list[str] = []
    for key, label in ABSOLUTE_CONTRAINDICATIONS.items():
        if key in acute_conditions:
            absolute_hits.append(label)
            refuse_reasons.append(f"存在绝对禁忌证：{label}。")

    vo2_peak = cpet.get("vo2_peak")
    vo2_at = cpet.get("vo2_at")
    six_mwd = tests.get("six_mwd")
    nyha = check.get("nyha")
    criteria = [
        _metric_criterion(
            key="vo2_peak",
            name="VO2peak",
            value=vo2_peak,
            unit="mL/kg/min",
            refuse_condition="VO2peak < 10",
            allow_condition="VO2peak >= 10",
            refuses=_lt(vo2_peak, 10),
        ),
        _metric_criterion(
            key="vo2_at",
            name="AT",
            value=vo2_at,
            unit="mL/kg/min",
            refuse_condition="AT < 8",
            allow_condition="AT >= 8",
            # Weber: VO2peak 为主锚；AT 只在 VO2peak 缺失时才作为拒绝依据（冲突以 VO2peak 为准）。
            refuses=(vo2_peak is None and _lt(vo2_at, 8)),
        ),
        _metric_criterion(
            key="six_mwd",
            name="6MWD",
            value=six_mwd,
            unit="m",
            refuse_condition="6MWD < 150",
            allow_condition="6MWD >= 150",
            refuses=_lt(six_mwd, 150),
        ),
        _metric_criterion(
            key="nyha",
            name="NYHA",
            value=nyha,
            unit="级",
            refuse_condition="NYHA IV",
            allow_condition="NYHA I-III",
            refuses=_gte(nyha, 4),
        ),
    ]
    if absolute_hits:
        criteria.insert(
            0,
            {
                "key": "absolute_contraindications",
                "name": "绝对禁忌证",
                "value": absolute_hits,
                "unit": "",
                "refuse_condition": "存在任一绝对禁忌证",
                "allow_condition": "未见绝对禁忌证",
                "status": "hit_refuse",
                "detail": f"存在绝对禁忌证：{'、'.join(absolute_hits)}，命中拒绝条件。",
            },
        )

    # VO2peak 在且 AT<8：AT 被 VO2peak 覆盖，criteria 里如实标注（避免误显为“满足 AT≥8”）。
    if vo2_peak is not None and _lt(vo2_at, 8):
        for item in criteria:
            if item.get("key") == "vo2_at":
                item["status"] = "overridden"
                item["detail"] = (
                    f"AT 为 {_format_value(vo2_at, 'mL/kg/min')}（< 8），"
                    f"但 VO2peak 为 {_format_value(vo2_peak, 'mL/kg/min')}（≥ 10）不属高危，"
                    "按规则以 VO2peak 为主，AT 不作为拒绝依据。"
                )

    # High-risk functional thresholds. High risk is not recommended for exercise.
    # Weber：VO2peak 为主锚。AT < 8 只在 VO2peak 缺失时才触发拒绝；二者都在且冲突时以 VO2peak 为准。
    if _lt(vo2_peak, 10):
        refuse_reasons.append(f"VO2peak 为 {_format_value(vo2_peak, 'mL/kg/min')}，命中 Weber D：VO2peak < 10。")
    if vo2_peak is None and _lt(vo2_at, 8):
        refuse_reasons.append(f"AT 为 {_format_value(vo2_at, 'mL/kg/min')}，命中 Weber D：AT < 8。")
    if _lt(six_mwd, 150):
        refuse_reasons.append(f"6MWD 为 {_format_value(six_mwd, 'm')}，命中 6MWD < 150 m 高危条件。")
    if _gte(nyha, 4):
        refuse_reasons.append(f"NYHA 为 {_format_value(nyha, '级')}，命中 NYHA IV 高危条件。")

    if refuse_reasons:
        rationale = _build_rationale(
            "refuse",
            criteria,
            refuse_reasons,
            data_incomplete=False,
            uncertainty_reasons=[],
        )
        return {
            "eligibility_status": "refuse",
            "risk_level": "high",
            "rationale": rationale,
            "criteria": criteria,
            "data_incomplete": False,
            "annotations": annotations,
        }

    # Allowed — flag the conservative "allowed but data missing/uncertain" cases.
    allow_reasons: list[str] = []
    data_incomplete = False
    uncertainty_reasons: list[str] = []

    # Weber 冲突以 VO2peak 为主：AT<8 但 VO2peak≥10 时已放行，理由里如实点明 AT 偏低。
    if vo2_peak is not None and _lt(vo2_at, 8):
        uncertainty_reasons.append(
            f"AT {_format_value(vo2_at, 'mL/kg/min')} < 8 偏低，"
            f"但以 VO2peak {_format_value(vo2_peak, 'mL/kg/min')}（≥10、不属高危）为主，未据此拒绝"
        )

    if vo2_peak is None and vo2_at is None and six_mwd is None and nyha is None:
        data_incomplete = True
        allow_reasons.append("CPET（VO2peak/AT）、6MWD、NYHA 均缺失。")
        annotations.append(
            {
                "type": "data_incomplete",
                "detail": "缺关键运动资格数据，后续可行域按最保守策略下调。",
                "affected": "global",
            }
        )

    if cpet.get("rer_peak") is not None and cpet.get("rer_peak") < 1.05:
        data_incomplete = True
        uncertainty_reasons.append(
            f"RER 为 {_format_value(cpet.get('rer_peak'), '')}，低于 1.05，提示 CPET 可能为次极量。"
        )
        annotations.append(
            {
                "type": "submaximal_test",
                "detail": "RER < 1.05，CPET 可能未达极量，分层置信度下调，按保守处理。",
                "affected": "function_layer",
            }
        )

    risk_level = "unknown" if data_incomplete else "low"

    if not allow_reasons:
        allow_reasons.append("未命中高危标准。")

    rationale = _build_rationale(
        "allow",
        criteria,
        allow_reasons,
        data_incomplete=data_incomplete,
        uncertainty_reasons=uncertainty_reasons,
    )

    return {
        "eligibility_status": "allow",
        "risk_level": risk_level,
        "rationale": rationale,
        "criteria": criteria,
        "data_incomplete": data_incomplete,
        "annotations": annotations,
    }


def _metric_criterion(
    *,
    key: str,
    name: str,
    value: Any,
    unit: str,
    refuse_condition: str,
    allow_condition: str,
    refuses: bool,
) -> dict[str, Any]:
    if value is None:
        status = "missing"
        detail = f"{name} 缺失，无法用于本轮资格判断。"
    elif refuses:
        status = "hit_refuse"
        detail = f"{name} 为 {_format_value(value, unit)}，命中 {refuse_condition}。"
    else:
        status = "pass"
        detail = f"{name} 为 {_format_value(value, unit)}，满足 {allow_condition}，未命中高危标准。"
    return {
        "key": key,
        "name": name,
        "value": value,
        "unit": unit,
        "refuse_condition": refuse_condition,
        "allow_condition": allow_condition,
        "status": status,
        "detail": detail,
    }


def _build_rationale(
    decision: str,
    criteria: list[dict[str, Any]],
    decision_reasons: list[str],
    *,
    data_incomplete: bool,
    uncertainty_reasons: list[str],
) -> str:
    metric_text = _compact_metric_sentence(criteria)
    missing_text = _compact_missing_sentence(criteria)
    uncertainty_text = _compact_uncertainty_sentence(uncertainty_reasons)
    if decision == "refuse":
        absolute_text = _compact_absolute_sentence(criteria)
        hit_text = _compact_refuse_sentence(criteria)
        return f"不建议自动生成处方。{absolute_text}{metric_text}{missing_text}{hit_text}"

    if data_incomplete:
        prefix = "允许进入后续流程，但按保守策略处理。"
    else:
        prefix = "允许进入后续处方。"
    # The decision (prefix) already states the verdict; the listed metrics/missing
    # fields are the justification — no trailing "未命中高危标准" bookend needed.
    return f"{prefix}{metric_text}{missing_text}{uncertainty_text}"


def _compact_absolute_sentence(criteria: list[dict[str, Any]]) -> str:
    absolute = next(
        (
            item for item in criteria
            if item.get("key") == "absolute_contraindications" and item.get("status") == "hit_refuse"
        ),
        None,
    )
    if absolute:
        return f"存在绝对禁忌证：{'、'.join(absolute.get('value') or [])}。"
    return ""


def _compact_metric_sentence(criteria: list[dict[str, Any]]) -> str:
    present = [
        item for item in criteria
        if item.get("key") != "absolute_contraindications" and item.get("status") != "missing"
    ]
    if present:
        return "当前指标：" + "、".join(_criterion_value_text(item) for item in present) + "。"
    return ""


def _compact_missing_sentence(criteria: list[dict[str, Any]]) -> str:
    missing = [
        item for item in criteria
        if item.get("key") != "absolute_contraindications" and item.get("status") == "missing"
    ]
    if missing:
        missing_names = "、".join(item["name"] for item in missing)
        suffix = "均缺失" if len(missing) == 4 else "缺失"
        return f"{missing_names} {suffix}。"
    return ""


def _compact_refuse_sentence(criteria: list[dict[str, Any]]) -> str:
    # Absolute contraindications are stated in their own clause (see
    # _compact_absolute_sentence), so they are excluded from the high-risk list here.
    hits = [
        item for item in criteria
        if item.get("status") == "hit_refuse" and item.get("key") != "absolute_contraindications"
    ]
    if not hits:
        return ""
    return "命中以下高危标准：" + "；".join(_compact_hit_text(item) for item in hits) + "。"


def _compact_uncertainty_sentence(uncertainty_reasons: list[str]) -> str:
    if not uncertainty_reasons:
        return ""
    cleaned = "；".join(_strip_terminal_punctuation(reason) for reason in uncertainty_reasons)
    return f"需注意：{cleaned}。"


def _criterion_value_text(item: dict[str, Any]) -> str:
    return f"{item['name']} 为 {_format_value(item.get('value'), item.get('unit', ''))}"


def _compact_hit_text(item: dict[str, Any]) -> str:
    if item.get("key") == "absolute_contraindications":
        return f"绝对禁忌证（{'、'.join(item.get('value') or [])}）"
    key = item.get("key")
    value = _format_value(item.get("value"), item.get("unit", ""))
    if key == "vo2_peak":
        return f"VO2peak {value} < 10 mL/kg/min"
    if key == "vo2_at":
        return f"AT {value} < 8 mL/kg/min"
    if key == "six_mwd":
        return f"6MWD {value} < 150 m"
    if key == "nyha":
        return "NYHA IV 级"
    return (
        f"{item['name']} 为 {value}，"
        f"命中 {item.get('refuse_condition')} 的高危标准"
    )


def _format_value(value: Any, unit: str) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    if unit == "级":
        return f"{text}{unit}"
    return f"{text} {unit}".strip()


def _strip_terminal_punctuation(text: str) -> str:
    return text.rstrip("。；; ")


def _lt(value: Any, limit: float) -> bool:
    return value is not None and value < limit


def _lte(value: Any, limit: float) -> bool:
    return value is not None and value <= limit


def _gte(value: Any, limit: float) -> bool:
    return value is not None and value >= limit


def _between(value: Any, low: float, high: float) -> bool:
    return value is not None and low <= value < high
