"""Tool 3: risk assessment — enumerate patient risk / limitation signals.

Tool 3 only ASSESSES: it produces a discrete risk list, each item carrying the
patient's actual value + threshold + a human-readable detail (Tool 1 / Tool 2
explainability style). It does NOT project risks onto per-form parameters or dose.

The risk -> form/parameter/dose matching (which form, which 幅度/节奏/休息 档, which
dose cap) is done by Tool 4 (action matching) via the KB4 mapping table, keyed on
each risk's ``risk_id``. This keeps assessment (Tool 3) and projection (Tool 4)
cleanly separated — 评估 vs 匹配.
"""

from __future__ import annotations

from typing import Any

from constants import STOP_RULES
from profile_access import data_bool, data_number, data_section


DEVICE_KEYS = {"pacemaker": "起搏器", "icd": "ICD", "crt": "CRT"}
VALVULAR_FIELDS = {
    "ecoar": "主动脉瓣重度返流",
    "ecomr": "二尖瓣重度返流",
    "ecoas": "主动脉瓣重度狭窄",
    "ecoms": "二尖瓣重度狭窄",
}
VALVE_DATA_FIELDS = {
    "ecoar": "valve_ar",
    "ecomr": "valve_mr",
    "ecoas": "valve_as",
    "ecoms": "valve_ms",
}


def assess_risk(profile: dict[str, Any], eligibility: dict[str, Any]) -> dict[str, Any]:
    medication = data_section(profile, "MEDICATION")
    ecg = data_section(profile, "ECG")
    tests = data_section(profile, "TEST")

    risks: list[dict[str, Any]] = []

    # --- 术式 / 设备 ---
    if data_bool(profile, "PACE", "device"):
        device_type = data_section(profile, "PACE").get("device_type")
        text = _device_label(device_type)
        risks.append(_risk("device_implant", "device", text, None, f"植入{text}：上肢过头动作受限。"))

    # --- 用药 ---
    if medication.get("beta_blocker"):
        risks.append(_risk("beta_blocker", "medication", "β受体阻滞剂", None, "服用 β 受体阻滞剂：运动心率反应可能钝化。"))

    # --- 通气 ---
    if data_bool(profile, "HIST", "copd"):
        risks.append(_risk("copd", "ventilatory", "COPD 病史", None, "COPD 病史：通气受限。"))
    ve_vco2_slope = data_number(profile, "CPX", "ve_vco2_slope")
    if _gte(ve_vco2_slope, 35):
        v = _fmt(ve_vco2_slope)
        risks.append(_risk("high_ve_vco2", "ventilatory", f"VE/VCO₂ {v}", "≥35", f"VE/VCO₂ {v}≥35：通气效率差。"))
    dyspnea = data_number(profile, "SYMPTOM", "dyspnea")
    if _gte(dyspnea, 6):
        v = _fmt(dyspnea)
        risks.append(_risk("dyspnea_high", "symptom", f"气促评分 {v}", "≥6", f"气促评分 {v}≥6：气促明显。"))

    # --- 心律 ---
    if ecg.get("arrhythmia"):
        risks.append(_risk("arrhythmia", "rhythm", "心电/Holter 示心律失常", None, "存在心律失常证据：心律风险。"))

    # --- 血流动力学 / 严重度 ---
    lvef = data_number(profile, "ECHO", "lvef")
    if _lte(lvef, 30):
        v = _fmt(lvef)
        risks.append(_risk("low_lvef", "hemodynamic", f"LVEF {v}%", "≤30%", f"LVEF {v}%≤30%：左室收缩功能显著降低。"))
    nt_pro_bnp = data_number(profile, "LABS", "nt_pro_bnp")
    if _gte(nt_pro_bnp, 2000):
        v = _fmt(nt_pro_bnp)
        risks.append(_risk("high_bnp", "hemodynamic", f"NT-proBNP {v}", "≥2000", f"NT-proBNP {v}≥2000：心衰负荷重。"))
    hrr = data_number(profile, "CPX", "hrr")
    if _lte(hrr, 12):
        v = _fmt(hrr)
        risks.append(_risk("low_hrr", "hemodynamic", f"HRR {v}", "≤12", f"心率储备 HRR {v}≤12：自主调节能力差。"))
    peak_sbp = data_number(profile, "CPX", "peak_sbp")
    if _gte(peak_sbp, 200):
        v = _fmt(peak_sbp)
        risks.append(_risk("high_peak_sbp", "hemodynamic", f"峰值SBP {v}", "≥200", f"峰值收缩压 {v}≥200：血压反应过强。"))
    sbp = data_number(profile, "CHECK", "sbp")
    if _gte(sbp, 160):
        v = _fmt(sbp)
        risks.append(_risk("high_rest_sbp", "hemodynamic", f"静息SBP {v}", "≥160", f"静息收缩压 {v}≥160：血压偏高。"))
    rest_hr = data_number(profile, "CHECK", "rest_hr")
    if _gte(rest_hr, 100):
        v = _fmt(rest_hr)
        risks.append(_risk("high_rest_hr", "hemodynamic", f"静息HR {v}", "≥100", f"静息心率 {v}≥100：偏快。"))

    # --- 动作受限（病史 / 查体 / 症状）---
    if data_bool(profile, "HIST", "stroke"):
        risks.append(_risk("stroke", "physical", "中风病史", None, "中风病史：肢体活动受限。"))
    pain = data_number(profile, "EQ5D", "pain")
    if _gte(pain, 4):
        v = _fmt(pain)
        risks.append(_risk("body_pain", "physical", f"EQ5D 疼痛 {v}", "≥4", f"EQ5D 疼痛/不适 {v}≥4：身体疼痛较重。"))
    edema = _max_present(data_number(profile, "CHECK", "edema_left"), data_number(profile, "CHECK", "edema_right"))
    if _gte(edema, 2):
        v = _fmt(edema)
        risks.append(_risk("lower_limb_edema", "physical", f"下肢水肿 {v} 度", "2–3 度", f"下肢水肿 {v} 度：下肢负荷受限。"))
    if data_bool(profile, "HIST", "pad"):
        risks.append(_risk("pad_lower_limb", "physical", "下肢动脉闭塞", None, "下肢动脉闭塞：下蹲/马步受限。"))
    mobility_time = data_number(profile, "TEST", "mobility_time")
    mobility_class = data_number(profile, "TEST", "mobility_class")
    if _gte(mobility_time, 20) or mobility_class in {3, 4}:
        parts = []
        if mobility_time is not None:
            parts.append(f"3米起立行走 {_fmt(mobility_time)}s")
        if mobility_class is not None:
            parts.append(f"活动分级 {_fmt(mobility_class)}")
        text = "、".join(parts) or "平衡能力差"
        risks.append(_risk("poor_balance", "physical", text, None, f"{text}：平衡能力差。"))
    if tests.get("assistive_device"):
        risks.append(_risk("assistive_device", "physical", "轮椅/拐杖或活动障碍", None, "轮椅/拐杖或活动障碍：整体活动受限。"))
    fatigue = data_number(profile, "SYMPTOM", "fatigue")
    if _gte(fatigue, 6):
        v = _fmt(fatigue)
        risks.append(_risk("fatigue_high", "symptom", f"疲劳评分 {v}", "≥6", f"疲劳评分 {v}≥6：疲劳负担高。"))
    nyha = data_number(profile, "CHECK", "nyha")
    if nyha == 3:
        risks.append(_risk("nyha_iii", "hemodynamic", "NYHA III", "NYHA=3", "NYHA III：运动耐量较低。"))

    # --- 带 CRF 字段的受限表格：局部肌骨/关节动作限制 ---
    if data_bool(profile, "MSK", "cervical"):
        risks.append(_risk("cervical_spine_problem", "physical", "颈椎问题", None, "颈椎问题：转头、摇头相关动作受限。"))
    if data_bool(profile, "MSK", "shoulder"):
        risks.append(_risk("shoulder_problem", "physical", "肩关节问题", None, "肩关节问题：上肢过头动作受限。"))
    if data_bool(profile, "MSK", "lumbar"):
        risks.append(_risk("lumbar_spine_problem", "physical", "腰椎问题", None, "腰椎问题：转腰、前屈相关动作受限。"))
    if data_bool(profile, "MSK", "knee"):
        risks.append(_risk("knee_problem", "physical", "膝关节术后/明显受限", None, "膝关节术后/明显受限：马步动作受限。"))

    baduanjin_learning = data_number(profile, "MSK", "learned_baduanjin")
    if baduanjin_learning == 0:
        risks.append(_risk("baduanjin_newcomer", "physical", "八段锦新手", None, "八段锦新手：既往未学过八段锦。"))

    # --- 自我效能（依从性）---
    see = data_number(profile, "SEE", "seesum")
    if _lte(see, 50):
        v = _fmt(see)
        risks.append(_risk("low_self_efficacy", "psychosocial", f"自我效能 SEESUM {v}", "≤50", f"自我效能 SEESUM {v}≤50：依从性偏弱。"))

    # --- 全局安全相关风险（冠心病/房颤、重度瓣膜）---
    if data_bool(profile, "HIST", "cad") or data_bool(profile, "HIST", "af"):
        risks.append(_risk("cad_or_af", "dose_safety", "冠心病/房颤", None, "合并冠心病/房颤。"))
    severe_valves = [
        f"{label}（{field.upper()}=3）"
        for field, label in VALVULAR_FIELDS.items()
        if _number_equals(data_number(profile, "ECHO", VALVE_DATA_FIELDS[field]), 3)
    ]
    if severe_valves:
        text = "、".join(severe_valves)
        risks.append(_risk("severe_valvular", "dose_safety", text, "ECHO=3", f"原始 ECHO 提示{text}，处方需体现已考虑重度瓣膜反流/狭窄情况。"))

    # Tool 3 only COLLECTS facts: one risk = one self-contained dict (each carries
    # its own value/threshold/detail). No flattened narrative is synthesized here —
    # a summary, if needed, is derivable from the per-risk details by the consumer.
    # Derived conclusions are made downstream:
    #   - β-blocker remains a patient fact / KB trigger only; it is not projected by
    #     Tool 4 because it is absent from the CRF-linked restricted-action table.
    #   - data_incomplete -> conservative posture is driven by Tool 1 / feasible_region
    return {
        "risks": risks,
        "stop_rules": STOP_RULES,
    }


def _risk(risk_id: str, category: str, value_text: str, threshold: str | None, detail: str) -> dict[str, Any]:
    return {
        "risk_id": risk_id,
        "category": category,
        "value_text": value_text,
        "threshold": threshold,
        "detail": detail,
    }


def _device_label(device_type: Any) -> str:
    if device_type in (None, ""):
        return "心脏植入设备"
    text = str(device_type).strip()
    if text.replace(".", "", 1).isdigit():
        return "心脏植入设备"
    normalized = text.lower()
    if normalized in DEVICE_KEYS:
        return DEVICE_KEYS[normalized]
    return text


def _max_present(*values: Any) -> Any:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _gte(value: Any, limit: float) -> bool:
    return value is not None and value >= limit


def _lte(value: Any, limit: float) -> bool:
    return value is not None and value <= limit


def _number_equals(value: Any, expected: float) -> bool:
    return value is not None and value == expected
