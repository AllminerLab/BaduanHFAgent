"""Knowledge base: static core knowledge + patient-relevant guideline cards."""

from __future__ import annotations

from typing import Any
import json
import os

from profile_access import data_section


# ===== 静态核心：每例必备、对所有患者相同 =====

KB1_BADUANJIN_STANDARD = """
KB1 八段锦标准与动作拆解：
八段锦训练应保留八式完整结构：1双手托天理三焦、2左右开弓似射雕、3调理脾胃须单举、4五劳七伤往后瞧、5摇头摆尾去心火、6两手攀足固肾腰、7攒拳怒目增气力、8背后七颠百病消。各式主要动作需求：第1/3式上肢上举过头；第2/5/七式马步/下蹲、下肢负荷较高；第4式转头后瞧（颈部）；第5式转头+下蹲+重心转移（颈+下肢+平衡）；第6式前屈转腰（腰）；第8式提踵（平衡）。每式内部关键动作序列不可拆分，左右对称动作按完整循环执行。幅度与节奏应从标准动作向更保守方向调整，不宜超过标准动作要求；极端受限时可采用坐式、简化动作、慢速和更少循环。
"""

KB2_HF_SAFETY = """
KB2 心衰康复安全：
绝对禁忌包括急性失代偿心衰、严重心律失常、不稳定心绞痛、急性心肌炎/心包炎、严重主动脉瓣狭窄、急性肺栓塞或深静脉血栓、急性全身疾病发热。Weber D（VO2peak<10）、6MWD<300m、NYHA IV 不生成自动处方。β受体阻滞剂使用者避免只按心率设靶。停练红线包括胸闷、心前区疼痛、心悸、严重心律不齐、头晕黑矇、明显呼吸困难、严重关节肌肉疼痛。
"""

KB3_FUNCTION_DOSE = """
KB3 运动能力评估与剂量：
CPET（VO2peak/AT）、6MWD 与 Borg/RPE 可用于解释患者运动耐量和训练强度感受。运动能力低者通常维持标准循环数或采取更保守处方；运动能力中、高者可在安全边界内逐步上调训练刺激。循环数以标准为基础量、封顶10次/式；高危或数据不足时不宜上调。NYHA 不适合作为运动能力分层的主要依据，更适合用于安全风险判断。
"""

# KB4 is kept as internal documentation only and is not injected into prompts.
KB4_ACTION_LIMITS = """
KB4 动作需求-受限映射（内部规则源；不注入 prompt）：
中风、明显身体疼痛、活动障碍可触发全套幅度简化和慢速。起搏器/ICD/CRT 限制第1/3式上肢过头幅度。下肢水肿、下肢动脉闭塞、平衡差主要限制第2/5/7式马步和第8式提踵/重心转移。COPD、VE/VCO2斜率升高、气促明显、心律风险、焦虑明显倾向慢速。NYHA III、LVEF低、HRR差、峰值血压高、NT-proBNP高、静息心率高倾向延长休息。CRF 缺少膝/肩/腰/颈等肌骨部位结构化字段时，不可臆断具体部位，只能标注 unsupported_signal。
"""

_STATIC_KB = [KB1_BADUANJIN_STANDARD, KB2_HF_SAFETY, KB3_FUNCTION_DOSE]


# ===== 动态卡片：冻结在同包 dynamic_cards.json =====
# Cards are curated from guideline documents and selected by patient-relevant signals.
def _load_dynamic_cards() -> list[dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "dynamic_cards.json")
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return []
    for card in raw:
        card["triggers"] = set(card.get("triggers") or [])
    return raw


DYNAMIC_CARDS: list[dict[str, Any]] = _load_dynamic_cards()


def get_knowledge_bundle(include_kb: bool = True, context: dict[str, Any] | None = None) -> str:
    """Static core (KB1-3) plus patient-relevant guideline cards."""

    if not include_kb:
        return ""

    parts = [block.strip() for block in _STATIC_KB]

    signals = _patient_signals(context)
    cards = [card for card in DYNAMIC_CARDS if signals & card["triggers"]]
    if cards:
        parts.append("KB-动态 按患者情况匹配的通用医学知识摘录（仅作辅助介绍）：")
        for card in cards:
            parts.append(f"（{card['source']}）{card['text']}")

    return "\n".join(parts)


def _patient_signals(context: dict[str, Any] | None) -> set[str]:
    """Deterministic retrieval keys for matching DYNAMIC_CARDS.triggers.

    Keyed on available patient data and upstream structured risk/function outputs.
    """

    signals: set[str] = set()
    if not context:
        return signals

    # 评估模态：患者画像里有哪种功能评估数据
    profile = context.get("patient_profile") or {}
    cpet = data_section(profile, "CPX")
    if cpet.get("vo2_peak") is not None or cpet.get("vo2_at") is not None:
        signals.add("has_cpet")
    if data_section(profile, "TEST").get("six_mwd") is not None:
        signals.add("has_6mwd")

    tool_outputs = context.get("tool_outputs") or {}

    # 资格状态
    if (tool_outputs.get("tool_1_eligibility") or {}).get("data_incomplete"):
        signals.add("data_incomplete")

    # 功能层 / 解析指标
    function_layer = tool_outputs.get("tool_2_function_layer") or {}
    level = function_layer.get("candidate_level")
    if level == "low":
        signals.add("function_low")
    elif level == "medium":
        signals.add("function_mid")
    elif level == "high":
        signals.add("function_high")
    resolved_by = function_layer.get("resolved_by")
    if resolved_by == "borg":
        signals.add("resolved_borg")
    # 风险信号：直接用 risk_id 命中对应接地卡
    risk_constraints = tool_outputs.get("tool_3_risk_constraints") or {}
    for risk in risk_constraints.get("risks") or []:
        risk_id = risk.get("risk_id")
        if risk_id:
            signals.add(risk_id)

    return signals
