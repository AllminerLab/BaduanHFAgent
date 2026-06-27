"""Shared constants for the Baduanjin prescription framework."""

BADUANJIN_FORMS = [
    {"form_id": 1, "name": "双手托天理三焦", "standard_cycles": 4},
    {"form_id": 2, "name": "左右开弓似射雕", "standard_cycles": 6},
    {"form_id": 3, "name": "调理脾胃须单举", "standard_cycles": 6},
    {"form_id": 4, "name": "五劳七伤往后瞧", "standard_cycles": 6},
    {"form_id": 5, "name": "摇头摆尾去心火", "standard_cycles": 6},
    {"form_id": 6, "name": "两手攀足固肾腰", "standard_cycles": 4},
    {"form_id": 7, "name": "攒拳怒目增气力", "standard_cycles": 6},
    {"form_id": 8, "name": "背后七颠百病消", "standard_cycles": 7},
]

FORM_IDS = [item["form_id"] for item in BADUANJIN_FORMS]
FORM_NAME_BY_ID = {item["form_id"]: item["name"] for item in BADUANJIN_FORMS}
STANDARD_CYCLES_BY_ID = {
    item["form_id"]: item["standard_cycles"] for item in BADUANJIN_FORMS
}

# Cycle direction: per the four-indicator stratification spec, the group standard
# is the BASE quantity (not a ceiling). Cycles may be titrated UP above standard
# for higher-function patients, bounded by MAX_CYCLES (§1.3 cycle range 1-10).
# This applies to CYCLES ONLY — amplitude/tempo/rest remain "只减不增".
MAX_CYCLES = 10
HAND_FORMS = [1, 3, 6]  # 手部动作 — B 适度上调侧重
LEG_FORMS = [2, 5, 7]   # 腿部/马步动作 — C 较大上调侧重

AMPLITUDE_LEVELS = ["坐式", "简化", "标准"]
BASE_AMPLITUDE_LEVELS = ["简化", "标准"]
TEMPO_LEVELS = ["慢速", "标准"]
REST_LEVELS = ["标准", "延长"]

STOP_RULES = [
    "胸闷",
    "心前区疼痛",
    "心悸",
    "严重心律不齐",
    "头晕黑矇",
    "明显呼吸困难",
    "严重关节肌肉疼痛",
]

SOP_VOLUME_LEVELS = [
    {
        "level": 1,
        "sets_per_session": 1,
        "times_per_day": 1,
        "frequency_per_week": 7,
        "weekly_sets": 7,
        "weekly_minutes_at_12min": 84,
    },
    {
        "level": 2,
        "sets_per_session": 1,
        "times_per_day": 2,
        "frequency_per_week": 5,
        "weekly_sets": 10,
        "weekly_minutes_at_12min": 120,
    },
    {
        "level": 3,
        "sets_per_session": 1,
        "times_per_day": 2,
        "frequency_per_week": 6,
        "weekly_sets": 12,
        "weekly_minutes_at_12min": 144,
    },
    {
        "level": 4,
        "sets_per_session": 1,
        "times_per_day": 2,
        "frequency_per_week": 7,
        "weekly_sets": 14,
        "weekly_minutes_at_12min": 168,
    },
    {
        "level": 5,
        "sets_per_session": 1,
        "times_per_day": 3,
        "frequency_per_week": 5,
        "weekly_sets": 15,
        "weekly_minutes_at_12min": 180,
    },
    {
        "level": 6,
        "sets_per_session": 1,
        "times_per_day": 3,
        "frequency_per_week": 6,
        "weekly_sets": 18,
        "weekly_minutes_at_12min": 216,
    },
    {
        "level": 7,
        "sets_per_session": 1,
        "times_per_day": 4,
        "frequency_per_week": 5,
        "weekly_sets": 20,
        "weekly_minutes_at_12min": 240,
    },
    {
        "level": 8,
        "sets_per_session": 1,
        "times_per_day": 4,
        "frequency_per_week": 6,
        "weekly_sets": 24,
        "weekly_minutes_at_12min": 288,
    },
]

ABSOLUTE_CONTRAINDICATIONS = {
    "acute_decompensated_hf": "急性失代偿心衰",
    "severe_arrhythmia": "严重心律失常",
    "unstable_angina": "不稳定心绞痛",
    "acute_myocarditis_pericarditis": "急性心肌炎或心包炎",
    "severe_aortic_stenosis": "严重主动脉瓣狭窄",
    "acute_pe_dvt": "急性肺栓塞或深静脉血栓",
    "fever_or_systemic_illness": "急性全身疾病或发热",
}
