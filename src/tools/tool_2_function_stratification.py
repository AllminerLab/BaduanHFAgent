"""Tool 2: function stratification for dose range."""

from __future__ import annotations

from typing import Any

from profile_access import data_section


LOW = "low"
MID = "medium"
HIGH = "high"

LEVEL_LABELS = {
    LOW: "运动能力低",
    MID: "运动能力中",
    HIGH: "运动能力高",
}
LEVEL_ORDER = {LOW: 0, MID: 1, HIGH: 2}

# Tool 2 only BOUNDS the cycle candidate range (ceiling). The Skill (LLM) selects
# the exact per-form cycles within [standard, standard+ceiling], guided by the
# sub-band table that now lives in the Full Skill. Standard cycles are defined
# per form in constants.py; cap = MAX_CYCLES.
# Forms 1/3/6 = hand, 2/5/7 = leg/horse-stance. low: no room;
# medium: 第一式 +6, 第三式 +4, 第六式 +2, 第二/五/七式 +2;
# high: 第一式 +6, 第三式 +4, 第六式 +6, 第二/五/七式 +4.
HAND_FORMS = [1, 3, 6]
LEG_FORMS = [2, 5, 7]
LEVEL_CEILING_INCREMENT = {
    LOW: {},
    MID: {1: 6, 3: 4, 6: 2, 2: 2, 5: 2, 7: 2},
    HIGH: {1: 6, 3: 4, 6: 6, 2: 4, 5: 4, 7: 4},
}

# Borg-only fallback (no baduanjin CPET intensity, no standard CPET): keep cycles
# at standard and titrate 参数⑤ sets-per-session instead.
BORG_SETS_BY_LEVEL = {LOW: 1, MID: 2, HIGH: 3}

# Candidate cycle range guidance (the Skill refines the exact pick within range).
CYCLE_RANGE_GUIDANCE = {
    LOW: "运动能力低：全套维持标准循环数（第1式4，第2/3/4/5/7式6，第6式4，第8式7）。",
    MID: "运动能力中：第一式可由4上调至10，第三式由6上调至10，第六式由4上调至6，第二/五/七式由6上调至8；第四/八式维持标准。具体每式循环数由 Skill 按指标档位在范围内选定。",
    HIGH: "运动能力高：第一式可由4上调至10，第三式由6上调至10，第六式由4上调至10，第二/五/七式由6上调至10；第四/八式维持标准。具体每式循环数由 Skill 按指标档位在范围内选定。",
}

# Conflict-resolution priority among the four exercise-capacity indicators
# (review decision 3,1,4,2): baduanjin-specific CPET intensity (gold standard)
# > standard CPET > baduanjin Borg > 6MWD. The highest-priority indicator that
# has data decides the function level; lower-priority indicators are ignored on conflict
# (no most-conservative pooling, no weighting). When none of the four anchors
# has data, no level is inferred and the caller defaults to 运动能力低.
# NYHA is deliberately NOT a stratification anchor.
PRIORITY_ORDER = ["baduanjin_cpx", "cpet", "borg", "six_mwd"]

# Display labels for the four anchors, in priority order. Used to build the
# per-indicator criteria list and the patient-specific reason summary (Tool 1 style).
INDICATOR_META = [
    ("baduanjin_cpx", "处方前通用八段锦CPET强度"),
    ("cpet", "CPET"),
    ("borg", "处方前通用八段锦Borg"),
    ("six_mwd", "6MWD"),
]
PRIORITY_TEXT = "处方前通用八段锦CPET强度>CPET>处方前通用八段锦Borg>6MWD"


def stratify_function(profile: dict[str, Any], eligibility: dict[str, Any]) -> dict[str, Any]:
    cpet = data_section(profile, "CPX")
    tests = data_section(profile, "TEST")
    baduanjin_borg = data_section(profile, "BDJ_BORG")
    baduanjin_cpx = data_section(profile, "BDJ_CPX")

    data_tier = _data_tier(cpet, tests, baduanjin_borg, baduanjin_cpx)
    level_votes = _indicator_levels(cpet, tests, baduanjin_borg, baduanjin_cpx)
    resolved_by, resolved_level = _resolve_by_priority(level_votes)
    candidate_level = resolved_level or LOW

    # data_incomplete (Tool 1) conservatively pulls the level down one notch. Keep
    # the indicator's raw level (resolved_level) so the rationale can explain the
    # downgrade against the patient's actual value, mirroring Tool 1's criteria style.
    downgraded = False
    if eligibility.get("data_incomplete"):
        lowered = _downgrade(candidate_level)
        downgraded = lowered != candidate_level
        candidate_level = lowered

    # Tool 2 only bounds the cycle CANDIDATE RANGE (level-level ceiling); the Skill
    # selects the exact per-form cycles within it. Borg-only titrates 参数⑤ sets and
    # keeps cycles at standard. Incomplete data never up-titrates.
    resolving_value = _resolving_value(resolved_by, cpet, tests, baduanjin_borg, baduanjin_cpx)
    if eligibility.get("data_incomplete"):
        cycle_increment, sets_per_session = {}, 1
        dose_detail = "关键数据不足：循环维持标准、套数维持 1。"
    elif resolved_by == "borg":
        sets_per_session = BORG_SETS_BY_LEVEL[candidate_level]
        cycle_increment = {}
        dose_detail = f"仅有八段锦 Borg：循环维持标准，每次套数候选定为 {sets_per_session}。"
    else:
        cycle_increment = dict(LEVEL_CEILING_INCREMENT[candidate_level])
        sets_per_session = 1
        dose_detail = "返回循环候选范围（层级天花板），具体每式循环数由 Skill 在范围内按指标档位选定。"

    route = "precision" if data_tier in {1, 2} else "conservative"
    decision_notes: list[dict[str, Any]] = []
    if len(set(level_votes.values())) > 1:
        decision_notes.append(
            {
                "type": "stratification_priority_resolved",
                "detail": (
                    f"分层指标不一致：{_level_labels(level_votes)}；按优先级"
                    f" {PRIORITY_TEXT}，采用 {resolved_by}={LEVEL_LABELS[candidate_level]}。"
                ),
                "affected": "function_layer",
            }
        )
    if eligibility.get("data_incomplete"):
        decision_notes.append(
            {
                "type": "data_incomplete",
                "detail": "缺关键分层锚点，循环数上界按运动能力低处理。",
                "affected": "cycles",
            }
        )

    criteria = _build_criteria(
        level_votes, resolved_by, resolved_level, cpet, tests, baduanjin_borg, baduanjin_cpx
    )
    rationale = _build_rationale(
        candidate_level, resolved_by, resolved_level, downgraded, criteria, dose_detail
    )

    return {
        "data_tier": data_tier,
        "candidate_level": candidate_level,
        "level_votes": level_votes,
        "resolved_by": resolved_by,
        "resolving_value": resolving_value,
        "route": route,
        "cycle_increment": cycle_increment,
        "sets_per_session": sets_per_session,
        "cycle_range_guidance": CYCLE_RANGE_GUIDANCE[candidate_level],
        "criteria": criteria,
        "rationale": rationale,
        "decision_notes": decision_notes,
    }


def _data_tier(
    cpet: dict[str, Any],
    tests: dict[str, Any],
    baduanjin_borg: dict[str, Any],
    baduanjin_cpx: dict[str, Any],
) -> int:
    # Data-quality tier for the precision/conservative route. NYHA was removed as
    # an anchor per the four-indicator revision (八段锦CPET/CPET/Borg/6MWD);
    # only CPET/6MWD give precise data, Borg-only stays conservative.
    if baduanjin_cpx.get("ave_vo2_pct_vo2peak") is not None:
        return 1
    if cpet.get("vo2_peak") is not None or cpet.get("vo2_at") is not None:
        return 1
    if tests.get("six_mwd") is not None:
        return 2
    if baduanjin_borg.get("borg_avg") is not None:
        return 4
    return 4


def _indicator_levels(
    cpet: dict[str, Any],
    tests: dict[str, Any],
    baduanjin_borg: dict[str, Any],
    baduanjin_cpx: dict[str, Any],
) -> dict[str, str]:
    """Map each available exercise-capacity indicator to low/medium/high."""

    levels: dict[str, str] = {}

    # Baduanjin-specific CPET intensity (gold standard, top priority).
    bdj_cpx = _baduanjin_cpx_level(baduanjin_cpx)
    if bdj_cpx is not None:
        levels["baduanjin_cpx"] = bdj_cpx

    # Standard CPET (VO2peak / AT)
    cpet_votes: list[str] = []
    if cpet.get("vo2_peak") is not None:
        cpet_votes.append(_level_by_threshold(cpet["vo2_peak"], 16, 20))
    if cpet.get("vo2_at") is not None:
        cpet_votes.append(_level_by_threshold(cpet["vo2_at"], 11, 14))
    if cpet_votes:
        levels["cpet"] = min(cpet_votes, key=lambda level: LEVEL_ORDER[level])

    # Baduanjin Borg
    borg = baduanjin_borg.get("borg_avg")
    if borg is not None:
        levels["borg"] = _borg_level(borg)

    # 6MWD
    six_mwd = tests.get("six_mwd")
    if six_mwd is not None:
        levels["six_mwd"] = _level_by_threshold(six_mwd, 300, 450)

    return levels


def _resolve_by_priority(levels: dict[str, str]) -> tuple[str | None, str | None]:
    for key in PRIORITY_ORDER:
        if key in levels:
            return key, levels[key]
    return None, None


def _level_labels(levels: dict[str, str]) -> dict[str, str]:
    return {key: LEVEL_LABELS.get(level, level) for key, level in levels.items()}


def _resolving_value(
    resolved_by: str | None,
    cpet: dict[str, Any],
    tests: dict[str, Any],
    baduanjin_borg: dict[str, Any],
    baduanjin_cpx: dict[str, Any],
) -> float | None:
    if resolved_by == "baduanjin_cpx":
        return _cpx_pct(baduanjin_cpx)
    if resolved_by == "cpet":
        peak = cpet.get("vo2_peak")
        return peak if peak is not None else cpet.get("vo2_at")
    if resolved_by == "borg":
        return baduanjin_borg.get("borg_avg")
    if resolved_by == "six_mwd":
        return tests.get("six_mwd")
    return None


def _build_criteria(
    level_votes: dict[str, str],
    resolved_by: str | None,
    resolved_level: str | None,
    cpet: dict[str, Any],
    tests: dict[str, Any],
    baduanjin_borg: dict[str, Any],
    baduanjin_cpx: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per-anchor breakdown (Tool 1 style): each indicator's patient value, the band
    it falls in, the level it implies, and its role in the decision.

    status: deciding (the priority winner) | concordant (agrees with the winner) |
    overridden (present but lower priority on conflict) | missing.
    """

    criteria: list[dict[str, Any]] = []
    for key, name in INDICATOR_META:
        level = level_votes.get(key)
        value_text = _indicator_value_text(key, cpet, tests, baduanjin_borg, baduanjin_cpx)
        if level is None:
            criteria.append(
                {
                    "key": key,
                    "name": name,
                    "value_text": "",
                    "band": "",
                    "implies_level": None,
                    "implies_level_label": None,
                    "status": "missing",
                    "detail": f"{name} 缺失，未参与分层。",
                }
            )
            continue

        label = LEVEL_LABELS[level]
        band = _indicator_band_text(key, level)
        band_clause = f"（{band}）" if band else ""
        if key == resolved_by:
            status = "deciding"
            detail = f"{name} 为 {value_text}{band_clause}，按优先级作为本次主要参考，倾向按{label}处理。"
        elif level == resolved_level:
            status = "concordant"
            detail = f"{name} 为 {value_text}{band_clause}，方向上也支持{label}。"
        else:
            status = "overridden"
            detail = (
                f"{name} 为 {value_text}{band_clause}，提示{label}；"
                f"但优先级低于本次主要参考指标，因此作为交叉参考记录。"
            )
        criteria.append(
            {
                "key": key,
                "name": name,
                "value_text": value_text,
                "band": band,
                "implies_level": level,
                "implies_level_label": label,
                "status": status,
                "detail": detail,
            }
        )
    return criteria


def _build_rationale(
    candidate_level: str,
    resolved_by: str | None,
    resolved_level: str | None,
    downgraded: bool,
    criteria: list[dict[str, Any]],
    dose_detail: str,
) -> str:
    candidate_label = LEVEL_LABELS[candidate_level]

    if resolved_by is None:
        return (
            f"当前四项分层指标（处方前通用八段锦CPET强度 / CPET / 处方前通用八段锦Borg / 6MWD）均缺失，"
            f"暂按最保守的{candidate_label}处理。{dose_detail}"
        )

    deciding = next(item for item in criteria if item["status"] == "deciding")
    resolved_label = LEVEL_LABELS[resolved_level]
    deciding_band = f"，{deciding['band']}" if deciding["band"] else ""

    parts = [f"当前更适合按{candidate_label}处理。"]
    if downgraded:
        parts.append(
            f"主要参考指标是{deciding['name']}（{deciding['value_text']}{deciding_band}），"
            f"原本指向{resolved_label}；考虑到关键运动资格数据不足，本轮先下调为{candidate_label}。"
        )
    else:
        parts.append(
            f"主要依据是{deciding['name']}（{deciding['value_text']}{deciding_band}）。"
        )

    concordant = [item for item in criteria if item["status"] == "concordant"]
    if concordant:
        names = "、".join(f"{item['name']} {item['value_text']}" for item in concordant)
        parts.append(f"{names} 的方向也一致。")

    for item in (entry for entry in criteria if entry["status"] == "overridden"):
        parts.append(
            f"{item['name']} {item['value_text']} 提示{item['implies_level_label']}，"
            f"与主要参考不完全一致，先作为交叉验证信息保留。"
        )

    missing = [item["name"] for item in criteria if item["status"] == "missing"]
    if missing:
        parts.append(f"{'、'.join(missing)} 暂缺，未参与本轮分层。")

    parts.append(dose_detail)
    return "".join(parts)


def _indicator_value_text(
    key: str,
    cpet: dict[str, Any],
    tests: dict[str, Any],
    baduanjin_borg: dict[str, Any],
    baduanjin_cpx: dict[str, Any],
) -> str:
    if key == "baduanjin_cpx":
        pct = _cpx_pct(baduanjin_cpx)
        return f"aveVO2pVO2peak {round(pct * 100)}%" if pct is not None else ""
    if key == "cpet":
        parts = []
        if cpet.get("vo2_peak") is not None:
            parts.append(f"VO2peak {_format_num(cpet['vo2_peak'])}")
        if cpet.get("vo2_at") is not None:
            parts.append(f"AT {_format_num(cpet['vo2_at'])}")
        return "、".join(parts) + " mL/kg/min" if parts else ""
    if key == "borg":
        # The indicator name already ends in "Borg"; value_text stays bare to avoid
        # "...八段锦Borg Borg 12" duplication when name + value_text are joined.
        borg = baduanjin_borg.get("borg_avg")
        return f"{_format_num(borg)}" if borg is not None else ""
    if key == "six_mwd":
        # Name is "6MWD"; keep value_text bare to avoid "6MWD 6MWD 620 m".
        six = tests.get("six_mwd")
        return f"{_format_num(six)} m" if six is not None else ""
    return ""


def _indicator_band_text(key: str, level: str) -> str:
    bands = {
        "baduanjin_cpx": {LOW: "aveVO2pVO2peak≥60%", MID: "40–60%", HIGH: "<40%"},
        "six_mwd": {LOW: "<300 m", MID: "300–450 m", HIGH: "≥450 m"},
        "borg": {LOW: "≥14", MID: "11–13", HIGH: "6–10"},
        # CPET pools VO2peak and AT (different cutoffs); the value_text already
        # carries the numbers, so no single band string is shown.
        "cpet": {},
    }
    return bands.get(key, {}).get(level, "")


def _format_num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _level_by_threshold(value: float, low_mid_cut: float, mid_high_cut: float) -> str:
    if value < low_mid_cut:
        return LOW
    if value < mid_high_cut:
        return MID
    return HIGH


def _baduanjin_cpx_level(baduanjin_cpx: dict[str, Any]) -> str | None:
    """Level from prescription-baseline baduanjin CPET intensity (aveVO2pVO2peak).

    Higher %VO2peak during baduanjin means it is harder for the patient -> lower
    capacity: >=60% -> 运动能力低, 40-60% -> 运动能力中,
    <40% -> 运动能力高 (project stratification rule).
    """

    pct = _cpx_pct(baduanjin_cpx)
    if pct is None:
        return None
    if pct >= 0.60:
        return LOW
    if pct >= 0.40:
        return MID
    return HIGH


def _cpx_pct(baduanjin_cpx: dict[str, Any]) -> float | None:
    value = baduanjin_cpx.get("ave_vo2_pct_vo2peak")
    if value is None:
        return None
    return value / 100.0 if value > 1.5 else value


def _baduanjin_borg_fields(profile: dict[str, Any]) -> dict[str, Any]:
    return data_section(profile, "BDJ_BORG")


def _baduanjin_cpx_fields(profile: dict[str, Any]) -> dict[str, Any]:
    return data_section(profile, "BDJ_CPX")


def _borg_level(borg: float) -> str:
    # Project rule: 第一周平均 Borg >=14 -> 运动能力低, 11-13 -> 运动能力中, 6-10 -> 运动能力高.
    if borg >= 14:
        return LOW
    if borg >= 11:
        return MID
    return HIGH


def _downgrade(function_level: str) -> str:
    if function_level == HIGH:
        return MID
    return LOW
