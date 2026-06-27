"""patient_profile v7 builder.

Transforms a hydrated v6 profile (semantic sections + source_files) into the v7
schema where `data["<CODE>"]["<field>"]` is the single main area tools read from:

    {patient_id, visit, randomization,
     data:        {CODE: {field: value}},          # tools read ONLY here
     raw:         {CODE: [original rows]},          # kept, audit-only, tools never read
     field_index: {standard:{CODE:{field:src_col}}, # redundant annotation (stats/audit)
                   derived:{CODE:{field:{from,method}}}, row:{CODE:[unpromoted cols]}},
     data_quality:{missing_fields, invalid_fields, inconsistencies}}

No hydrate at load time: every field a tool needs is materialised here at build time.
See docs/patient_profile_v7_field_spec.md.
"""

from __future__ import annotations

import json
import os
from typing import Any


_CODE_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "source_file_codes.json")


def _load_code_map() -> dict[str, str]:
    with open(_CODE_MAP_PATH, encoding="utf-8") as handle:
        return json.load(handle)["code_to_file"]


CODE_TO_FILE: dict[str, str] = _load_code_map()
FILE_TO_CODE: dict[str, str] = {v: k for k, v in CODE_TO_FILE.items()}


# ── data[CODE][field] sourced from an already-computed hydrated section ──
# field -> (section, section_field). The value equals what v6 tools read today;
# provenance (real source column) is recorded in field_index.standard below.
SECTION_SOURCED: dict[str, dict[str, tuple[str, str]]] = {
    "CPX": {"vo2_peak": ("cpet", "vo2_peak"), "vo2_at": ("cpet", "vo2_at"),
            "ve_vco2_slope": ("cpet", "ve_vco2_slope"), "peak_hr": ("cpet", "peak_hr"),
            "rest_hr": ("cpet", "rest_hr"), "peak_sbp": ("cpet", "peak_sbp"),
            "hrr": ("cpet", "hrr"), "rer_peak": ("cpet", "rer_peak")},
    "TEST": {"six_mwd": ("tests", "six_mwd"), "mobility_time": ("tests", "mobility_time"),
             "mobility_class": ("tests", "mobility_class"),
             "grip_strength_left": ("tests", "grip_strength_left"),
             "grip_strength_right": ("tests", "grip_strength_right"),
             "assistive_device": ("tests", "assistive_device")},
    "ECHO": {"lvef": ("echo", "lvef"), "lvedd": ("echo", "lvedd")},
    "CHECK": {"sbp": ("check", "sbp"), "dbp": ("check", "dbp"), "rest_hr": ("check", "rest_hr"),
              "edema_left": ("check", "edema_left"), "edema_right": ("check", "edema_right"),
              "nyha": ("history", "nyha"), "height_cm": ("demographics", "height_cm"),
              "weight_kg": ("demographics", "weight_kg"), "bmi": ("demographics", "bmi")},
    "MEDICATION": {"beta_blocker": ("medication", "beta_blocker"),
                   "diuretic": ("medication", "diuretic"),
                   "anticoagulant": ("medication", "anticoagulant")},
    "SYMPTOM": {"dyspnea": ("symptoms", "dyspnea"), "fatigue": ("symptoms", "fatigue")},
    "EQ5D": {"pain": ("symptoms", "pain")},
    "ECG": {"arrhythmia": ("ecg", "arrhythmia")},
    "LABS": {"nt_pro_bnp": ("labs", "nt_pro_bnp")},
    "SEE": {"seesum": ("activity", "see")},
    "IPAQ": {"ipaq": ("activity", "ipaq")},
    "DEMO": {"age": ("demographics", "age"), "sex": ("demographics", "sex")},
}

# ── data[CODE][field] sourced from a source_files[<file>]["fields"] entry (the
# baduanjin zero-week summaries, computed at ingestion and not in a section). ──
SOURCEFILE_FIELD_SOURCED: dict[str, dict[str, str]] = {
    "BDJ_BORG": {"borg_avg": "borg_avg"},
    "BDJ_CPX": {"ave_vo2_pct_vo2peak": "ave_vo2_pct_vo2peak"},
}

# ── data[CODE][field] sourced directly from raw rows (currently scanned lazily by
# Tool 3/5). alias list -> first matching raw column value, normalised to number/bool. ──
RAW_NUMERIC_SOURCED: dict[str, dict[str, list[str]]] = {
    "HIST": {"cad": ["histcad"], "af": ["histara9"], "av_block_code": ["histarb3"],
             "stroke": ["histcvd6"], "pad": ["histpad8"], "copd": ["histcpd"]},
    "ECHO": {"valve_ar": ["ecoar"], "valve_mr": ["ecomr"], "valve_as": ["ecoas"], "valve_ms": ["ecoms"]},
    "MSK": {"cervical": ["颈椎病史（1/0）", "cervical_spine_history", "neck_history"],
            "shoulder": ["肩关节病史（1/0）", "shoulder_history", "shoulder_joint_history"],
            "lumbar": ["腰椎病史（1/0）", "lumbar_spine_history", "low_back_history"],
            "knee": ["膝关节手术史（1/0）", "膝关节病史（1/0）", "knee_surgery_history", "knee_history"],
            "learned_baduanjin": ["既往是否学过八段锦", "learned_baduanjin", "baduanjin_learned"]},
    "PACE": {"device": ["pmyn"], "device_type": ["pmtype"]},
    "SEE": {"seesum": ["seesum", "see", "see_score"]},
}

# field_index.standard provenance: data field -> real source column (for audit).
STANDARD_SOURCE_COL: dict[str, dict[str, str]] = {
    "CPX": {"vo2_peak": "CPXVKPK", "vo2_at": "CPXVKAT", "ve_vco2_slope": "CPXSLOP",
            "peak_hr": "CPXHRPK", "rest_hr": "CPXHRRT", "peak_sbp": "CPXSBPK",
            "rer_peak": "CPXRERPK"},
    "TEST": {"six_mwd": "SMWDT", "mobility_time": "MOBTI", "mobility_class": "MOBCL",
             "grip_strength_left": "STGLF", "grip_strength_right": "STGRT"},
    "ECHO": {"lvef": "ecolvefs2", "lvedd": "ECOLVEDD",
             "valve_ar": "ECOAR", "valve_mr": "ECOMR", "valve_as": "ECOAS", "valve_ms": "ECOMS"},
    "CHECK": {"sbp": "BDSBP", "dbp": "BDDBP", "rest_hr": "BDHTR", "edema_left": "EDEMALT",
              "edema_right": "EDEMART", "nyha": "NYHA", "height_cm": "BDHGT", "weight_kg": "BDWGT"},
    "SYMPTOM": {"dyspnea": "DYSPNEA1", "fatigue": "FATIGUE1"},
    "EQ5D": {"pain": "EQ5D4"},
    "LABS": {"nt_pro_bnp": "BNPNT"},
    "SEE": {"seesum": "SEESUM"},
    "IPAQ": {"ipaq": "IPAQ"},
    "HIST": {"cad": "HISTCAD", "af": "HISTARA9", "av_block_code": "HISTARB3",
             "stroke": "HISTCVD6", "pad": "HISTPAD8", "copd": "HISTCPD"},
    "MSK": {"cervical": "颈椎病史（1/0）", "shoulder": "肩关节病史（1/0）",
            "lumbar": "腰椎病史（1/0）", "knee": "膝关节手术史（1/0）", "learned_baduanjin": "既往是否学过八段锦"},
    "PACE": {"device": "PMYN", "device_type": "PMTYPE"},
    "INFO": {"consdate": "CONSDATE"},
    "DEMO": {"dmbidate": "DMBIDATE", "sex": "DMSEX"},
}

# field_index.derived provenance.
DERIVED_INDEX: dict[str, dict[str, dict[str, Any]]] = {
    "CPX": {"hrr": {"from": ["CPX:peak_hr", "CPX:rest_hr"], "method": "peak_hr - rest_hr（CPXHRR 直读优先）"}},
    "CHECK": {"bmi": {"from": ["CHECK:weight_kg", "CHECK:height_cm"], "method": "weight / (height/100)^2"}},
    "ECG": {"arrhythmia": {"from": ["ECG:ECGRTB", "ECG:ECGLBBB", "ECG:ECGRBBB"], "method": "任一真 → true"}},
    "MEDICATION": {"beta_blocker": {"from": ["MEDICATION:rows.MEHFNAME"], "method": "药名分类"},
                   "diuretic": {"from": ["MEDICATION:rows.MEHFNAME"], "method": "药名分类"},
                   "anticoagulant": {"from": ["MEDICATION:rows.MEHFNAME"], "method": "药名分类"}},
    "DEMO": {"age": {"from": ["INFO:consdate", "DEMO:dmbidate"], "method": "(consdate - dmbidate) / 365.25，四舍五入为岁"}},
    "HIST": {"av_block_high": {"from": ["HIST:av_block_code"], "method": "HISTARB3 == 4"}},
}

SEX_LABELS = {
    "1": "male",
    "2": "female",
    "男": "male",
    "女": "female",
    "m": "male",
    "f": "female",
    "male": "male",
    "female": "female",
}


def _norm(key: str) -> str:
    return str(key).strip().lower()


def _raw_first(rows: list[dict[str, Any]], aliases: list[str]) -> Any:
    wanted = {_norm(a) for a in aliases}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if _norm(key) in wanted and value not in (None, ""):
                return value
    return None


def _source_rows(profile: dict[str, Any], code: str) -> list[dict[str, Any]]:
    filename = CODE_TO_FILE.get(code)
    entry = (profile.get("source_files") or {}).get(filename) if filename else None
    return (entry or {}).get("rows") or [] if isinstance(entry, dict) else []


def _number_equals(value: Any, expected: float) -> bool:
    try:
        return float(value) == expected
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_sex(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return SEX_LABELS.get(text.lower(), text)


def _apply_requested_clinical_features(data: dict[str, dict[str, Any]], profile: dict[str, Any]) -> None:
    """Materialise frequently used baseline descriptors from explicit source columns."""

    info_rows = _source_rows(profile, "INFO")
    demo_rows = _source_rows(profile, "DEMO")
    check_rows = _source_rows(profile, "CHECK")
    echo_rows = _source_rows(profile, "ECHO")
    labs_rows = _source_rows(profile, "LABS")

    consdate = _to_float(_raw_first(info_rows, ["consdate"]))
    dmbidate = _to_float(_raw_first(demo_rows, ["dmbidate"]))
    if consdate is not None:
        data.setdefault("INFO", {})["consdate"] = consdate
    if dmbidate is not None:
        data.setdefault("DEMO", {})["dmbidate"] = dmbidate
    if consdate is not None and dmbidate is not None and consdate > dmbidate:
        age = (consdate - dmbidate) / 365.25
        if 18 <= age <= 120:
            data.setdefault("DEMO", {})["age"] = round(age)

    sex = _decode_sex(_raw_first(demo_rows, ["dmsex"]))
    if sex is not None:
        data.setdefault("DEMO", {})["sex"] = sex

    height_cm = _to_float(_raw_first(check_rows, ["bdhgt"]))
    weight_kg = _to_float(_raw_first(check_rows, ["bdwgt"]))
    if height_cm is not None:
        data.setdefault("CHECK", {})["height_cm"] = height_cm
    if weight_kg is not None:
        data.setdefault("CHECK", {})["weight_kg"] = weight_kg
    if height_cm not in (None, 0) and weight_kg is not None:
        data.setdefault("CHECK", {})["bmi"] = round(weight_kg / (height_cm / 100) ** 2, 1)

    lvef = _to_float(_raw_first(echo_rows, ["ecolvefs2"]))
    if lvef is not None:
        data.setdefault("ECHO", {})["lvef"] = lvef

    nt_pro_bnp = _to_float(_raw_first(labs_rows, ["bnpnt"]))
    if nt_pro_bnp is not None:
        data.setdefault("LABS", {})["nt_pro_bnp"] = nt_pro_bnp


def build_v7_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Build the v7 schema from a hydrated v6 profile (sections + source_files)."""

    data: dict[str, dict[str, Any]] = {}

    # 1) section-sourced standard/derived fields, re-keyed to file short codes
    for code, fields in SECTION_SOURCED.items():
        for field, (section, sfield) in fields.items():
            value = (profile.get(section) or {}).get(sfield)
            if value is not None:
                data.setdefault(code, {})[field] = value

    # 2) lazy fields read straight from raw rows (HIST/ECHO valves/MSK/PACE)
    for code, fields in RAW_NUMERIC_SOURCED.items():
        rows = _source_rows(profile, code)
        for field, aliases in fields.items():
            value = _raw_first(rows, aliases)
            if value is not None:
                data.setdefault(code, {})[field] = value
    hist = data.get("HIST")
    if hist is not None and "av_block_code" in hist:
        hist["av_block_high"] = _number_equals(hist.get("av_block_code"), 4)

    # 2b) summary fields stored in source_files[<file>]["fields"] (baduanjin zero-week)
    for code, fields in SOURCEFILE_FIELD_SOURCED.items():
        filename = CODE_TO_FILE.get(code)
        entry = (profile.get("source_files") or {}).get(filename) if filename else None
        sf_fields = (entry or {}).get("fields") if isinstance(entry, dict) else None
        if isinstance(sf_fields, dict):
            for field, src in fields.items():
                value = sf_fields.get(src)
                if value is not None:
                    data.setdefault(code, {})[field] = value

    # 2c) explicit baseline descriptors requested for processed patient profiles.
    _apply_requested_clinical_features(data, profile)

    # 3) raw region: original rows kept verbatim, keyed by short code
    raw: dict[str, list[dict[str, Any]]] = {}
    for filename, entry in (profile.get("source_files") or {}).items():
        if not isinstance(entry, dict):
            continue
        code = FILE_TO_CODE.get(filename, filename)
        raw[code] = entry.get("rows") or []

    # 4) field_index (redundant annotation; tools never read it)
    field_index = {
        "standard": {c: dict(m) for c, m in STANDARD_SOURCE_COL.items()},
        "derived": {c: {f: dict(v) for f, v in m.items()} for c, m in DERIVED_INDEX.items()},
        "row": {},
    }

    return {
        "patient_id": profile.get("patient_id"),
        "visit": profile.get("visit"),
        "randomization": profile.get("randomization"),
        "data": data,
        "raw": raw,
        "field_index": field_index,
        "data_quality": {
            "missing_fields": profile.get("missing_fields") or [],
            "invalid_fields": profile.get("invalid_fields") or [],
            "inconsistencies": profile.get("inconsistencies") or [],
        },
    }
