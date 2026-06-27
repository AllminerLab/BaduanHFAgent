"""Tool 0: clinical data ingestion and normalization.

This tool standardizes heterogeneous clinical fields into patient_profile JSON.
It deliberately avoids LLM calls. The first supported production input is an
already-standardized JSON/dict; CSV and simple XLSX reading are provided so
module-level exports can be inspected without adding third-party dependencies.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import glob
import json
import os
import re
import zipfile
from xml.etree import ElementTree
from typing import Any

from tools.profile_v7 import build_v7_profile
from profile_access import data_section


PATIENT_ID_ALIASES = ["RANID002", "patient_id", "subject_id", "subjid", "id", "pid"]
VISIT_ALIASES = [
    "VISIT",
    "visit",
    "viscode",
    "visit_id",
    "rvvisit",
    "PMVISIT",
    "pmvisit",
    "csvisit",
    "CHMVISIT",
]
BASELINE_VISITS = {"", "1", "v0", "baseline", "base"}
DEFAULT_ALLOWED_GROUPS: set[str] | None = None
BADUANJIN_ZERO_WEEK_DIR = "0周八段锦数据"
OUTCOME_MODULE_PATTERNS = [
    "25.",
    "AEEV",
    "HOST",
    "HFEX",
    "MACE",
    "DEAD",
    "MECH",
    "codebook",
    ".~",
]
ALLOWED_MODULE_PATTERNS = [
    "DEMO",
    "INFO",
    "MLHF",
    "EQ5D",
    "SYMPTOM",
    "PHQ",
    "GAD",
    "HADS",
    "IPAQ",
    "SEE",
    "CHSYN",
    "CHECK",
    "TEST",
    "ECHO",
    "HOLTER",
    "LABS",
    "ICG",
    "ECG",
    "CPX",
    "HIST",
    "既往疾病",
    "REVA",
    "PACE",
    "CASU",
    "MEDICATION",
]


# Hard plausibility bounds (NOT clinical-decision thresholds). A present value
# outside its range is treated as invalid: recorded in profile["invalid_fields"]
# + audit, and nulled so an implausible value can never drive a tool decision.
VALIDATION_RANGES: dict[str, tuple[float, float]] = {
    "demographics.age": (18, 120),
    "demographics.weight_kg": (20, 250),
    "demographics.height_cm": (100, 220),
    "demographics.bmi": (10, 60),
    "tests.grip_strength_left": (0, 100),
    "tests.grip_strength_right": (0, 100),
    "cpet.vo2_peak": (0, 60),
    "cpet.vo2_at": (0, 50),
    "cpet.ve_vco2_slope": (10, 80),
    "cpet.rer_peak": (0.5, 1.5),
    "cpet.peak_hr": (40, 220),
    "cpet.rest_hr": (30, 150),
    "cpet.peak_sbp": (60, 300),
    "tests.six_mwd": (0, 1000),
    "echo.lvef": (0, 100),
    "history.nyha": (1, 4),
    "check.sbp": (60, 300),
    "check.dbp": (30, 200),
    "check.rest_hr": (30, 150),
    "labs.nt_pro_bnp": (0, 100000),
}


class ClinicalDataIngestion:
    """Normalize raw patient data to the feature view used by Tools 1-4."""

    def __init__(
        self,
        *,
        baseline_visits: set[str] | None = None,
        allowed_groups: set[str] | None = None,
        patient_ids: set[str] | None = None,
    ):
        self.baseline_visits = {_norm_visit(value) for value in (baseline_visits or BASELINE_VISITS)}
        self.allowed_groups = (
            {str(value) for value in allowed_groups}
            if allowed_groups is not None
            else DEFAULT_ALLOWED_GROUPS
        )
        self.selected_patient_ids = _normalize_patient_id_set(patient_ids)
        self.last_audit: dict[str, Any] = {}

    def run(self, source: dict[str, Any] | str) -> dict[str, Any]:
        profiles = self.run_many(source)
        if len(profiles) != 1:
            raise ValueError(
                f"Input contains {len(profiles)} patient profiles. "
                "Use run_many()/BaduanjinAgent.run_batch() for multi-patient input."
            )
        return profiles[0]

    def run_folder(
        self,
        folder: str,
        *,
        patterns: tuple[str, ...] = ("*.xlsx", "*.csv"),
    ) -> list[dict[str, Any]]:
        """Ingest the CRF cohort modules under a raw-data folder.

        The cohort lives in the ``临床CRF数据`` subfolder when present. The
        ``0周八段锦数据`` specialty exports are keyed by three-digit ranid, so they
        are loaded separately and attached after CRF rows are merged by RANID002.
        Each CRF module file's stem becomes its module name (``19.CPX.xlsx`` ->
        ``19.CPX``) and is filtered by the allow/deny lists, so pointing at
        ``data/raw`` just works.
        """

        crf_dir = os.path.join(folder, "临床CRF数据")
        search_dir = crf_dir if os.path.isdir(crf_dir) else folder

        files: dict[str, str] = {}
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(search_dir, pattern))):
                module = os.path.splitext(os.path.basename(path))[0]
                if _module_allowed(module):
                    files[module] = path
        if not files:
            raise ValueError(f"No allowed CRF module files found under: {search_dir}")
        source: dict[str, Any] = {"files": files}
        baduanjin_files = _find_baduanjin_zero_week_files(folder, patterns)
        if baduanjin_files:
            source["baduanjin_zero_week_files"] = baduanjin_files
        return self.run_many(source)

    def run_many(self, source: dict[str, Any] | str) -> list[dict[str, Any]]:
        if isinstance(source, str) and os.path.isdir(source):
            return self.run_folder(source)
        self.last_audit = self._empty_audit()
        raw = self._load_source(source)
        baduanjin_index: dict[str, dict[str, Any]] = {}
        if isinstance(raw, dict):
            baduanjin_index = self._build_baduanjin_zero_week_index(
                raw.pop("__baduanjin_zero_week__", None)
            )
        if self._looks_like_patient_profile(raw):
            self.last_audit["input_mode"] = "patient_profile"
            profiles = [
                raw if isinstance(raw.get("data"), dict) else self._with_missing_flags(raw)
            ]
            profiles = self._filter_patient_profiles(profiles)
            self.last_audit["patient_count"] = len(profiles)
            self._record_randomization_audit(profiles)
            return self._finalize_profiles(profiles)
        if self._looks_like_table_bundle(raw):
            profiles = []
            for record in self._group_records_by_patient(raw):
                profile = self._normalize(record)
                self._attach_baduanjin_zero_week(profile, baduanjin_index)
                profiles.append(self._with_missing_flags(profile))
            profiles = self._filter_patient_profiles(profiles)
            self.last_audit["patient_count"] = len(profiles)
            self._record_randomization_audit(profiles)
            return self._finalize_profiles(profiles)
        if isinstance(raw, list):
            records = self._group_records_by_patient({"input": raw})
            profiles = [
                self._with_missing_flags(self._normalize(record))
                for record in records
            ]
            profiles = self._filter_patient_profiles(profiles)
            self.last_audit["patient_count"] = len(profiles)
            self._record_randomization_audit(profiles)
            return self._finalize_profiles(profiles)
        profile = self._normalize(raw)
        self.last_audit["input_mode"] = "single_record"
        profiles = [self._with_missing_flags(profile)]
        profiles = self._filter_patient_profiles(profiles)
        self.last_audit["patient_count"] = len(profiles)
        self._record_randomization_audit(profiles)
        return self._finalize_profiles(profiles)

    def _group_records_by_patient(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}

        for module, rows in raw.items():
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list):
                continue
            module_name = str(module)
            if not _module_allowed(module_name):
                self._audit_excluded_module(module_name, len(rows), "module_not_allowed_for_initial_prescription")
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                self.last_audit["raw_rows_seen"] += 1
                patient_id = _normalize_patient_id(_first_value(row, PATIENT_ID_ALIASES))
                if patient_id is None:
                    self.last_audit["missing_patient_id_rows"] += 1
                    self._audit_excluded_row(module_name, None, None, "missing_RANID002")
                    continue
                randomization = _parse_ranid002(patient_id)
                if not randomization or not randomization.get("parse_valid"):
                    self._audit_excluded_row(module_name, patient_id, None, "invalid_RANID002_encoding")
                    continue
                group = randomization.get("group")
                if self._group_filter_active() and group not in self.allowed_groups:
                    self._audit_excluded_row(module_name, patient_id, None, "non_baduanjin_group")
                    self.last_audit["excluded_group_counts"][group] = (
                        self.last_audit["excluded_group_counts"].get(group, 0) + 1
                    )
                    continue
                if self.selected_patient_ids is not None and patient_id not in self.selected_patient_ids:
                    self._audit_excluded_row(module_name, patient_id, None, "not_in_requested_patient_ids")
                    self.last_audit["excluded_patient_id_counts"][patient_id] = (
                        self.last_audit["excluded_patient_id_counts"].get(patient_id, 0) + 1
                    )
                    continue
                visit = _clean_str(_first_value(row, VISIT_ALIASES)) or ""
                if not self._is_baseline_visit(visit):
                    self._audit_excluded_row(module_name, patient_id, visit, "non_baseline_visit")
                    continue
                visit = visit or "1"
                key = (patient_id, visit)
                record = grouped.setdefault(
                    key,
                    {
                        "patient_id": patient_id,
                        "visit": visit or None,
                        "modules": {},
                    },
                )
                module_rows = record["modules"].setdefault(str(module), [])
                module_rows.append(row)
                self._audit_included_row(module_name, patient_id, visit)

        self.last_audit["group_count"] = len(grouped)
        return list(grouped.values())

    def _is_baseline_visit(self, visit: Any) -> bool:
        return _norm_visit(visit) in self.baseline_visits

    def _empty_audit(self) -> dict[str, Any]:
        return {
            "input_mode": "table_bundle",
            "patient_id_field": "RANID002",
            "baseline_visits": sorted(self.baseline_visits),
            "raw_rows_seen": 0,
            "included_rows": 0,
            "excluded_rows": 0,
            "missing_patient_id_rows": 0,
            "included_modules": {},
            "excluded_modules": {},
            "excluded_rows_by_module": {},
            "excluded_reasons": {},
            "allowed_groups": sorted(self.allowed_groups) if self._group_filter_active() else None,
            "group_filter_active": self._group_filter_active(),
            "excluded_group_counts": {},
            "selected_patient_ids": sorted(self.selected_patient_ids) if self.selected_patient_ids is not None else None,
            "excluded_patient_id_counts": {},
            "excluded_profile_group_counts": {},
            "excluded_profile_patient_id_counts": {},
            "excluded_profile_ids": [],
            "patients": {},
            "patient_count": 0,
            "group_count": 0,
            "validation": {},
            "randomization": {},
            "baduanjin_zero_week": {
                "source_dir": BADUANJIN_ZERO_WEEK_DIR,
                "files": {},
                "ranid_counts": {},
                "matched_profiles": 0,
                "matched_ranids": [],
                "unmatched_profile_ranids": [],
            },
        }

    def _audit_included_row(self, module: str, patient_id: str, visit: str) -> None:
        self.last_audit["included_rows"] += 1
        module_entry = self.last_audit["included_modules"].setdefault(module, {"rows": 0})
        module_entry["rows"] += 1
        patient_entry = self.last_audit["patients"].setdefault(
            patient_id,
            {"visits": {}, "modules": {}},
        )
        patient_entry["visits"][visit] = patient_entry["visits"].get(visit, 0) + 1
        patient_entry["modules"][module] = patient_entry["modules"].get(module, 0) + 1

    def _audit_excluded_module(self, module: str, rows: int, reason: str) -> None:
        entry = self.last_audit["excluded_modules"].setdefault(
            module,
            {"rows": 0, "reason": reason},
        )
        entry["rows"] += rows
        self.last_audit["raw_rows_seen"] += rows
        self.last_audit["excluded_rows"] += rows
        self.last_audit["excluded_reasons"][reason] = (
            self.last_audit["excluded_reasons"].get(reason, 0) + rows
        )

    def _audit_excluded_row(
        self,
        module: str,
        patient_id: str | None,
        visit: str | None,
        reason: str,
    ) -> None:
        self.last_audit["excluded_rows"] += 1
        self.last_audit["excluded_reasons"][reason] = (
            self.last_audit["excluded_reasons"].get(reason, 0) + 1
        )
        module_entry = self.last_audit["excluded_rows_by_module"].setdefault(
            module,
            {"rows": 0, "reason": "row_level_exclusion"},
        )
        module_entry["rows"] += 1

    def _looks_like_table_bundle(self, raw: Any) -> bool:
        if not isinstance(raw, dict) or self._looks_like_patient_profile(raw):
            return False
        return any(isinstance(value, list) for value in raw.values())

    def _looks_like_patient_profile(self, raw: Any) -> bool:
        return isinstance(raw, dict) and (
            {"cpet", "history", "tests"}.issubset(raw.keys())
            or ("patient_id" in raw and isinstance(raw.get("data"), dict))
        )

    def _finalize_profiles(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for profile in profiles:
            if isinstance(profile.get("data"), dict):
                finalized.append(profile)
            else:
                finalized.append(build_v7_profile(profile))
        return finalized

    def _normalize(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict) and "modules" in raw:
            source_for_fields: Any = raw["modules"]
            patient_id = raw.get("patient_id")
            visit = raw.get("visit")
        else:
            source_for_fields = raw
            patient_id = _clean_str(_first_value(raw, PATIENT_ID_ALIASES))
            visit = _clean_str(_first_value(raw, VISIT_ALIASES))

        patient_id_clean = _clean_str(patient_id)
        source_files = _build_source_files(source_for_fields)
        profile = {
            "patient_id": patient_id_clean,
            "visit": _clean_str(visit),
            "randomization": _parse_ranid002(patient_id_clean),
            "source_files": source_files,
            "demographics": {
                "age": _compute_age(source_for_fields),
                "sex": _decode_sex(source_for_fields),
                "weight_kg": _to_float(
                    _first_value(source_for_fields, ["weight", "weight_kg", "体重", "demowt", "bdwgt"])
                ),
                "height_cm": _to_float(_first_value(source_for_fields, ["height", "height_cm", "身高", "bdhgt"])),
                "bmi": None,  # derived below from weight + height
            },
            "cpet": {
                # VO2 must be mL/kg/min. cpxvkpk/cpxvkat are the per-kg variables;
                # cpxvopk/cpxvoat are L/min and are deliberately NOT matched here so
                # an L/min value can never be silently read as mL/kg/min.
                "vo2_peak": _to_float(
                    _first_value(source_for_fields, ["vo2_peak", "cpxvkpk", "peak_vo2"])
                ),
                "vo2_at": _to_float(
                    _first_value(source_for_fields, ["vo2_at", "cpxvkat", "at_vo2"])
                ),
                "w_peak": _to_float(_first_value(source_for_fields, ["w_peak", "cpxwpk"])),
                "w_at": _to_float(_first_value(source_for_fields, ["w_at", "cpxwat"])),
                "ve_vco2_slope": _to_float(
                    _first_value(source_for_fields, ["ve_vco2_slope", "cpxslop", "vevco2"])
                ),
                "peak_hr": _to_float(_first_value(source_for_fields, ["peak_hr", "cpxhrpk"])),
                "rest_hr": _to_float(
                    _first_value(source_for_fields, ["rest_hr", "cpxhrrt", "bdhtr"])
                ),
                "hrr": _to_float(_first_value(source_for_fields, ["hrr", "heart_rate_recovery"])),
                "rer_peak": _to_float(_first_value(source_for_fields, ["rer_peak", "cpxrerpk"])),
                "peak_sbp": _to_float(_first_value(source_for_fields, ["peak_sbp", "cpxsbpk"])),
            },
            "tests": {
                "six_mwd": _to_float(_first_value(source_for_fields, ["six_mwd", "smwdt", "6mwd"])),
                "mobility_time": _to_float(_first_value(source_for_fields, ["mobti", "mobility_time"])),
                "mobility_class": _to_float(_first_value(source_for_fields, ["mobcl", "mobility_class"])),
                "grip_strength_left": _to_float(_first_value(source_for_fields, ["grip_strength_left", "stglf"])),
                "grip_strength_right": _to_float(_first_value(source_for_fields, ["grip_strength_right", "stgrt"])),
                "assistive_device": _to_bool(
                    _first_value(source_for_fields, ["assistive_device", "dysyn", "dysty"])
                ),
            },
            "echo": {
                "lvef": _to_float(
                    _first_value(source_for_fields, ["lvef", "ecolvefs2", "echo_lvef"])
                ),
                "lvedd": _to_float(_first_value(source_for_fields, ["lvedd", "ecolvidd"])),
                "phenotype": _clean_str(_first_value(source_for_fields, ["phenotype", "hf_type"])),
            },
            "history": {
                "nyha": _to_float(_first_value(source_for_fields, ["nyha", "nyha_class"])),
                "comorbidities": sorted(_collect_comorbidities(source_for_fields)),
                "procedures": sorted(_collect_procedures(source_for_fields)),
                "acute_conditions": sorted(_collect_acute_conditions(source_for_fields)),
            },
            "medication": _collect_medications(source_for_fields),
            "symptoms": {
                "dyspnea": _to_float(_first_value(source_for_fields, ["dyspnea", "dyspnea1"])),
                "fatigue": _to_float(_first_value(source_for_fields, ["fatigue", "fatigue1"])),
                "pain": _to_float(_first_value(source_for_fields, ["pain", "eq5d4", "body_pain"])),
                "dizziness": _to_bool(_first_value(source_for_fields, ["dizziness", "头晕"])),
            },
            "check": {
                "edema_left": _to_float(_first_value(source_for_fields, ["edemalt", "edema_left"])),
                "edema_right": _to_float(_first_value(source_for_fields, ["edemart", "edema_right"])),
                "rest_hr": _to_float(_first_value(source_for_fields, ["bdhtr", "rest_hr"])),
                "sbp": _to_float(_first_value(source_for_fields, ["bdsbp", "sbp"])),
                "dbp": _to_float(_first_value(source_for_fields, ["bddbp", "dbp"])),
            },
            "ecg": {
                "arrhythmia": any(
                    _to_bool(_first_value(source_for_fields, [field]))
                    for field in ["ecgrtb", "ecglbbb", "ecgrbbb", "arrhythmia"]
                )
            },
            "labs": {
                "nt_pro_bnp": _to_float(_first_value(source_for_fields, ["bnpnt", "nt_pro_bnp"]))
            },
            "psychology": {
                "phq9": _to_float(_first_value(source_for_fields, ["phq9", "phq_9"])),
                "gad7": _to_float(_first_value(source_for_fields, ["gad7", "gad_7"])),
                "hads_anxiety": _to_float(_first_value(source_for_fields, ["hads_a", "hadsa"])),
            },
            "activity": {
                "ipaq": _to_float(_first_value(source_for_fields, ["ipaq"])),
                "see": _to_float(_first_value(source_for_fields, ["see", "see_score", "seesum"])),
            },
            "v0_motion": _first_value(source_for_fields, ["v0_motion"]),
            "missing_fields": [],
        }

        if profile["cpet"]["hrr"] is None:
            peak_hr = profile["cpet"]["peak_hr"]
            rest_hr = profile["cpet"]["rest_hr"] or profile["check"]["rest_hr"]
            if peak_hr is not None and rest_hr is not None:
                profile["cpet"]["hrr"] = peak_hr - rest_hr

        weight = profile["demographics"].get("weight_kg")
        height = profile["demographics"].get("height_cm")
        if weight is not None and height not in (None, 0):
            profile["demographics"]["bmi"] = round(weight / (height / 100) ** 2, 1)

        _attach_normalized_fields_to_source_files(profile)
        return profile

    def _load_source(self, source: dict[str, Any] | str) -> Any:
        if isinstance(source, dict):
            if "files" in source and isinstance(source["files"], dict):
                loaded = {
                    module: self._load_source(path)
                    for module, path in source["files"].items()
                }
                baduanjin_files = source.get("baduanjin_zero_week_files")
                if isinstance(baduanjin_files, dict):
                    loaded["__baduanjin_zero_week__"] = {
                        kind: self._load_source(path)
                        for kind, path in baduanjin_files.items()
                    }
                return loaded
            return source

        ext = os.path.splitext(source)[1].lower()
        if ext == ".json":
            with open(source, "r", encoding="utf-8") as handle:
                return json.load(handle)
        if ext == ".csv":
            return self._read_csv(source)
        if ext == ".xlsx":
            return self._read_xlsx_first_sheet(source)
        raise ValueError(f"Unsupported input file type: {source}")

    def _build_baduanjin_zero_week_index(self, raw: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}

        index: dict[str, dict[str, Any]] = {}
        audit = self.last_audit.setdefault("baduanjin_zero_week", {})
        files_audit = audit.setdefault("files", {})
        ranid_counts = audit.setdefault("ranid_counts", {})

        borg_rows = raw.get("borg")
        if isinstance(borg_rows, list):
            borg_by_ranid = _summarize_baduanjin_borg(borg_rows)
            ranid_counts["borg"] = len(borg_by_ranid)
            files_audit["borg"] = {"rows": len(borg_rows), "matched_ranids": len(borg_by_ranid)}
            for ranid, summary in borg_by_ranid.items():
                index.setdefault(ranid, {})["borg"] = summary

        cpet_rows = raw.get("cpet")
        if isinstance(cpet_rows, list):
            cpet_by_ranid = _summarize_baduanjin_cpet(cpet_rows)
            ranid_counts["cpet"] = len(cpet_by_ranid)
            files_audit["cpet"] = {"rows": len(cpet_rows), "matched_ranids": len(cpet_by_ranid)}
            for ranid, summary in cpet_by_ranid.items():
                index.setdefault(ranid, {})["cpet"] = summary

        return index

    def _attach_baduanjin_zero_week(
        self,
        profile: dict[str, Any],
        index: dict[str, dict[str, Any]],
    ) -> None:
        if not index:
            return
        randomization = profile.get("randomization") or _parse_ranid002(profile.get("patient_id"))
        ranid = randomization.get("ranid") if isinstance(randomization, dict) else None
        if ranid is None:
            return
        summary = index.get(ranid)
        audit = self.last_audit.setdefault("baduanjin_zero_week", {})
        if not summary:
            audit.setdefault("unmatched_profile_ranids", []).append(ranid)
            return

        source_files = profile.setdefault("source_files", {})
        if summary.get("borg"):
            borg = summary["borg"]
            source_files[borg["source"]] = {
                "source_file": borg["source"],
                "match_key": {"ranid": ranid},
                "rows": borg.get("rows", []),
                "fields": {
                    "ranid": borg.get("ranid"),
                    "bdjwk": borg.get("bdjwk"),
                    "borg_avg": borg.get("borg_avg"),
                    "borg_daily_scores": borg.get("daily_scores"),
                },
                "derived": {},
            }
        if summary.get("cpet"):
            cpet = summary["cpet"]
            source_files[cpet["source"]] = {
                "source_file": cpet["source"],
                "match_key": {"ranid": ranid},
                "rows": cpet.get("rows", []),
                "fields": {
                    "ranid": cpet.get("ranid"),
                    "hr_rest": cpet.get("hr_rest"),
                    "hr_at": cpet.get("hr_at"),
                    "hr_max": cpet.get("hr_max"),
                    "vo2_rest": cpet.get("vo2_rest"),
                    "vo2_at": cpet.get("vo2_at"),
                    "vo2_peak": cpet.get("vo2_peak"),
                    "ave_hr_pct_hrmax": cpet.get("ave_hr_pct_hrmax"),
                    "ave_vo2_pct_vo2peak": cpet.get("ave_vo2_pct_vo2peak"),
                },
                "derived": {
                    "sample_count": cpet.get("sample_count"),
                },
            }

        audit["matched_profiles"] = audit.get("matched_profiles", 0) + 1
        audit.setdefault("matched_ranids", []).append(ranid)

    def _with_missing_flags(self, profile: dict[str, Any]) -> dict[str, Any]:
        cpet = profile.setdefault("cpet", {})
        check = profile.setdefault("check", {})

        # Validate plausibility first so implausible peak/rest HR can't seed a bogus HRR.
        self._validate_ranges(profile)

        if cpet.get("hrr") is None:
            peak_hr = cpet.get("peak_hr")
            rest_hr = cpet.get("rest_hr") or check.get("rest_hr")
            if peak_hr is not None and rest_hr is not None:
                cpet["hrr"] = peak_hr - rest_hr

        missing = set(profile.get("missing_fields") or [])
        if profile.get("cpet", {}).get("vo2_peak") is None:
            missing.add("cpet.vo2_peak")
        if profile.get("tests", {}).get("six_mwd") is None:
            missing.add("tests.six_mwd")
        if profile.get("history", {}).get("nyha") is None:
            missing.add("history.nyha")
        if profile.get("echo", {}).get("lvef") is None:
            missing.add("echo.lvef")
        profile["missing_fields"] = sorted(missing)
        profile.setdefault("patient_id", None)
        if "randomization" not in profile:
            profile["randomization"] = _parse_ranid002(profile.get("patient_id"))
        return profile

    def _filter_patient_profiles(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for profile in profiles:
            randomization = profile.get("randomization")
            if randomization is None:
                randomization = _parse_ranid002(profile.get("patient_id"))
                profile["randomization"] = randomization
            parse_valid = bool(randomization.get("parse_valid")) if isinstance(randomization, dict) else False
            patient_id = _normalize_patient_id(profile.get("patient_id"))
            if not parse_valid:
                if self.selected_patient_ids is not None and patient_id not in self.selected_patient_ids:
                    self.last_audit["excluded_profile_ids"].append(profile.get("patient_id"))
                    key = patient_id or "invalid"
                    self.last_audit["excluded_profile_patient_id_counts"][key] = (
                        self.last_audit["excluded_profile_patient_id_counts"].get(key, 0) + 1
                    )
                    continue
                kept.append(profile)
                continue
            patient_id = randomization.get("raw")
            group = randomization.get("group")
            if self._group_filter_active() and group not in self.allowed_groups:
                self.last_audit["excluded_profile_group_counts"][group] = (
                    self.last_audit["excluded_profile_group_counts"].get(group, 0) + 1
                )
                self.last_audit["excluded_profile_ids"].append(profile.get("patient_id"))
                continue
            if self.selected_patient_ids is not None and patient_id not in self.selected_patient_ids:
                self.last_audit["excluded_profile_patient_id_counts"][patient_id or "invalid"] = (
                    self.last_audit["excluded_profile_patient_id_counts"].get(patient_id or "invalid", 0) + 1
                )
                self.last_audit["excluded_profile_ids"].append(profile.get("patient_id"))
                continue
            kept.append(profile)
        return kept

    def _group_filter_active(self) -> bool:
        return self.allowed_groups is not None

    def _filter_rule(self) -> str:
        if self.selected_patient_ids is not None:
            return "RANID002"
        if self._group_filter_active():
            return "group"
        return "none"

    def _record_randomization_audit(self, profiles: list[dict[str, Any]]) -> None:
        counts: dict[str, int] = {}
        invalid_ids: list[str] = []
        for profile in profiles:
            randomization = profile.get("randomization") or {}
            group = randomization.get("group")
            if group is not None:
                counts[group] = counts.get(group, 0) + 1
            elif profile.get("patient_id") is not None:
                invalid_ids.append(str(profile.get("patient_id")))
        self.last_audit["randomization"] = {
            "source": "RANID002",
            "encoding": "AAA-BBB-G-CC-PPP",
            "group_position": 7,
            "filter_rule": self._filter_rule(),
            "allowed_groups": sorted(self.allowed_groups) if self._group_filter_active() else None,
            "group_filter_active": self._group_filter_active(),
            "group_counts": counts,
            "excluded_group_counts": self.last_audit.get("excluded_group_counts", {}),
            "excluded_profile_group_counts": self.last_audit.get("excluded_profile_group_counts", {}),
            "selected_patient_id_count": len(self.selected_patient_ids) if self.selected_patient_ids is not None else None,
            "selected_patient_ids": sorted(self.selected_patient_ids) if self.selected_patient_ids is not None else None,
            "excluded_patient_id_counts": self.last_audit.get("excluded_patient_id_counts", {}),
            "excluded_profile_patient_id_counts": self.last_audit.get("excluded_profile_patient_id_counts", {}),
            "invalid_patient_ids": invalid_ids,
        }

    def _validate_ranges(self, profile: dict[str, Any]) -> None:
        """Flag implausible values as invalid (not missing) and null them.

        Out-of-range present values are recorded in ``profile["invalid_fields"]``
        and the audit, then set to None so they cannot drive a tool decision.
        Cross-field inconsistencies are recorded only (ambiguous which side is wrong).
        """

        invalid: list[dict[str, Any]] = []
        for path, (low, high) in VALIDATION_RANGES.items():
            section, key = path.split(".")
            block = profile.get(section)
            if not isinstance(block, dict):
                continue
            value = block.get(key)
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value < low or value > high:
                invalid.append({"field": path, "value": value, "reason": f"out_of_range[{low},{high}]"})
                block[key] = None

        inconsistencies: list[dict[str, Any]] = []
        check = profile.get("check", {})
        sbp, dbp = check.get("sbp"), check.get("dbp")
        if _is_num(sbp) and _is_num(dbp) and sbp <= dbp:
            inconsistencies.append({"fields": ["check.sbp", "check.dbp"], "values": [sbp, dbp], "reason": "sbp_le_dbp"})
        cpet = profile.get("cpet", {})
        phr, rhr = cpet.get("peak_hr"), cpet.get("rest_hr")
        if _is_num(phr) and _is_num(rhr) and phr <= rhr:
            inconsistencies.append({"fields": ["cpet.peak_hr", "cpet.rest_hr"], "values": [phr, rhr], "reason": "peak_hr_le_rest_hr"})

        profile["invalid_fields"] = invalid
        profile["inconsistencies"] = inconsistencies
        if invalid or inconsistencies:
            self.last_audit.setdefault("validation", {})[profile.get("patient_id")] = {
                "invalid_fields": invalid,
                "inconsistencies": inconsistencies,
            }

    def _read_csv(self, path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _read_xlsx_first_sheet(self, path: str) -> list[dict[str, Any]]:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _read_shared_strings(archive)
            sheet_name = _first_sheet_name(archive)
            xml = archive.read(sheet_name)
        rows = _parse_sheet_rows(xml, shared_strings)
        if not rows:
            return []
        headers = [str(item or "").strip() for item in rows[0]]
        records: list[dict[str, Any]] = []
        for row in rows[1:]:
            record = {
                headers[index]: row[index] if index < len(row) else None
                for index in range(len(headers))
                if headers[index]
            }
            if any(value not in (None, "") for value in record.values()):
                records.append(record)
        return records


def _build_source_files(source_for_fields: Any) -> dict[str, dict[str, Any]]:
    """Keep Tool 0 output organized by original raw file/module."""

    source_files: dict[str, dict[str, Any]] = {}
    if isinstance(source_for_fields, dict):
        for module, rows in source_for_fields.items():
            if isinstance(rows, dict):
                normalized_rows = [rows]
            elif isinstance(rows, list):
                normalized_rows = [row for row in rows if isinstance(row, dict)]
            else:
                continue
            if not normalized_rows:
                continue
            source_file = _source_file_name(module)
            source_files[source_file] = {
                "source_file": source_file,
                "rows": normalized_rows,
                "fields": {},
                "derived": {},
            }
    elif isinstance(source_for_fields, list):
        rows = [row for row in source_for_fields if isinstance(row, dict)]
        if rows:
            source_files["input"] = {
                "source_file": "input",
                "rows": rows,
                "fields": {},
                "derived": {},
            }
    return source_files


def _attach_normalized_fields_to_source_files(profile: dict[str, Any]) -> None:
    source_files = profile.setdefault("source_files", {})
    _update_source_fields(
        source_files,
        "21.DEMO.xlsx",
        {
            "sex": profile.get("demographics", {}).get("sex"),
        },
        derived={
            "age": profile.get("demographics", {}).get("age"),
        },
    )
    _update_source_fields(
        source_files,
        "11.CHECK.xlsx",
        {
            "height_cm": profile.get("demographics", {}).get("height_cm"),
            "weight_kg": profile.get("demographics", {}).get("weight_kg"),
            "nyha": profile.get("history", {}).get("nyha"),
            **(profile.get("check") or {}),
        },
        derived={
            "bmi": profile.get("demographics", {}).get("bmi"),
        },
    )
    cpet_fields = dict(profile.get("cpet") or {})
    hrr = cpet_fields.pop("hrr", None)
    _update_source_fields(source_files, "19.CPX.xlsx", cpet_fields, derived={"hrr": hrr})
    _update_source_fields(source_files, "12.TEST.xlsx", profile.get("tests") or {})
    _update_source_fields(source_files, "14.ECHO.xlsx", profile.get("echo") or {})
    _update_source_fields(
        source_files,
        "22.HIST.xlsx",
        {
            "procedures": profile.get("history", {}).get("procedures"),
            "acute_conditions": profile.get("history", {}).get("acute_conditions"),
        },
    )
    _update_source_fields(source_files, "既往疾病.xlsx", {"comorbidities": profile.get("history", {}).get("comorbidities")})
    _update_source_fields(source_files, "24.MEDICATION.xlsx", profile.get("medication") or {})
    _update_source_fields(source_files, "4.SYMPTOM.xlsx", profile.get("symptoms") or {})
    _update_source_fields(source_files, "18.ECG.xlsx", profile.get("ecg") or {})
    _update_source_fields(source_files, "16.LABS.xlsx", profile.get("labs") or {})
    _update_source_fields(source_files, "5.PHQ-9.xlsx", {"phq9": profile.get("psychology", {}).get("phq9")})
    _update_source_fields(source_files, "6.GAD-7.xlsx", {"gad7": profile.get("psychology", {}).get("gad7")})
    _update_source_fields(source_files, "7.HADS.xlsx", {"hads_anxiety": profile.get("psychology", {}).get("hads_anxiety")})
    _update_source_fields(source_files, "8.IPAQ.xlsx", {"ipaq": profile.get("activity", {}).get("ipaq")})
    _update_source_fields(source_files, "9.SEE.xlsx", {"see": profile.get("activity", {}).get("see")})


def _update_source_fields(
    source_files: dict[str, dict[str, Any]],
    source_file: str,
    fields: dict[str, Any],
    *,
    derived: dict[str, Any] | None = None,
) -> None:
    entry = source_files.get(source_file)
    if entry is None:
        return
    entry.setdefault("fields", {}).update(_drop_none(fields))
    if derived:
        entry.setdefault("derived", {}).update(_drop_none(derived))


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _source_file_name(module: Any) -> str:
    text = str(module)
    return text if text.lower().endswith((".xlsx", ".csv", ".json")) else f"{text}.xlsx"


def build_feature_views(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Create auditable views for downstream tools."""

    return {
        "eligibility_view": {
            "CPX": data_section(profile, "CPX"),
            "TEST": data_section(profile, "TEST"),
            "CHECK": data_section(profile, "CHECK"),
            "data_quality": profile.get("data_quality", {}),
        },
        "function_view": {
            "CPX": data_section(profile, "CPX"),
            "TEST": data_section(profile, "TEST"),
            "BDJ_BORG": data_section(profile, "BDJ_BORG"),
            "BDJ_CPX": data_section(profile, "BDJ_CPX"),
            "SEE": data_section(profile, "SEE"),
        },
        "risk_view": {
            "HIST": data_section(profile, "HIST"),
            "MEDICATION": data_section(profile, "MEDICATION"),
            "SYMPTOM": data_section(profile, "SYMPTOM"),
            "EQ5D": data_section(profile, "EQ5D"),
            "ECHO": data_section(profile, "ECHO"),
            "ECG": data_section(profile, "ECG"),
            "LABS": data_section(profile, "LABS"),
            "CHECK": data_section(profile, "CHECK"),
            "PACE": data_section(profile, "PACE"),
        },
        "movement_view": {
            "HIST": data_section(profile, "HIST"),
            "MSK": data_section(profile, "MSK"),
            "SYMPTOM": data_section(profile, "SYMPTOM"),
            "EQ5D": data_section(profile, "EQ5D"),
            "TEST": data_section(profile, "TEST"),
            "CHECK": data_section(profile, "CHECK"),
            "ECG": data_section(profile, "ECG"),
        },
    }


def hydrate_feature_sections(profile: dict[str, Any]) -> dict[str, Any]:
    """Restore runtime feature sections from source_files when loading cache."""

    if profile.get("cpet") and profile.get("tests") and profile.get("history"):
        return profile
    hydrated = deepcopy(profile)
    source_files = hydrated.get("source_files") or {}
    check_fields = _source_file_fields(source_files, "11.CHECK.xlsx")
    demo_fields = _source_file_fields(source_files, "21.DEMO.xlsx")
    demo_derived = _source_file_derived(source_files, "21.DEMO.xlsx")
    check_derived = _source_file_derived(source_files, "11.CHECK.xlsx")

    hydrated["demographics"] = {
        "age": demo_derived.get("age"),
        "sex": demo_fields.get("sex"),
        "height_cm": check_fields.get("height_cm"),
        "weight_kg": check_fields.get("weight_kg"),
        "bmi": check_derived.get("bmi"),
    }
    cpet = dict(_source_file_fields(source_files, "19.CPX.xlsx"))
    cpet_derived = _source_file_derived(source_files, "19.CPX.xlsx")
    if "hrr" in cpet_derived:
        cpet["hrr"] = cpet_derived.get("hrr")
    hydrated["cpet"] = cpet
    hydrated["tests"] = dict(_source_file_fields(source_files, "12.TEST.xlsx"))
    hydrated["echo"] = dict(_source_file_fields(source_files, "14.ECHO.xlsx"))
    hydrated["history"] = {
        "nyha": check_fields.get("nyha"),
        "comorbidities": _source_file_fields(source_files, "既往疾病.xlsx").get("comorbidities", []),
        "procedures": _source_file_fields(source_files, "22.HIST.xlsx").get("procedures", []),
        "acute_conditions": _source_file_fields(source_files, "22.HIST.xlsx").get("acute_conditions", []),
    }
    # Recompute medication from the stored long-format rows (MEHFNAME) rather than the
    # pre-derived flag fields, so cached profiles benefit from the drug-name classifier
    # without a forced re-ingest.
    hydrated["medication"] = _collect_medications(source_files.get("24.MEDICATION.xlsx") or {})
    hydrated["symptoms"] = dict(_source_file_fields(source_files, "4.SYMPTOM.xlsx"))
    hydrated["check"] = {
        key: value
        for key, value in check_fields.items()
        if key not in {"height_cm", "weight_kg", "nyha"}
    }
    hydrated["ecg"] = dict(_source_file_fields(source_files, "18.ECG.xlsx"))
    hydrated["labs"] = dict(_source_file_fields(source_files, "16.LABS.xlsx"))
    hydrated["psychology"] = {
        **_source_file_fields(source_files, "5.PHQ-9.xlsx"),
        **_source_file_fields(source_files, "6.GAD-7.xlsx"),
        **_source_file_fields(source_files, "7.HADS.xlsx"),
    }
    hydrated["activity"] = {
        **_source_file_fields(source_files, "8.IPAQ.xlsx"),
        **_source_file_fields(source_files, "9.SEE.xlsx"),
    }
    return hydrated


def _source_file_fields(source_files: dict[str, Any], source_file: str) -> dict[str, Any]:
    entry = source_files.get(source_file)
    if not isinstance(entry, dict):
        return {}
    fields = entry.get("fields")
    return fields if isinstance(fields, dict) else {}


def _source_file_derived(source_files: dict[str, Any], source_file: str) -> dict[str, Any]:
    entry = source_files.get(source_file)
    if not isinstance(entry, dict):
        return {}
    derived = entry.get("derived")
    return derived if isinstance(derived, dict) else {}


def _find_baduanjin_zero_week_files(
    folder: str,
    patterns: tuple[str, ...],
) -> dict[str, str]:
    data_dir = os.path.join(folder, BADUANJIN_ZERO_WEEK_DIR)
    if not os.path.isdir(data_dir):
        return {}

    files: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(data_dir, pattern))):
            name = os.path.basename(path).lower()
            if name.startswith(".~") or name.startswith("~$"):
                continue
            if "borg" in name:
                files["borg"] = path
            elif "cpet" in name or "cpx" in name:
                files["cpet"] = path
    return files


def _summarize_baduanjin_borg(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ranid = _normalize_ranid(row.get("ranid"))
        borg = _to_float(row.get("borg_avg"))
        if ranid is None or borg is None:
            continue
        candidates.setdefault(ranid, []).append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for ranid, ranid_rows in candidates.items():
        selected = min(
            ranid_rows,
            key=lambda row: (
                _to_float(row.get("bdjwk")) if _to_float(row.get("bdjwk")) is not None else 999,
            ),
        )
        bdjwk = _to_int(_to_float(selected.get("bdjwk")))
        borg_avg = _to_float(selected.get("borg_avg"))
        summaries[ranid] = {
            "source": "0周通用八段锦Borg评分.xlsx",
            "ranid": ranid,
            "rows": ranid_rows,
            "bdjwk": bdjwk,
            "borg_avg": borg_avg,
            "week": bdjwk,
            "borg": borg_avg,
            "daily_scores": [
                _to_float(selected.get(f"borg{index}"))
                for index in range(1, 8)
            ],
        }
    return summaries


def _summarize_baduanjin_cpet(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_ranid: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ranid = _normalize_ranid(row.get("ranid"))
        if ranid is None:
            continue
        by_ranid.setdefault(ranid, []).append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for ranid, ranid_rows in by_ranid.items():
        ave_vo2_pct = _first_numeric(ranid_rows, "aveVO2pVO2peak")
        if ave_vo2_pct is None:
            ave_vo2_pct = _mean_numeric(ranid_rows, "八段锦VO2%VO2peak")
        ave_hr_pct = _first_numeric(ranid_rows, "aveHRpHRmax")
        if ave_hr_pct is None:
            ave_hr_pct = _mean_numeric(ranid_rows, "八段锦HR%HRmax")

        summaries[ranid] = {
            "source": "0周通用八段锦CPET.xlsx",
            "ranid": ranid,
            "rows": ranid_rows,
            "sample_count": len(ranid_rows),
            "hr_rest": _first_numeric(ranid_rows, "HRrest"),
            "hr_at": _first_numeric(ranid_rows, "HRat"),
            "hr_max": _first_numeric(ranid_rows, "HRmax"),
            "vo2_rest": _first_numeric(ranid_rows, "VO2rest"),
            "vo2_at": _first_numeric(ranid_rows, "VO2at"),
            "vo2_peak": _first_numeric(ranid_rows, "VO2peak"),
            "ave_hr_pct_hrmax": ave_hr_pct,
            "ave_vo2_pct_vo2peak": ave_vo2_pct,
        }
    return summaries


def _first_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    for row in rows:
        value = _to_float(row.get(field))
        if value is not None:
            return value
    return None


def _mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        value
        for row in rows
        for value in [_to_float(row.get(field))]
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _to_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(xml)
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for item in root.findall("main:si", ns):
        texts = [node.text or "" for node in item.findall(".//main:t", ns)]
        values.append("".join(texts))
    return values


def _first_sheet_name(archive: zipfile.ZipFile) -> str:
    names = sorted(name for name in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", name))
    if not names:
        raise ValueError("XLSX contains no worksheets.")
    return names[0]


def _parse_sheet_rows(xml: bytes, shared_strings: list[str]) -> list[list[Any]]:
    root = ElementTree.fromstring(xml)
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[Any]] = []
    for row in root.findall(".//main:row", ns):
        values: list[Any] = []
        for cell in row.findall("main:c", ns):
            ref = cell.attrib.get("r", "")
            col_index = _column_index(ref)
            while len(values) < col_index:
                values.append(None)
            values.append(_cell_value(cell, shared_strings, ns))
        rows.append(values)
    return rows


def _cell_value(cell: ElementTree.Element, shared_strings: list[str], ns: dict[str, str]) -> Any:
    value_node = cell.find("main:v", ns)
    if value_node is None or value_node.text is None:
        text_node = cell.find(".//main:t", ns)
        return text_node.text if text_node is not None else None
    raw = value_node.text
    if cell.attrib.get("t") == "s":
        index = int(raw)
        return shared_strings[index] if index < len(shared_strings) else raw
    try:
        as_float = float(raw)
    except ValueError:
        return raw
    if as_float.is_integer():
        return int(as_float)
    return as_float


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    total = 0
    for ch in letters:
        total = total * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(total - 1, 0)


def _first_value(raw: Any, aliases: list[str]) -> Any:
    aliases_norm = {_norm_key(alias) for alias in aliases}
    for key, value in _walk_key_values(raw):
        if _norm_key(key) in aliases_norm and value not in (None, ""):
            return value
    return None


def _walk_key_values(data: Any):
    if isinstance(data, dict):
        for key, value in data.items():
            yield str(key), value
            yield from _walk_key_values(value)
    elif isinstance(data, list):
        for item in data:
            yield from _walk_key_values(item)


def _norm_key(key: str) -> str:
    return re.sub(r"[\s_\-./()（）]+", "", key).lower()


def _clean_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def _normalize_patient_id(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_patient_id_set(values: set[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {_normalize_patient_id(value) for value in values}
    normalized.discard(None)
    return normalized


def _parse_ranid002(value: Any) -> dict[str, Any] | None:
    text = _normalize_patient_id(value)
    if text is None:
        return None
    if not re.fullmatch(r"\d{12}", text):
        return {
            "source": "RANID002",
            "raw": text,
            "parse_valid": False,
        }
    return {
        "source": "RANID002",
        "raw": text,
        "prefix_code": text[:3],
        "ranid": text[3:6],
        "group": text[6],
        "post_group_code": text[7:9],
        "project_id": text[9:12],
        "parse_valid": True,
    }


def _ranid002_group(value: Any) -> str | None:
    parsed = _parse_ranid002(value)
    if not parsed or not parsed.get("parse_valid"):
        return None
    return parsed.get("group")


def _normalize_ranid(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return None
    return text.zfill(3)


def _norm_visit(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _module_allowed(module: str) -> bool:
    upper = module.upper()
    if any(pattern.upper() in upper for pattern in OUTCOME_MODULE_PATTERNS):
        return False
    return any(pattern.upper() in upper for pattern in ALLOWED_MODULE_PATTERNS)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "有"}


SEX_LABELS = {
    "1": "male", "2": "female", "男": "male", "女": "female",
    "m": "male", "f": "female", "male": "male", "female": "female",
}


def _decode_sex(raw: Any) -> str | None:
    value = _first_value(raw, ["sex", "gender", "dmsex", "demosex"])
    if value in (None, ""):
        return None
    return SEX_LABELS.get(str(value).strip().lower(), _clean_str(value))


def _compute_age(raw: Any) -> float | None:
    """Age in years. Uses a direct age field if present, else computes from
    birthdate (DEMO.dmbidate) against a baseline date (consent/CHECK/TEST/CPX),
    both as Excel serials — the shared epoch cancels in the difference."""

    direct = _to_float(_first_value(raw, ["age", "demoage"]))
    if direct is not None:
        return direct
    birth = _to_float(_first_value(raw, ["dmbidate", "birthdate", "birth_date"]))
    if birth is None:
        return None
    reference = None
    for key in ["consdate", "bddate", "dysdate", "cpxdate", "ecodate"]:
        reference = _to_float(_first_value(raw, [key]))
        if reference is not None:
            break
    if reference is None or reference <= birth:
        return None
    age = (reference - birth) / 365.25
    return round(age) if 18 <= age <= 120 else None


def _collect_comorbidities(raw: Any) -> set[str]:
    mappings = {
        "stroke": ["histcvd6", "stroke", "中风", "脑卒中"],
        "pad_lower_limb": ["histpad8", "pad_lower_limb", "下肢动脉闭塞"],
        "copd": ["histcpd", "copd", "慢阻肺"],
        "diabetes": ["diabetes", "histdmt", "糖尿病"],
        "hypertension": ["hypertension", "histhyp", "高血压"],
        "ckd": ["ckd", "histckd", "慢性肾脏病"],
        "atrial_arrhythmia": ["atrial_arrhythmia", "histara", "房性心律失常", "房颤"],
        "coronary_heart_disease": ["coronary_heart_disease", "histcad", "冠心病"],
        "valvular_disease": ["valvular_disease", "histval", "瓣膜病"],
    }
    found = set()
    existing = _first_value(raw, ["comorbidities"])
    if isinstance(existing, list):
        found.update(str(item) for item in existing)
    for name, aliases in mappings.items():
        if _to_bool(_first_value(raw, aliases)):
            found.add(name)
    return found


def _collect_procedures(raw: Any) -> set[str]:
    mappings = {
        "pacemaker": ["pmyn", "pacemaker", "起搏器"],
        "icd": ["icd", "implantable_cardioverter_defibrillator"],
        "crt": ["crt"],
        "cabg": ["cabg", "冠脉搭桥"],
        "pci": ["pci", "stent", "支架"],
    }
    found = set()
    existing = _first_value(raw, ["procedures"])
    if isinstance(existing, list):
        found.update(str(item) for item in existing)
    for name, aliases in mappings.items():
        if _to_bool(_first_value(raw, aliases)):
            found.add(name)
    return found


def _collect_acute_conditions(raw: Any) -> set[str]:
    names = [
        "acute_decompensated_hf",
        "severe_arrhythmia",
        "unstable_angina",
        "acute_myocarditis_pericarditis",
        "severe_aortic_stenosis",
        "acute_pe_dvt",
        "fever_or_systemic_illness",
    ]
    found = set()
    existing = _first_value(raw, ["acute_conditions"])
    if isinstance(existing, list):
        found.update(str(item) for item in existing)
    for name in names:
        if _to_bool(_first_value(raw, [name])):
            found.add(name)
    return found


# --- 用药识别 ---
# CRF 的 24.MEDICATION 是"长表"：每行一种药，药名在 MEHFNAME 的"值"里（如倍他乐克），
# 不是"每个药类一列 1/0"的宽表。因此按药名字符串匹配类别，而不是查列名。
_MEDICATION_NAME_KEYS = [
    "mehfname", "drug_name", "drugname", "medication_name", "medname",
    "药名", "药物名称", "通用名", "商品名",
]

# 各类别中文通用名/品牌关键词（子串匹配）。下游仅作监测推断（β阻滞剂→改 RPE 设靶）
# 与接地卡命中，不改剂量；故偏向高灵敏，宁可多标也不漏标。
_DRUG_CLASS_KEYWORDS: dict[str, list[str]] = {
    # β受体阻滞剂：通用名多以"洛尔"结尾，辅以常见品牌名。
    "beta_blocker": [
        "美托洛尔", "倍他乐克", "比索洛尔", "康忻", "卡维地洛", "络德", "金络",
        "阿替洛尔", "普萘洛尔", "心得安", "阿罗洛尔", "阿尔马尔", "拉贝洛尔",
        "奈必洛尔", "索他洛尔", "艾司洛尔", "洛尔",
    ],
    # 利尿剂（含袢利尿/噻嗪/醛固酮受体拮抗剂 MRA/血管加压素受体拮抗剂）。
    "diuretic": [
        "呋塞米", "速尿", "托拉塞米", "布美他尼", "氢氯噻嗪", "噻嗪", "吲达帕胺",
        "螺内酯", "依普利酮", "托伐普坦", "利尿",
    ],
    # 抗凝药（不含抗血小板：阿司匹林/氯吡格雷/替格瑞洛/吲哚布芬等不计入此类）。
    "anticoagulant": [
        "华法林", "利伐沙班", "拜瑞妥", "达比加群", "泰毕全", "阿哌沙班", "艾乐妥",
        "艾多沙班", "依诺肝素", "那屈肝素", "低分子肝素", "磺达肝癸", "克赛", "肝素",
    ],
}

# 旧"宽表"布尔列的向后兼容别名（每个药类一列 1/0）。
_DRUG_CLASS_FLAG_ALIASES: dict[str, list[str]] = {
    "beta_blocker": ["beta_blocker", "β受体阻滞剂", "metoprolol"],
    "diuretic": ["diuretic", "利尿剂"],
    "anticoagulant": ["anticoagulant", "抗凝"],
}


def _collect_medication_names(raw: Any) -> list[str]:
    """Gather drug-name strings from the long-format medication table (MEHFNAME 值)."""

    name_keys = {_norm_key(key) for key in _MEDICATION_NAME_KEYS}
    names: list[str] = []
    for key, value in _walk_key_values(raw):
        if _norm_key(key) in name_keys and isinstance(value, str) and value.strip():
            names.append(value.strip())
    return names


def _collect_medications(raw: Any) -> dict[str, bool]:
    """Classify HF medications by drug name (long table) with a wide-format fallback.

    The MEDICATION table is long-format (one row per drug), so a class is detected by
    matching the drug-name strings — not by looking up a per-class column. If a legacy
    wide-format file with explicit 1/0 flag columns is supplied, that is used as fallback.
    """

    blob = " ".join(_collect_medication_names(raw))
    result: dict[str, bool] = {}
    for drug_class, keywords in _DRUG_CLASS_KEYWORDS.items():
        hit = any(keyword in blob for keyword in keywords)
        if not hit:
            hit = _to_bool(_first_value(raw, _DRUG_CLASS_FLAG_ALIASES[drug_class]))
        result[drug_class] = hit
    return result
