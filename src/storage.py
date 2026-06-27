"""Filesystem storage helpers for processed data and interim predictions."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import uuid


PROCESSED_PROFILE_SCHEMA_VERSION = "tool0-v7-file-field-unified"


def save_processed_profile(
    profile: dict[str, Any],
    *,
    processed_dir: str = "data/processed",
    source: str | None = None,
) -> dict[str, str]:
    """Upsert Tool 0 cleaned patient_profile under data/processed."""

    root = Path(processed_dir)
    profile_dir = root / "patient_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    patient_id = _patient_id(profile)
    timestamp = _timestamp()
    profile_path = profile_dir / f"{patient_id}.json"
    payload = {
        "metadata": {
            "patient_id": patient_id,
            "artifact_type": "tool_0_patient_profile",
            "schema_version": PROCESSED_PROFILE_SCHEMA_VERSION,
            "updated_at": timestamp,
            "source": source,
        },
        "patient_profile": _storage_profile(profile),
    }
    _write_json(profile_path, payload)

    index_path = root / "patient_profiles_index.json"
    index = _read_json(index_path, default={"patients": {}})
    index.setdefault("patients", {})[patient_id] = {
        "patient_id": patient_id,
        "profile_path": str(profile_path),
        "schema_version": PROCESSED_PROFILE_SCHEMA_VERSION,
        "updated_at": timestamp,
        "source": source,
    }
    index["updated_at"] = timestamp
    _write_json(index_path, index)

    return {
        "processed_profile": str(profile_path),
        "processed_index": str(index_path),
    }


def _storage_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(profile)


def load_processed_profile(
    patient_id: str,
    *,
    processed_dir: str = "data/processed",
) -> dict[str, Any] | None:
    """Load a cached Tool 0 patient_profile if it already exists."""

    profile_path = _processed_profile_path(patient_id, processed_dir=processed_dir)
    if not profile_path.exists():
        return None
    payload = _read_json(profile_path, default=None)
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("schema_version") != PROCESSED_PROFILE_SCHEMA_VERSION:
        return None
    profile = payload.get("patient_profile")
    return profile if isinstance(profile, dict) else None


def processed_profile_artifacts(
    patient_id: str,
    *,
    processed_dir: str = "data/processed",
) -> dict[str, str]:
    """Return artifact paths for an existing processed profile."""

    root = Path(processed_dir)
    profile_path = _processed_profile_path(patient_id, processed_dir=processed_dir)
    artifacts: dict[str, str] = {}
    if profile_path.exists():
        artifacts["processed_profile"] = str(profile_path)
    index_path = root / "patient_profiles_index.json"
    if index_path.exists():
        artifacts["processed_index"] = str(index_path)
    return artifacts


def save_processing_audit(
    audit: dict[str, Any],
    *,
    processed_dir: str = "data/processed",
) -> dict[str, str]:
    """Save the latest Tool 0 data-boundary audit under data/processed."""

    root = Path(processed_dir)
    audit_path = root / "data_processing_audit.json"
    payload = {
        "metadata": {
            "artifact_type": "tool_0_data_processing_audit",
            "updated_at": _timestamp(),
        },
        "audit": audit,
    }
    _write_json(audit_path, payload)
    return {"processing_audit": str(audit_path)}


def save_prediction_result(
    result: dict[str, Any],
    *,
    interim_dir: str = "data/interim",
) -> dict[str, str]:
    """Save versioned and latest prediction artifacts under data/log."""

    root = _log_root_from_interim_dir(interim_dir)
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    patient_id = _patient_id(result)
    timestamp = _timestamp()
    version_path = prediction_dir / f"{patient_id}_{timestamp}.json"
    latest_path = prediction_dir / f"{patient_id}_latest.json"
    payload = {
        "metadata": {
            "patient_id": patient_id,
            "artifact_type": "prediction_result",
            "created_at": timestamp,
        },
        "prediction_result": result,
    }
    _write_json(version_path, payload)
    _write_json(latest_path, payload)

    index_path = root / "predictions_index.json"
    index = _read_json(index_path, default={"patients": {}})
    entry = index.setdefault("patients", {}).setdefault(
        patient_id,
        {"patient_id": patient_id, "history": []},
    )
    entry["latest_path"] = str(latest_path)
    entry["latest_version_path"] = str(version_path)
    entry["updated_at"] = timestamp
    entry.setdefault("history", []).append(
        {"path": str(version_path), "created_at": timestamp}
    )
    index["updated_at"] = timestamp
    _write_json(index_path, index)

    return {
        "prediction_result": str(version_path),
        "latest_prediction": str(latest_path),
        "prediction_index": str(index_path),
    }


def save_patient_final_prescription(
    final_prescription: dict[str, Any],
    *,
    patient_id: str,
    patient_name: str | None = None,
    interim_dir: str = "data/interim",
    prescription_suffix: str | None = None,
) -> dict[str, str]:
    """Save the final delivered prescription result as P_valid under a prescription output dir."""

    root = _prescription_root(interim_dir, prescription_suffix)
    filename_base = _patient_filename_base(patient_id, patient_name)
    record_path = root / f"{filename_base}_final_prescription.json"
    payload = {
        "metadata": {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "artifact_type": "final_prescription",
            "suffix": _normalize_suffix(prescription_suffix),
            "updated_at": _timestamp(),
        },
        "P_valid": final_prescription,
    }
    _write_json(record_path, payload)
    return {"final_prescription": str(record_path)}


def save_patient_generation_process(
    generation_process: dict[str, Any],
    *,
    patient_id: str,
    patient_name: str | None = None,
    interim_dir: str = "data/interim",
    prescription_suffix: str | None = None,
) -> dict[str, str]:
    """Save Tool 1-4 outputs and every LLM+Skill/Tool 5 attempt under a prescription output dir."""

    root = _prescription_root(interim_dir, prescription_suffix)
    filename_base = _patient_filename_base(patient_id, patient_name)
    record_path = root / f"{filename_base}_generation_process.json"
    payload = {
        "metadata": {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "artifact_type": "generation_process",
            "suffix": _normalize_suffix(prescription_suffix),
            "updated_at": _timestamp(),
        },
        "generation_process": generation_process,
    }
    _write_json(record_path, payload)
    return {"generation_process": str(record_path)}


def load_patient_name_map(
    roster_path: str | None = "data/processed/baduanjin_patient_roster.json",
) -> dict[str, str]:
    """Load an optional RANID002 -> patient name map without filtering patients."""

    if roster_path is None:
        return {}
    path = Path(roster_path)
    if not path.exists():
        return {}
    payload = _read_json(path, default={})
    patients = payload.get("patients", payload) if isinstance(payload, dict) else payload
    if not isinstance(patients, list):
        return {}
    mapping: dict[str, str] = {}
    for item in patients:
        if not isinstance(item, dict):
            continue
        patient_id = item.get("RANID002") or item.get("patient_id")
        name = item.get("name")
        if patient_id in (None, "") or name in (None, ""):
            continue
        text_id = str(patient_id).strip()
        if text_id.endswith(".0"):
            text_id = text_id[:-2]
        mapping[text_id] = str(name).strip()
    return mapping


def _patient_id(data: dict[str, Any]) -> str:
    raw = data.get("patient_id")
    if raw is None and isinstance(data.get("patient_profile"), dict):
        raw = data["patient_profile"].get("patient_id")
    text = str(raw or "unknown").strip()
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "unknown"


def _processed_profile_path(patient_id: str, *, processed_dir: str) -> Path:
    root = Path(processed_dir)
    safe_id = _patient_id({"patient_id": patient_id})
    return root / "patient_profiles" / f"{safe_id}.json"


def _safe_filename(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text or "unknown"


def _patient_filename_base(patient_id: str, patient_name: str | None) -> str:
    safe_id = _patient_id({"patient_id": patient_id})
    safe_name = _safe_filename(patient_name) if patient_name else ""
    return f"{safe_id}_{safe_name}" if safe_name else safe_id


def _prescription_root(interim_dir: str, suffix: str | None) -> Path:
    normalized_suffix = _normalize_suffix(suffix)
    dirname = "prescription"
    if normalized_suffix:
        dirname = f"{dirname}_{normalized_suffix}"
    return Path(interim_dir) / dirname


def _normalize_suffix(value: str | None) -> str:
    if value in (None, ""):
        return ""
    return _safe_filename(str(value))


def _log_root_from_interim_dir(interim_dir: str) -> Path:
    interim_path = Path(interim_dir)
    return interim_path.parent / "log"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)
