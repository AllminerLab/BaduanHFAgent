"""Access helpers for patient_profile v7.

The runtime contract is intentionally small: tools read patient data from
``profile["data"][<SOURCE_CODE>][<field>]`` only. The helpers below keep the
individual tools mechanical while preserving that contract.
"""

from __future__ import annotations

from typing import Any


def data_section(profile: dict[str, Any], code: str) -> dict[str, Any]:
    data = profile.get("data")
    if not isinstance(data, dict):
        return {}
    section = data.get(code)
    return section if isinstance(section, dict) else {}


def data_value(
    profile: dict[str, Any],
    code: str,
    field: str,
    default: Any = None,
) -> Any:
    return data_section(profile, code).get(field, default)


def data_number(profile: dict[str, Any], code: str, field: str) -> float | None:
    return to_number(data_value(profile, code, field))


def data_bool(profile: dict[str, Any], code: str, field: str) -> bool:
    return is_truthy(data_value(profile, code, field))


def to_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_truthy(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"0", "false", "no", "否", "无", "none", "nan"}
