"""Compact Tool5 feedback history for LLM regeneration prompts."""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_guardrail_feedback_context(
    attempts: list[dict[str, Any]],
    *,
    max_attempts: int = 5,
    max_rules_per_attempt: int = 12,
) -> dict[str, Any]:
    """Summarize prior failed Tool5 attempts without including prior prescriptions.

    The latest failure is still sent separately as ``guardrail_feedback``. This
    compact history helps the LLM avoid repeating earlier violations while keeping
    the prompt small and focused.
    """

    failed_attempts = []
    rule_counts: Counter[str] = Counter()

    for attempt in attempts:
        validation = (
            attempt.get("tool5_validation")
            or attempt.get("tool_5_validation")
            or {}
        )
        if validation.get("passed"):
            continue
        violations = validation.get("violations") or []
        rules = []
        for violation in violations[:max_rules_per_attempt]:
            rule = str(violation.get("rule") or "unknown_rule")
            rule_counts[rule] += 1
            rules.append(
                {
                    "rule": rule,
                    "severity": violation.get("severity"),
                    "form_id": violation.get("form_id"),
                    "value": _compact_value(violation.get("value")),
                    "limit": _compact_value(violation.get("limit")),
                }
            )
        if rules:
            failed_attempts.append(
                {
                    "call_count": attempt.get("call_count") or _fallback_call_count(attempt),
                    "action": validation.get("action"),
                    "rules": rules,
                }
            )

    repeated = [
        {"rule": rule, "count": count}
        for rule, count in sorted(rule_counts.items())
        if count >= 2
    ]
    return {
        "guardrail_feedback_history": failed_attempts[-max_attempts:],
        "repeated_guardrail_violations": repeated,
    }


def _fallback_call_count(attempt: dict[str, Any]) -> int | None:
    index = attempt.get("attempt_index")
    if isinstance(index, int):
        return index + 1
    return None


def _compact_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:10]]
    if isinstance(value, tuple):
        return [_compact_value(item) for item in value[:10]]
    if isinstance(value, dict):
        return {str(key): _compact_value(val) for key, val in list(value.items())[:10]}
    return str(value)
