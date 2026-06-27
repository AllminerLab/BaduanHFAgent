"""Run a 1-based inclusive slice of the Baduanjin patient roster.

Example:
    python3 src/run_roster_slice.py
    python3 src/run_roster_slice.py --start 11 --end 20
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any


DEFAULT_START = 1
DEFAULT_END = 63
DEFAULT_LLM_CONFIG = "src/config/llm_config.example.json"
DEFAULT_FORCE_REGENERATE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_client import LLMClientError  # noqa: E402
from orchestrator import BaduanjinAgent  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run patients from data/processed/baduanjin_patient_roster.json by "
            "1-based inclusive row range."
        )
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START,
        help=f"1-based start row in the roster, inclusive. Default: {DEFAULT_START}.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=DEFAULT_END,
        help=f"1-based end row in the roster, inclusive. Default: {DEFAULT_END}.",
    )
    parser.add_argument("--roster", default="data/processed/baduanjin_patient_roster.json")
    parser.add_argument("--input", default="data/raw", help="Raw CRF folder or other agent input path.")
    parser.add_argument(
        "--llm-config",
        default=DEFAULT_LLM_CONFIG,
        help=f"LLM config JSON path. Default: {DEFAULT_LLM_CONFIG}.",
    )
    parser.add_argument(
        "--skill",
        default="full",
        choices=["full", "full_skill", "tools_kb", "tools+kb", "tools_only", "tools-only"],
    )
    parser.add_argument("--target-weekly-minutes", type=int, default=None)
    parser.add_argument("--max-regenerations", type=int, default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--summary-dir", default="data/log/roster_slices")
    parser.add_argument("--summary-out", default=None, help="Optional explicit summary JSON path.")
    parser.add_argument("--refresh-tool0", action="store_true", help="Force Tool0 refresh for this slice.")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        default=DEFAULT_FORCE_REGENERATE,
        help=(
            "Regenerate even when final_prescription already exists. "
            f"Default: {DEFAULT_FORCE_REGENERATE}."
        ),
    )
    parser.add_argument("--no-save-artifacts", action="store_true", help="Run without writing patient artifacts.")
    parser.add_argument("--no-intermediate", action="store_true", help="Do not include intermediates in returned output.")
    parser.add_argument("--no-agent-progress", action="store_true", help="Hide detailed agent progress prints.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the slice at the first patient error.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected patients; do not run the agent.")
    args = parser.parse_args(argv)

    roster_path = _resolve_path(args.roster)
    rows = _load_roster(roster_path)
    selected = _select_slice(rows, args.start, args.end)

    print(
        f"[批次] roster={roster_path} total={len(rows)} range={args.start}-{args.end} "
        f"count={len(selected)}",
        flush=True,
    )
    for index, row in selected:
        print(
            f"[批次] #{index}: RANID002={row['patient_id']} ranid={row.get('ranid') or ''} "
            f"name={row.get('name') or ''}",
            flush=True,
        )

    if args.dry_run:
        print("[批次] dry-run 完成：未调用智能体。", flush=True)
        return 0

    agent = BaduanjinAgent(
        llm_config_path=args.llm_config,
        progress=not args.no_agent_progress,
    )
    prescription_root = _prescription_root(args.interim_dir, agent.llm_config.suffix)

    started_at = _timestamp()
    results: list[dict[str, Any]] = []
    for position, row in selected:
        patient_id = row["patient_id"]
        patient_name = row.get("name")
        existing_final_path = _final_prescription_path(
            prescription_root,
            patient_id=patient_id,
            patient_name=patient_name,
        )
        if (
            not args.force_regenerate
            and not args.no_save_artifacts
            and existing_final_path.exists()
        ):
            results.append(
                {
                    "position": position,
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "status": "skipped_existing",
                    "prescription_status": _existing_prescription_status(existing_final_path),
                    "existing_final_prescription": str(existing_final_path),
                    "started_at": None,
                    "finished_at": _timestamp(),
                    "artifacts": {"final_prescription": str(existing_final_path)},
                }
            )
            print(
                f"[批次] 跳过 #{position}/{len(rows)} RANID002={patient_id} "
                f"name={patient_name or ''}: 已存在最终处方 {existing_final_path}",
                flush=True,
            )
            continue

        print(
            f"[批次] 开始 #{position}/{len(rows)} RANID002={patient_id} name={patient_name or ''}",
            flush=True,
        )
        patient_started_at = _timestamp()
        try:
            output = agent.run(
                args.input,
                skill=args.skill,
                patient_id=patient_id,
                target_weekly_minutes=args.target_weekly_minutes,
                max_regenerations=args.max_regenerations,
                include_intermediate=not args.no_intermediate,
                save_artifacts=not args.no_save_artifacts,
                force_tool0=args.refresh_tool0,
                processed_dir=args.processed_dir,
                interim_dir=args.interim_dir,
            )
            status = (output.get("result") or {}).get("status")
            artifacts = output.get("artifacts") or {}
            results.append(
                {
                    "position": position,
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "status": "ok",
                    "prescription_status": status,
                    "started_at": patient_started_at,
                    "finished_at": _timestamp(),
                    "artifacts": artifacts,
                }
            )
            print(
                f"[批次] 完成 #{position} RANID002={patient_id} prescription_status={status}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch progress visible.
            is_llm_error = isinstance(exc, LLMClientError)
            error_record = {
                "position": position,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "status": "error",
                "is_llm_error": is_llm_error,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_at": patient_started_at,
                "finished_at": _timestamp(),
                "traceback": traceback.format_exc(),
            }
            results.append(error_record)
            print(
                f"[批次] 失败 #{position} RANID002={patient_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if is_llm_error:
                summary_path = _write_run_summary(
                    args=args,
                    roster_path=roster_path,
                    selected=selected,
                    results=results,
                    started_at=started_at,
                    exit_reason="llm_error",
                )
                print(
                    f"[批次] LLM 调用失败，已终止后续患者；摘要已保存: {summary_path}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            if args.stop_on_error:
                break

    summary_path = _write_run_summary(
        args=args,
        roster_path=roster_path,
        selected=selected,
        results=results,
        started_at=started_at,
        exit_reason="completed",
    )
    print(f"[批次] 摘要已保存: {summary_path}", flush=True)
    return 1 if any(item["status"] == "error" for item in results) else 0


def _write_run_summary(
    *,
    args: argparse.Namespace,
    roster_path: Path,
    selected: list[tuple[int, dict[str, Any]]],
    results: list[dict[str, Any]],
    started_at: str,
    exit_reason: str,
) -> Path:
    summary = {
        "metadata": {
            "artifact_type": "roster_slice_run_summary",
            "roster": str(roster_path),
            "input": args.input,
            "llm_config": args.llm_config,
            "skill": args.skill,
            "start": args.start,
            "end": args.end,
            "exit_reason": exit_reason,
            "count_requested": len(selected),
            "count_finished": len(results),
            "started_at": started_at,
            "finished_at": _timestamp(),
        },
        "counts": {
            "ok": sum(1 for item in results if item["status"] == "ok"),
            "skipped_existing": sum(1 for item in results if item["status"] == "skipped_existing"),
            "error": sum(1 for item in results if item["status"] == "error"),
        },
        "patients": results,
    }
    summary_path = _summary_path(args, started_at)
    _write_json(summary_path, summary)
    return summary_path


def _load_roster(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("patients", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Roster must be a list or an object with patients[]: {path}")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Roster row #{index} is not an object.")
        patient_id = _normalize_patient_id(
            item.get("RANID002") or item.get("patient_id") or item.get("ranid002")
        )
        if patient_id is None:
            raise ValueError(f"Roster row #{index} has no RANID002/patient_id.")
        normalized.append(
            {
                **item,
                "patient_id": patient_id,
                "name": item.get("name") or item.get("姓名") or item.get("patient_name"),
            }
        )
    return normalized


def _select_slice(rows: list[dict[str, Any]], start: int, end: int) -> list[tuple[int, dict[str, Any]]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    if end < start:
        raise ValueError("--end must be >= --start.")
    if end > len(rows):
        raise ValueError(f"--end={end} exceeds roster length {len(rows)}.")
    return list(enumerate(rows[start - 1 : end], start=start))


def _normalize_patient_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or None


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _summary_path(args: argparse.Namespace, started_at: str) -> Path:
    if args.summary_out:
        return _resolve_path(args.summary_out)
    root = _resolve_path(args.summary_dir)
    return root / f"roster_slice_{args.start:03d}_{args.end:03d}_{started_at}.json"


def _prescription_root(interim_dir: str, suffix: str | None) -> Path:
    normalized_suffix = _safe_filename(suffix) if suffix else ""
    dirname = "prescription"
    if normalized_suffix:
        dirname = f"{dirname}_{normalized_suffix}"
    return _resolve_path(interim_dir) / dirname


def _final_prescription_path(root: Path, *, patient_id: str, patient_name: str | None) -> Path:
    filename_base = _patient_filename_base(patient_id, patient_name)
    return root / f"{filename_base}_final_prescription.json"


def _patient_filename_base(patient_id: str, patient_name: str | None) -> str:
    safe_id = _safe_patient_id(patient_id)
    safe_name = _safe_filename(patient_name) if patient_name else ""
    return f"{safe_id}_{safe_name}" if safe_name else safe_id


def _safe_patient_id(patient_id: str) -> str:
    text = str(patient_id or "unknown").strip()
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "unknown"


def _safe_filename(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text or "unknown"


def _existing_prescription_status(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    p_valid = payload.get("P_valid") if isinstance(payload, dict) else None
    if isinstance(p_valid, dict):
        status = p_valid.get("status")
        return str(status) if status is not None else None
    return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
