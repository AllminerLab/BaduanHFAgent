"""Command line interface for the Baduanjin agent."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from orchestrator import BaduanjinAgent
from storage import save_processed_profile, save_processing_audit
from tools.tool_0_data_ingestion import ClinicalDataIngestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Baduanjin HF prescription agent.")
    parser.add_argument("--input", default='data/raw', help="Patient JSON/CSV/XLSX input path or raw CRF folder.")
    parser.add_argument(
        "--llm-config",
        default="src/config/llm_config.qwen.json",
        help="LLM JSON config path.",
    )
    parser.add_argument(
        "--skill",
        default="full",
        choices=["full", "full_skill", "tools_kb", "tools+kb", "tools_only", "tools-only"],
        help="Skill prompt variant.",
    )
    parser.add_argument("--target-weekly-minutes", type=int, default=None)
    parser.add_argument(
        "--max-regenerations",
        type=int,
        default=None,
        help="Override max_regenerations in the LLM config for Tool5-triggered regeneration.",
    )
    parser.add_argument(
        "--patient-id",
        action="append",
        default=[],
        help="Target RANID002. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--patient-ids",
        default=None,
        help="Comma-separated target RANID002 values.",
    )
    parser.add_argument(
        "--patient-ids-file",
        default=None,
        help="JSON file containing RANID002 values, or patients[].RANID002.",
    )
    parser.add_argument("--no-intermediate", action="store_true")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat input as a multi-patient table bundle and process every patient_id/visit group.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory for Tool 0 cleaned patient_profile artifacts.",
    )
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="Directory for prediction/result artifacts.",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not write processed/interim artifacts.",
    )
    parser.add_argument(
        "--refresh-tool0",
        action="store_true",
        help="Ignore cached Tool 0 patient_profile files and re-run cleaning from raw data.",
    )
    parser.add_argument(
        "--tool0-only",
        action="store_true",
        help="Only run Tool 0 and save cleaned patient_profile files under data/processed.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress prints to stderr.",
    )
    args = parser.parse_args(argv)

    patient_ids = _collect_patient_ids(args)
    if args.tool0_only:
        output = _run_tool0_only(args, patient_ids, progress=not args.no_progress)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    agent = BaduanjinAgent(llm_config_path=args.llm_config, progress=not args.no_progress)
    run_method = agent.run_batch if args.batch or (patient_ids is not None and len(patient_ids) > 1) else agent.run
    output: dict[str, Any] = run_method(
        args.input,
        skill=args.skill,
        patient_ids=patient_ids,
        target_weekly_minutes=args.target_weekly_minutes,
        max_regenerations=args.max_regenerations,
        include_intermediate=not args.no_intermediate,
        save_artifacts=not args.no_save_artifacts,
        force_tool0=args.refresh_tool0,
        processed_dir=args.processed_dir,
        interim_dir=args.interim_dir,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _collect_patient_ids(args: argparse.Namespace) -> list[str] | None:
    values: list[str] = []
    for item in args.patient_id or []:
        values.extend(str(item).split(","))
    if args.patient_ids:
        values.extend(str(args.patient_ids).split(","))
    if args.patient_ids_file:
        values.extend(_load_patient_ids_file(args.patient_ids_file))

    patient_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.endswith(".0"):
            text = text[:-2]
        if text in seen:
            continue
        seen.add(text)
        patient_ids.append(text)
    return patient_ids or None


def _run_tool0_only(
    args: argparse.Namespace,
    patient_ids: list[str] | None,
    *,
    progress: bool,
) -> dict[str, Any]:
    _progress(progress, f"Tool0-only 开始: patients={patient_ids or 'all'}")
    ingestor = ClinicalDataIngestion(patient_ids=set(patient_ids) if patient_ids is not None else None)
    profiles = ingestor.run_many(args.input)
    _progress(progress, f"Tool0-only 清洗完成: {len(profiles)} 位患者")

    artifacts: dict[str, Any] = {}
    if not args.no_save_artifacts:
        _progress(progress, "Tool0-only 保存处理审计")
        artifacts.update(save_processing_audit(ingestor.last_audit, processed_dir=args.processed_dir))
        profile_artifacts = {}
        for profile in profiles:
            saved = save_processed_profile(
                profile,
                processed_dir=args.processed_dir,
                source=args.input if isinstance(args.input, str) else None,
            )
            profile_artifacts[str(profile.get("patient_id"))] = saved.get("processed_profile")
            _progress(progress, f"Tool0-only 患者画像已保存: RANID002={profile.get('patient_id')}")
        artifacts["processed_profiles"] = profile_artifacts

    _progress(progress, "Tool0-only 完成")
    return {
        "tool": "tool0",
        "count": len(profiles),
        "patient_ids": [profile.get("patient_id") for profile in profiles],
        "artifacts": artifacts,
    }


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[进度] {message}", file=sys.stderr, flush=True)


def _load_patient_ids_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("patients", payload.get("RANID002", []))
    if isinstance(payload, str):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Patient ID file must be a JSON list or contain patients[]: {path}")
    values: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            value = item.get("RANID002") or item.get("patient_id")
        else:
            value = item
        if value not in (None, ""):
            values.append(str(value))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
