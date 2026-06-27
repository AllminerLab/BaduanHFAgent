"""Top-level Baduanjin agent orchestration."""

from __future__ import annotations

from copy import deepcopy
import sys
from typing import Any

from config import LLMConfig, load_llm_config
from constants import FORM_NAME_BY_ID, STANDARD_CYCLES_BY_ID
from llm_client import OpenAICompatibleLLMClient
from skills import SKILL_BUILDERS
from storage import (
    load_patient_name_map,
    load_processed_profile,
    save_patient_final_prescription,
    save_patient_generation_process,
    processed_profile_artifacts,
    save_prediction_result,
    save_processed_profile,
    save_processing_audit,
)
from tools.tool_0_data_ingestion import (
    ClinicalDataIngestion,
    build_feature_views,
)
from tools.tool_1_eligibility import screen_exercise_eligibility
from tools.tool_2_function_stratification import stratify_function
from tools.tool_3_risk_assessment import assess_risk
from tools.tool_4_action_matching import build_action_limitation_profile
from tools.tool_5_guardrail import validate_prescription
from generation.feasible_region import build_feasible_region
from generation.volume_allocation import allocate_volume_options
from guardrail_feedback import build_guardrail_feedback_context


class BaduanjinAgent:
    """Fixed-order Tool + Skill + LLM + Guardrail pipeline."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        llm_config_path: str | None = None,
        llm_client: Any | None = None,
        patient_name_map_path: str | None = "data/processed/baduanjin_patient_roster.json",
        progress: bool = True,
    ):
        self.llm_config = llm_config or load_llm_config(llm_config_path)
        self.progress = progress
        self.llm_client = llm_client
        if self.llm_client is None and self.llm_config.provider != "mock":
            self.llm_client = OpenAICompatibleLLMClient(
                self.llm_config,
                progress=self._progress if self.progress else None,
            )
        self.patient_name_map = load_patient_name_map(patient_name_map_path)

    def run(
        self,
        source: dict[str, Any] | str,
        *,
        skill: str = "full",
        patient_id: str | None = None,
        patient_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        target_weekly_minutes: int | None = None,
        max_regenerations: int | None = None,
        include_intermediate: bool = True,
        save_artifacts: bool = True,
        force_tool0: bool = False,
        processed_dir: str = "data/processed",
        interim_dir: str = "data/interim",
    ) -> dict[str, Any]:
        requested_ids = _merge_patient_ids(patient_id, patient_ids)
        effective_max_regenerations = self._effective_max_regenerations(max_regenerations)
        self._progress(
            "启动单患者流程: "
            f"patients={_format_patient_ids(requested_ids)}, "
            f"skill={skill}, llm={self.llm_config.provider}/{self.llm_config.model}, "
            f"suffix={self.llm_config.suffix or 'default'}, "
            f"max_regenerations={effective_max_regenerations}"
        )
        if requested_ids is not None and len(requested_ids) != 1:
            raise ValueError("BaduanjinAgent.run() expects exactly one RANID002. Use run_batch() for multiple IDs.")
        prepared_profiles, run_artifact_paths = self._prepare_profiles(
            source,
            patient_ids=requested_ids,
            force_tool0=force_tool0,
            save_artifacts=save_artifacts,
            processed_dir=processed_dir,
        )
        if len(prepared_profiles) != 1:
            raise ValueError(
                f"Input contains {len(prepared_profiles)} patient profiles. "
                "Use run_batch(...)/CLI --batch for multi-patient input, "
                "or provide one --patient-id RANID002."
            )
        profile, profile_artifacts = prepared_profiles[0]
        return self._run_profile(
            profile,
            source=source,
            skill=skill,
            target_weekly_minutes=target_weekly_minutes,
            max_regenerations=effective_max_regenerations,
            include_intermediate=include_intermediate,
            save_artifacts=save_artifacts,
            save_processed_artifact=False,
            processed_dir=processed_dir,
            interim_dir=interim_dir,
            extra_artifacts={**run_artifact_paths, **profile_artifacts},
        )

    def run_batch(
        self,
        source: dict[str, Any] | str,
        *,
        skill: str = "full",
        patient_id: str | None = None,
        patient_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        target_weekly_minutes: int | None = None,
        max_regenerations: int | None = None,
        include_intermediate: bool = True,
        save_artifacts: bool = True,
        force_tool0: bool = False,
        processed_dir: str = "data/processed",
        interim_dir: str = "data/interim",
    ) -> dict[str, Any]:
        merged_ids = _merge_patient_ids(patient_id, patient_ids)
        effective_max_regenerations = self._effective_max_regenerations(max_regenerations)
        self._progress(
            "启动批量流程: "
            f"patients={_format_patient_ids(merged_ids)}, "
            f"skill={skill}, llm={self.llm_config.provider}/{self.llm_config.model}, "
            f"suffix={self.llm_config.suffix or 'default'}, "
            f"max_regenerations={effective_max_regenerations}"
        )
        prepared_profiles, run_artifact_paths = self._prepare_profiles(
            source,
            patient_ids=merged_ids,
            force_tool0=force_tool0,
            save_artifacts=save_artifacts,
            processed_dir=processed_dir,
        )
        results = [
            self._run_profile(
                profile,
                source=source,
                skill=skill,
                target_weekly_minutes=target_weekly_minutes,
                max_regenerations=effective_max_regenerations,
                include_intermediate=include_intermediate,
                save_artifacts=save_artifacts,
                save_processed_artifact=False,
                processed_dir=processed_dir,
                interim_dir=interim_dir,
                extra_artifacts={**run_artifact_paths, **profile_artifacts},
            )
            for profile, profile_artifacts in prepared_profiles
        ]
        return {"count": len(results), "artifacts": run_artifact_paths, "results": results}

    def _effective_max_regenerations(self, value: int | None) -> int:
        if value is None:
            value = self.llm_config.max_regenerations
        return max(0, int(value))

    def _prepare_profiles(
        self,
        source: dict[str, Any] | str,
        *,
        patient_ids: list[str] | None,
        force_tool0: bool,
        save_artifacts: bool,
        processed_dir: str,
    ) -> tuple[list[tuple[dict[str, Any], dict[str, str]]], dict[str, str]]:
        """Load cached Tool 0 profiles first; ingest only missing RANID002s."""

        requested_ids = _normalize_patient_ids(patient_ids)
        self._progress(
            "Tool0 数据准备开始: "
            f"patients={_format_patient_ids(requested_ids)}, "
            f"force_tool0={force_tool0}"
        )
        cached_profiles: dict[str, dict[str, Any]] = {}
        cached_artifacts: dict[str, dict[str, str]] = {}
        if requested_ids is not None and not force_tool0:
            for requested_id in requested_ids:
                self._progress(f"Tool0 检查缓存: RANID002={requested_id}")
                cached = load_processed_profile(requested_id, processed_dir=processed_dir)
                if cached is None:
                    self._progress(f"Tool0 缓存未命中: RANID002={requested_id}")
                    continue
                cached_profiles[requested_id] = cached
                cached_artifacts[requested_id] = processed_profile_artifacts(
                    requested_id,
                    processed_dir=processed_dir,
                )
                self._progress(f"Tool0 缓存命中: RANID002={requested_id}")

        ids_for_tool0: set[str] | None
        if requested_ids is None:
            ids_for_tool0 = None
        else:
            missing_ids = [patient_id for patient_id in requested_ids if patient_id not in cached_profiles]
            ids_for_tool0 = set(missing_ids)

        new_profiles_by_id: dict[str, dict[str, Any]] = {}
        new_artifacts_by_id: dict[str, dict[str, str]] = {}
        run_artifact_paths: dict[str, str] = {}
        if requested_ids is None or ids_for_tool0:
            self._progress(
                "Tool0 开始清洗原始数据: "
                f"patients={_format_patient_ids(sorted(ids_for_tool0) if ids_for_tool0 else None)}"
            )
            ingestor = ClinicalDataIngestion(patient_ids=ids_for_tool0)
            new_profiles = ingestor.run_many(source)
            self._progress(f"Tool0 原始数据清洗完成: {len(new_profiles)} 位患者")
            if save_artifacts:
                run_artifact_paths.update(
                    save_processing_audit(ingestor.last_audit, processed_dir=processed_dir)
                )
                self._progress("Tool0 处理审计已保存")
            for profile in new_profiles:
                profile_id = _normalize_patient_id(profile.get("patient_id"))
                if profile_id is None:
                    continue
                artifacts: dict[str, str] = {}
                if save_artifacts:
                    artifacts.update(
                        save_processed_profile(
                            profile,
                            processed_dir=processed_dir,
                            source=source if isinstance(source, str) else None,
                        )
                    )
                    self._progress(f"Tool0 患者画像已保存: RANID002={profile_id}")
                new_profiles_by_id[profile_id] = profile
                new_artifacts_by_id[profile_id] = artifacts
        else:
            self._progress("Tool0 全部患者均命中缓存，跳过原始数据清洗")

        if requested_ids is None:
            return [
                (profile, new_artifacts_by_id.get(_normalize_patient_id(profile.get("patient_id")) or "", {}))
                for profile in new_profiles_by_id.values()
            ], run_artifact_paths

        unresolved = [
            patient_id
            for patient_id in requested_ids
            if patient_id not in cached_profiles and patient_id not in new_profiles_by_id
        ]
        if unresolved:
            raise ValueError(f"指定的 RANID002 未在已处理缓存或本次原始数据中找到: {', '.join(unresolved)}")

        prepared: list[tuple[dict[str, Any], dict[str, str]]] = []
        for requested_id in requested_ids:
            if requested_id in cached_profiles:
                prepared.append((cached_profiles[requested_id], cached_artifacts.get(requested_id, {})))
            else:
                prepared.append((new_profiles_by_id[requested_id], new_artifacts_by_id.get(requested_id, {})))
        return prepared, run_artifact_paths

    def _run_profile(
        self,
        profile: dict[str, Any],
        *,
        source: dict[str, Any] | str,
        skill: str,
        target_weekly_minutes: int | None,
        max_regenerations: int,
        include_intermediate: bool,
        save_artifacts: bool,
        save_processed_artifact: bool,
        processed_dir: str,
        interim_dir: str,
        extra_artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        patient_label = self._patient_label(profile)
        self._progress(f"{patient_label} 开始处方生成流程")
        artifact_paths: dict[str, str] = dict(extra_artifacts or {})
        if save_artifacts and save_processed_artifact:
            artifact_paths.update(
                save_processed_profile(
                    profile,
                    processed_dir=processed_dir,
                    source=source if isinstance(source, str) else None,
                )
            )
            self._progress(f"{patient_label} Tool0 患者画像已保存")
        self._progress(f"{patient_label} 构建特征视图")
        feature_views = build_feature_views(profile)

        self._progress(f"{patient_label} Tool1 运动资格筛查开始")
        eligibility = screen_exercise_eligibility(profile)
        self._progress(
            f"{patient_label} Tool1 完成: status={eligibility.get('eligibility_status')}, "
            f"risk={eligibility.get('risk_level')}"
        )
        if eligibility["eligibility_status"] == "refuse":
            self._progress(f"{patient_label} Tool1 已拒绝运动处方，跳过 Tool2-Tool5")
            tool_outputs = {
                "tool_1_eligibility": eligibility,
                "tool_2_function_layer": None,
                "tool_3_risk_constraints": None,
                "tool_4_action_limitation_profile": None,
                "tool_5_guardrail": None,
            }
            output = self._final_output(
                {
                    "status": "refused",
                    "rationale": eligibility.get("rationale", ""),
                    "gate": "Tool 1 运动资格筛查",
                },
                profile,
                {
                    "feature_views": feature_views,
                    "eligibility": eligibility,
                    "p_raw": None,
                    "p_valid": None,
                },
                include_intermediate,
                artifact_paths,
            )
            if save_artifacts:
                self._progress(f"{patient_label} 保存完整运行日志")
                artifact_paths.update(save_prediction_result(output, interim_dir=interim_dir))
                self._progress(f"{patient_label} 保存最终处方和生成过程")
                artifact_paths.update(
                    self._save_patient_prescription_artifacts(
                        profile,
                        tool_outputs=tool_outputs,
                        final_prescription=output["result"],
                        p_raw=None,
                        p_valid=None,
                        prescription_attempts=[],
                        skill=skill,
                        max_regenerations=max_regenerations,
                        artifact_paths=artifact_paths,
                        interim_dir=interim_dir,
                    )
                )
                output["artifacts"] = artifact_paths
                self._progress(f"{patient_label} 文件保存完成")
            return output

        self._progress(f"{patient_label} Tool2 运动能力分层开始")
        function_layer = stratify_function(profile, eligibility)
        self._progress(
            f"{patient_label} Tool2 完成: level={function_layer.get('candidate_level')}, "
            f"resolved_by={function_layer.get('resolved_by')}"
        )
        self._progress(f"{patient_label} Tool3 风险评估开始")
        risk_constraints = assess_risk(profile, eligibility)
        self._progress(
            f"{patient_label} Tool3 完成: risk_count={len(risk_constraints.get('risks') or [])}"
        )
        self._progress(f"{patient_label} Tool4 动作限制匹配开始")
        action_profile = build_action_limitation_profile(risk_constraints)
        self._progress(
            f"{patient_label} Tool4 完成: "
            f"hard_constraints={len(action_profile.get('hard_constraints') or [])}, "
            f"soft_preferences={len(action_profile.get('soft_preferences') or [])}"
        )
        self._progress(f"{patient_label} 构建处方可行域")
        feasible_region = build_feasible_region(
            eligibility, function_layer, risk_constraints, action_profile
        )
        self._progress(
            f"{patient_label} 可行域构建完成: forms={len(feasible_region.get('forms') or {})}"
        )

        base_context = {
            "patient_profile": profile,
            "feature_views": feature_views,
            "tool_outputs": {
                "tool_1_eligibility": eligibility,
                "tool_2_function_layer": function_layer,
                "tool_3_risk_constraints": risk_constraints,
                "tool_4_action_limitation_profile": action_profile,
            },
            "feasible_region": feasible_region,
        }

        attempts = 0
        prescription_attempts: list[dict[str, Any]] = []
        selected_form_plan, volume_options, p_raw, validation = self._generate_prescription_attempt(
            base_context,
            profile,
            function_layer,
            eligibility,
            feasible_region,
            target_weekly_minutes,
            skill,
            guardrail_feedback=None,
            attempt_index=attempts,
        )
        prescription_attempts.append(
            _build_prescription_attempt_record(
                attempt_index=attempts,
                guardrail_feedback=None,
                selected_form_plan=selected_form_plan,
                volume_options=volume_options,
                p_raw=p_raw,
                validation=validation,
            )
        )
        while (
            not validation["passed"]
            and validation.get("action") == "regenerate"
            and self.llm_config.provider != "mock"
            and attempts < max_regenerations
        ):
            attempts += 1
            guardrail_feedback = validation.get("regenerate_feedback")
            guardrail_context = build_guardrail_feedback_context(prescription_attempts)
            self._progress(
                f"{patient_label} Tool5 未通过，开始第 {attempts + 1} 次 LLM+Skill 重生成"
            )
            selected_form_plan, volume_options, p_raw, validation = self._generate_prescription_attempt(
                base_context,
                profile,
                function_layer,
                eligibility,
                feasible_region,
                target_weekly_minutes,
                skill,
                guardrail_feedback=guardrail_feedback,
                guardrail_feedback_history=guardrail_context.get("guardrail_feedback_history"),
                repeated_guardrail_violations=guardrail_context.get("repeated_guardrail_violations"),
                attempt_index=attempts,
            )
            prescription_attempts.append(
                _build_prescription_attempt_record(
                    attempt_index=attempts,
                    guardrail_feedback=guardrail_feedback,
                    selected_form_plan=selected_form_plan,
                    volume_options=volume_options,
                    p_raw=p_raw,
                    validation=validation,
                )
            )

        if not validation["passed"]:
            if validation.get("action") == "refuse":
                self._progress(f"{patient_label} Tool5 安全阻断，最终拒绝生成处方")
                prescription = {
                    "status": "refused",
                    "rationale": validation.get("regenerate_feedback") or "Tool 5 安全阻断",
                    "gate": "Tool 5 护栏校验",
                    "violations": validation["violations"],
                }
            else:
                self._progress(f"{patient_label} 重生成达到上限，最终仍未通过 Tool5")
                prescription = {
                    "status": "generation_failure",
                    "rationale": f"重生成 {attempts} 次后仍未通过 Tool 5",
                    "violations": validation["violations"],
                }
            p_valid = None
        else:
            self._progress(f"{patient_label} Tool5 通过，得到最终处方")
            prescription = p_raw
            p_valid = p_raw

        intermediate = {
            "feature_views": feature_views,
            "eligibility": eligibility,
            "function_layer": function_layer,
            "risk_constraints": risk_constraints,
            "action_profile": action_profile,
            "feasible_region": feasible_region,
            "selected_form_plan": selected_form_plan,
            "volume_options": volume_options,
            "p_raw": p_raw,
            "p_valid": p_valid,
            "prescription_attempts": prescription_attempts,
            "tool_5_validation": validation,
        }
        output = self._final_output(
            prescription,
            profile,
            intermediate,
            include_intermediate,
            artifact_paths,
        )
        if save_artifacts:
            self._progress(f"{patient_label} 保存完整运行日志")
            artifact_paths.update(save_prediction_result(output, interim_dir=interim_dir))
            self._progress(f"{patient_label} 保存最终处方和生成过程")
            artifact_paths.update(
                self._save_patient_prescription_artifacts(
                    profile,
                    tool_outputs={
                        "tool_1_eligibility": eligibility,
                        "tool_2_function_layer": function_layer,
                        "tool_3_risk_constraints": risk_constraints,
                        "tool_4_action_limitation_profile": action_profile,
                        "tool_5_guardrail": validation,
                    },
                    final_prescription=prescription,
                    p_raw=p_raw,
                    p_valid=p_valid,
                    prescription_attempts=prescription_attempts,
                    skill=skill,
                    max_regenerations=max_regenerations,
                    artifact_paths=artifact_paths,
                    interim_dir=interim_dir,
                )
            )
            output["artifacts"] = artifact_paths
            self._progress(f"{patient_label} 文件保存完成")
        return output

    def _save_patient_prescription_artifacts(
        self,
        profile: dict[str, Any],
        *,
        tool_outputs: dict[str, Any],
        final_prescription: dict[str, Any],
        p_raw: dict[str, Any] | None,
        p_valid: dict[str, Any] | None,
        prescription_attempts: list[dict[str, Any]],
        skill: str,
        max_regenerations: int,
        artifact_paths: dict[str, str],
        interim_dir: str,
    ) -> dict[str, str]:
        patient_id = str(profile.get("patient_id") or "unknown")
        patient_name = self.patient_name_map.get(patient_id)
        patient_block = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "randomization": profile.get("randomization"),
        }
        tool_1_to_4_outputs = {
            key: tool_outputs.get(key)
            for key in (
                "tool_1_eligibility",
                "tool_2_function_layer",
                "tool_3_risk_constraints",
                "tool_4_action_limitation_profile",
            )
        }
        generation_process = {
            "patient": patient_block,
            "llm": {
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "suffix": self.llm_config.suffix,
            },
            "tool_outputs": tool_1_to_4_outputs,
            "llm_skill_calls": _build_llm_skill_call_records(prescription_attempts, skill=skill),
            "final": {
                "total_llm_skill_calls": len(prescription_attempts),
                "max_regenerations": max_regenerations,
                "P_raw": deepcopy(p_raw),
                "P_valid": deepcopy(final_prescription),
                "tool5_validation": deepcopy(tool_outputs.get("tool_5_guardrail")),
            },
        }
        saved_artifacts: dict[str, str] = {}
        saved_artifacts.update(
            save_patient_final_prescription(
                final_prescription,
                patient_id=patient_id,
                patient_name=patient_name,
                interim_dir=interim_dir,
                prescription_suffix=self.llm_config.suffix,
            )
        )
        saved_artifacts.update(
            save_patient_generation_process(
                generation_process,
                patient_id=patient_id,
                patient_name=patient_name,
                interim_dir=interim_dir,
                prescription_suffix=self.llm_config.suffix,
            )
        )
        return saved_artifacts

    def _generate_prescription_attempt(
        self,
        base_context: dict[str, Any],
        profile: dict[str, Any],
        function_layer: dict[str, Any],
        eligibility: dict[str, Any],
        feasible_region: dict[str, Any],
        target_weekly_minutes: int | None,
        skill: str,
        *,
        guardrail_feedback: str | None,
        guardrail_feedback_history: list[dict[str, Any]] | None = None,
        repeated_guardrail_violations: list[dict[str, Any]] | None = None,
        attempt_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run one full form -> volume -> final generation pass.

        Regeneration restarts from the form stage so that per-form violations
        (cycles/amplitude/tempo/rest outside the feasible region) can be repaired
        by feeding the Tool 5 report back to the LLM, matching the framework's
        "把违规清单喂回 LLM 重新生成" behavior. Volume candidates are re-enumerated
        from the (possibly corrected) form plan before the final synthesis.
        """

        patient_label = self._patient_label(profile)
        call_count = attempt_index + 1
        form_context = dict(base_context)
        if guardrail_feedback:
            form_context["guardrail_feedback"] = guardrail_feedback
        if guardrail_feedback_history:
            form_context["guardrail_feedback_history"] = guardrail_feedback_history
        if repeated_guardrail_violations:
            form_context["repeated_guardrail_violations"] = repeated_guardrail_violations
        self._progress(f"{patient_label} LLM+Skill 第 {call_count} 次: 生成逐式参数")
        selected_form_plan = self._generate_form_plan(form_context, skill)
        selected_form_plan = enrich_form_plan_for_clinicians(selected_form_plan, form_context)

        self._progress(f"{patient_label} 总量候选生成开始")
        volume_options = allocate_volume_options(
            selected_form_plan,
            profile,
            function_layer,
            eligibility,
            target_weekly_minutes,
        )
        self._progress(
            f"{patient_label} 总量候选生成完成: "
            f"candidates={len(volume_options.get('feasible_combinations') or [])}"
        )

        context = dict(base_context)
        context["selected_form_plan"] = selected_form_plan
        context["volume_options"] = volume_options
        if guardrail_feedback:
            context["guardrail_feedback"] = guardrail_feedback
        if guardrail_feedback_history:
            context["guardrail_feedback_history"] = guardrail_feedback_history
        if repeated_guardrail_violations:
            context["repeated_guardrail_violations"] = repeated_guardrail_violations

        self._progress(f"{patient_label} LLM+Skill 第 {call_count} 次: 生成 P_raw")
        p_raw = self._generate_final_prescription(context, skill)
        p_raw = enrich_prescription_for_clinicians(p_raw, context)
        self._progress(f"{patient_label} Tool5 第 {call_count} 次校验开始")
        validation = validate_prescription(p_raw, feasible_region, volume_options, eligibility, profile)
        self._progress(
            f"{patient_label} Tool5 第 {call_count} 次校验完成: "
            f"passed={validation.get('passed')}, action={validation.get('action')}, "
            f"violations={len(validation.get('violations') or [])}"
        )
        return selected_form_plan, volume_options, p_raw, validation

    def _progress(self, message: str) -> None:
        if self.progress:
            print(f"[进度] {message}", file=sys.stderr, flush=True)

    def _patient_label(self, profile: dict[str, Any]) -> str:
        patient_id = str(profile.get("patient_id") or "unknown")
        patient_name = self.patient_name_map.get(patient_id)
        if patient_name:
            return f"患者 {patient_id}({patient_name})"
        return f"患者 {patient_id}"

    def _generate_form_plan(self, context: dict[str, Any], skill: str) -> dict[str, Any]:
        if self.llm_config.provider == "mock":
            return select_default_form_plan(
                context["feasible_region"], _context_soft_preferences(context)
            )

        builder = SKILL_BUILDERS.get(skill)
        if builder is None:
            raise ValueError(f"Unknown skill: {skill}. Available: {sorted(SKILL_BUILDERS)}")
        return normalize_form_plan(self.llm_client.generate_json(builder["form"](context)))

    def _generate_final_prescription(self, context: dict[str, Any], skill: str) -> dict[str, Any]:
        if self.llm_config.provider == "mock":
            return generate_mock_prescription(context)

        builder = SKILL_BUILDERS.get(skill)
        if builder is None:
            raise ValueError(f"Unknown skill: {skill}. Available: {sorted(SKILL_BUILDERS)}")
        return self.llm_client.generate_json(builder["final"](context))

    def _final_output(
        self,
        prescription: dict[str, Any],
        profile: dict[str, Any],
        intermediate: dict[str, Any],
        include_intermediate: bool,
        artifact_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        output = {
            "patient_id": profile.get("patient_id"),
            "result": prescription,
        }
        if "p_raw" in intermediate:
            output["p_raw"] = intermediate["p_raw"]
        if "p_valid" in intermediate:
            output["p_valid"] = intermediate["p_valid"]
        if include_intermediate:
            output["intermediate"] = intermediate
        if artifact_paths:
            output["artifacts"] = artifact_paths
        return output


def select_default_form_plan(
    feasible_region: dict[str, Any],
    soft_preferences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministic form-parameter draft used for volume candidate generation.

    Within the feasible region it picks the highest executable stimulus, but it
    HONORS Tool 4 soft_preferences: where a relative precaution advises a down-titrated
    value (慢速/延长/简化), the mock applies it. This keeps the offline draft clinically
    sensible and lets it satisfy Tool 5's "relative precaution must be considered" check.
    """

    pref_map = _soft_preference_map(soft_preferences)
    forms = []
    for form_id_text in sorted(feasible_region.get("forms", {}), key=lambda value: int(value)):
        feasible = feasible_region["forms"][form_id_text]
        form_id = int(form_id_text)
        forms.append(
            {
                "form_id": form_id,
                "name": feasible.get("name") or FORM_NAME_BY_ID.get(form_id),
                "cycles": max(feasible.get("cycles") or [1]),
                "amplitude": _soft_or_default(pref_map, form_id, "amplitude", feasible, ["标准", "简化", "坐式"]),
                "tempo": _soft_or_default(pref_map, form_id, "tempo", feasible, ["标准", "慢速"]),
                "rest": _soft_or_default(pref_map, form_id, "rest", feasible, ["标准", "延长"]),
                "rationale": "默认选择可行域内的最高可执行训练刺激，并按 soft_preferences 应用相对 precaution 降档。",
            }
        )
    return {
        "status": "generated",
        "global": {"sets_per_session": 1},
        "forms": forms,
        "annotations": [],
    }


def _soft_preference_map(soft_preferences: list[dict[str, Any]] | None) -> dict[tuple[int, str], str]:
    pref_map: dict[tuple[int, str], str] = {}
    for pref in soft_preferences or []:
        forms = pref.get("forms")
        parameter = pref.get("parameter")
        prefer = pref.get("prefer")
        if parameter is None or prefer is None:
            continue
        form_ids = range(1, 9) if forms == "all" else (forms or [])
        for form_id in form_ids:
            pref_map[(int(form_id), parameter)] = prefer
    return pref_map


def _soft_or_default(
    pref_map: dict[tuple[int, str], str],
    form_id: int,
    parameter: str,
    feasible: dict[str, Any],
    default_order: list[str],
) -> str:
    allowed = feasible.get(parameter) or []
    prefer = pref_map.get((form_id, parameter))
    if prefer is not None and prefer in allowed:
        return prefer
    return _prefer(allowed, default_order)


def _context_soft_preferences(context: dict[str, Any]) -> list[dict[str, Any]]:
    action_profile = (context.get("tool_outputs") or {}).get("tool_4_action_limitation_profile") or {}
    return action_profile.get("soft_preferences") or []


def enrich_form_plan_for_clinicians(form_plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Add clinician-facing per-form rationale without changing parameters."""

    enriched = deepcopy(form_plan)
    for item in enriched.get("forms") or []:
        if not isinstance(item, dict):
            continue
        item["rationale"] = _clinician_form_rationale(item, context)
    return enriched


def enrich_prescription_for_clinicians(
    prescription: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Add readable total-volume explanations and sanitize per-form rationale.

    This is intentionally parameter-preserving: cycles/amplitude/tempo/rest and
    global dose values are left untouched so Tool 5 still validates the same plan.
    """

    enriched = deepcopy(prescription)
    if enriched.get("status") != "generated":
        return enriched
    body = enriched.get("prescription")
    if not isinstance(body, dict):
        return enriched

    for item in body.get("forms") or []:
        if isinstance(item, dict):
            item["rationale"] = _clinician_form_rationale(item, context)

    global_plan = body.get("global")
    if isinstance(global_plan, dict):
        global_plan["global_rationale"] = _global_rationale(global_plan, context)
        global_plan["parameter_explanations"] = _global_parameter_explanations(global_plan, context)

    enriched["clinical_summary"] = _clinical_summary(enriched, context)
    return enriched


def _clinician_form_rationale(item: dict[str, Any], context: dict[str, Any]) -> str:
    form_id = int(item.get("form_id") or 0)
    cycles = item.get("cycles")
    standard_cycles = STANDARD_CYCLES_BY_ID.get(form_id, 6)
    parts: list[str] = []

    indicator_text = _function_indicator_text(context)
    action_feature = _form_action_feature(form_id)
    if isinstance(cycles, int) and cycles > standard_cycles:
        parts.append(
            f"循环数选 {cycles} 次，较标准 {standard_cycles} 次上调；"
            f"{indicator_text}，因此将本式纳入上调范围。"
        )
    elif isinstance(cycles, int) and cycles < standard_cycles:
        parts.append(
            f"循环数设为 {cycles} 次，低于标准 {standard_cycles} 次，以降低本式负荷。"
        )
    else:
        parts.append(f"循环数维持标准 {standard_cycles} 次，未因运动能力证据上调本式循环。")

    if action_feature:
        parts.append(f"该式主要动作特点为{action_feature}。")

    choice_notes = _form_choice_notes(item, context)
    if choice_notes:
        parts.extend(choice_notes)
    else:
        parts.append("未见该式明确动作受限，幅度、节奏和休息按标准执行。")
    return "".join(parts)


def _form_choice_notes(item: dict[str, Any], context: dict[str, Any]) -> list[str]:
    form_id = int(item.get("form_id") or 0)
    notes: list[str] = []
    for parameter, label, standard_value in [
        ("amplitude", "幅度", "标准"),
        ("tempo", "节奏", "标准"),
        ("rest", "休息", "标准"),
    ]:
        selected = item.get(parameter)
        reason_groups = _projection_reason_groups(context, form_id, parameter, selected)
        specific_reasons = reason_groups["specific"]
        global_reasons = reason_groups["global"]
        if selected != standard_value:
            if specific_reasons:
                notes.append(f"{label}采用{selected}，{_join_reasons(specific_reasons)}。")
            elif global_reasons:
                notes.append(
                    f"{label}采用{selected}，随全套保守策略调整"
                    f"（{_global_strategy_label(parameter, global_reasons)}）。"
                )
            else:
                notes.append(f"{label}采用{selected}，用于降低该式负荷。")
        elif specific_reasons:
            notes.append(
                f"{label}保持标准；已考虑{_join_reasons(specific_reasons)}，当前仍可标准执行。"
            )
    return notes


def _projection_reason_groups(
    context: dict[str, Any],
    form_id: int,
    parameter: str,
    selected: Any,
) -> dict[str, list[str]]:
    action_profile = (context.get("tool_outputs") or {}).get("tool_4_action_limitation_profile") or {}
    specific: list[str] = []
    global_reasons: list[str] = []
    for constraint in action_profile.get("hard_constraints") or []:
        if constraint.get("parameter") != parameter or not _projection_targets_form(constraint, form_id):
            continue
        reason = _clean_reason(constraint.get("reason"))
        if reason:
            specific.append(reason)
    for preference in action_profile.get("soft_preferences") or []:
        if preference.get("parameter") != parameter or not _projection_targets_form(preference, form_id):
            continue
        if preference.get("prefer") != selected:
            continue
        reason = _clean_reason(preference.get("reason"))
        if not reason:
            continue
        if preference.get("forms") == "all":
            global_reasons.append(reason)
        else:
            specific.append(reason)
    return {
        "specific": _dedupe(specific)[:2],
        "global": _dedupe(global_reasons)[:2],
    }


def _global_strategy_label(parameter: str, reasons: list[str]) -> str:
    if parameter == "tempo":
        return "通气、心律或症状耐受因素提示宜放慢节奏"
    if parameter == "rest":
        return "血流动力学、症状或恢复能力因素提示宜延长休息"
    if parameter == "amplitude":
        return "疼痛、活动能力或依从性因素提示宜降低动作难度"
    return _join_reasons(reasons)


def _form_action_feature(form_id: int) -> str:
    features = {
        1: "上肢上举过头和躯干伸展",
        2: "开弓配合马步，下肢负荷较高",
        3: "单侧上肢上举和躯干伸展",
        4: "转头后瞧，主要涉及颈部旋转",
        5: "马步下蹲、转头摇头和重心转移",
        6: "前屈攀足，涉及腰背和下肢柔韧",
        7: "马步攒拳，下肢负荷和用力感较明显",
        8: "提踵下落和重心控制",
    }
    return features.get(form_id, "")


def _projection_targets_form(projection: dict[str, Any], form_id: int) -> bool:
    forms = projection.get("forms")
    if forms == "all":
        return True
    return form_id in {int(item) for item in (forms or [])}


def _function_indicator_text(context: dict[str, Any]) -> str:
    function_layer = (context.get("tool_outputs") or {}).get("tool_2_function_layer") or {}
    criteria = function_layer.get("criteria") or []
    deciding = next((item for item in criteria if item.get("status") == "deciding"), None)
    if deciding:
        name = _clean_indicator_name(deciding.get("name"))
        value_text = _clean_indicator_value(deciding.get("value_text"))
        if "八段锦CPET强度" in name:
            return _baduanjin_cpx_indicator_text(value_text, function_layer)
        if value_text:
            return f"主要参考{name}（{value_text}）"
        return f"主要参考{name}"
    return "根据当前运动耐量评估"


def _clean_indicator_name(value: Any) -> str:
    text = str(value or "运动耐量指标").strip()
    for prefix in ["①", "②", "③", "④"]:
        text = text.replace(prefix, "")
    return text.strip() or "运动耐量指标"


def _clean_indicator_value(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("aveVO2pVO2peak "):
        pct = text.replace("aveVO2pVO2peak ", "").strip()
        return f"平均摄氧约为峰值摄氧的 {pct}"
    return text


def _baduanjin_cpx_indicator_text(value_text: str, function_layer: dict[str, Any]) -> str:
    pct_text = value_text or "专项强度数据"
    level = function_layer.get("candidate_level")
    if level == "high":
        return f"主要参考处方前通用八段锦CPET（{pct_text}），处方前通用八段锦仅达到低强度，提示当前通用剂量负荷偏低"
    if level == "medium":
        return f"主要参考处方前通用八段锦CPET（{pct_text}），处方前通用八段锦达到中低强度，提示可在安全边界内适度上调"
    if level == "low":
        return f"主要参考处方前通用八段锦CPET（{pct_text}），处方前通用八段锦已达到中等及以上强度，提示不宜继续增加训练量"
    return f"主要参考处方前通用八段锦CPET（{pct_text}）"


def _global_rationale(global_plan: dict[str, Any], context: dict[str, Any]) -> str:
    frequency = global_plan.get("frequency_per_week")
    times_per_day = global_plan.get("times_per_day")
    sets_per_session = global_plan.get("sets_per_session")
    session_minutes = global_plan.get("single_session_minutes")
    weekly_minutes = global_plan.get("weekly_minutes")
    target = (context.get("volume_options") or {}).get("target_weekly_minutes")
    parts = [
        f"本次总量为每次 {sets_per_session} 套、每周 {frequency} 天、每天 {times_per_day} 次，"
        f"估算单次约 {session_minutes} 分钟，周总量约 {weekly_minutes} 分钟。"
    ]
    if target is not None:
        parts.append(_target_comparison_sentence(weekly_minutes, target))
    if _has_risk(context, "cad_or_af"):
        parts.append("患者合并冠心病/房颤相关风险，因此频率控制在每周 3-5 天、单次时长不超过 30 分钟。")
    if _has_risk(context, "low_self_efficacy"):
        parts.append("考虑自我效能偏低，选择相对容易坚持的分次完成方式。")
    return "".join(parts)


def _target_comparison_sentence(weekly_minutes: Any, target: Any) -> str:
    weekly = _as_number(weekly_minutes)
    target_value = _as_number(target)
    if weekly is None or target_value is None:
        return f"本轮目标周总量约 {target} 分钟，当前组合未超过每周 300 分钟上限。"
    delta = round(weekly - target_value, 1)
    if abs(delta) <= 30:
        if delta == 0:
            return f"该组合与本轮目标周总量 {target_value:g} 分钟一致，并未超过每周 300 分钟上限。"
        direction = "高于" if delta > 0 else "低于"
        return (
            f"该组合{direction}本轮目标周总量 {target_value:g} 分钟约 {abs(delta):g} 分钟，"
            "属于接近目标的可执行方案，且未超过每周 300 分钟上限。"
        )
    if delta > 0:
        return (
            f"该组合高于本轮目标周总量 {target_value:g} 分钟约 {delta:g} 分钟，"
            "但仍在每周 300 分钟协议上限内；总量偏高的原因主要来自当前逐式参数估算的单次时长和可选频次组合。"
        )
    return (
        f"该组合低于本轮目标周总量 {target_value:g} 分钟约 {abs(delta):g} 分钟，"
        "属于偏保守的可执行方案，且未超过每周 300 分钟上限。"
    )


def _global_parameter_explanations(global_plan: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    frequency = global_plan.get("frequency_per_week")
    times_per_day = global_plan.get("times_per_day")
    sets_per_session = global_plan.get("sets_per_session")
    session_minutes = global_plan.get("single_session_minutes")
    weekly_minutes = global_plan.get("weekly_minutes")
    selected_level = global_plan.get("selected_volume_level")
    target = (context.get("volume_options") or {}).get("target_weekly_minutes")
    weekly_sessions = None
    if isinstance(frequency, int) and isinstance(times_per_day, int):
        weekly_sessions = frequency * times_per_day

    explanations = {
        "sets_per_session": f"每次练习 {sets_per_session} 套；本轮以单套八式为单次训练单位，避免一次训练负荷过大。",
        "frequency_per_week": f"每周 {frequency} 天；该频率来自可选总量方案，并满足当前安全上限。",
        "times_per_day": f"每天 {times_per_day} 次；通过分次完成来分散单次负荷。",
        "single_session_minutes": (
            f"单次约 {session_minutes} 分钟；由八式循环数、节奏、式间休息和每次套数估算。"
        ),
        "weekly_minutes": (
            f"周总量约 {weekly_minutes} 分钟；按单次时长 × 每周天数 × 每天次数计算"
            f"{f'，共约 {weekly_sessions} 次/周' if weekly_sessions is not None else ''}。"
        ),
        "selected_volume_level": (
            f"选择第 {selected_level} 级总量组合（候选总量中的第 {selected_level} 档）；"
            "在可选方案中兼顾目标周总量、安全上限和可坚持性。"
        ),
    }
    if target is not None:
        weekly = _as_number(weekly_minutes)
        target_value = _as_number(target)
        if weekly is not None and target_value is not None:
            delta = round(weekly - target_value, 1)
            if abs(delta) <= 30:
                explanations["selected_volume_level"] += f"与目标周总量约 {target_value:g} 分钟相差 {abs(delta):g} 分钟。"
            elif delta > 0:
                explanations["selected_volume_level"] += (
                    f"周总量高于目标约 {delta:g} 分钟，但未超过协议上限。"
                )
            else:
                explanations["selected_volume_level"] += (
                    f"周总量低于目标约 {abs(delta):g} 分钟，属于偏保守选择。"
                )
        else:
            explanations["selected_volume_level"] += f" 本轮目标周总量约 {target} 分钟。"
    if _has_risk(context, "cad_or_af"):
        explanations["frequency_per_week"] += "合并冠心病/房颤时，每周 3-5 天更合适。"
        explanations["single_session_minutes"] += "合并冠心病/房颤时，单次训练控制在 30 分钟以内。"
    return explanations


def _clinical_summary(prescription: dict[str, Any], context: dict[str, Any]) -> str:
    eligibility = (context.get("tool_outputs") or {}).get("tool_1_eligibility") or {}
    function_text = _function_indicator_text(context)
    function_layer = (context.get("tool_outputs") or {}).get("tool_2_function_layer") or {}
    level_text = _function_level_text(function_layer)
    risk_text = _risk_summary_text(context)
    baseline_text = _heart_failure_baseline_text(context)
    global_plan = ((prescription.get("prescription") or {}).get("global") or {})
    dose_text = _dose_summary_text(global_plan)
    prefix = "本处方已通过运动资格筛查"
    if eligibility.get("data_incomplete"):
        prefix += "，但关键数据不完整，按保守策略处理"
    summary_parts = [
        f"处方摘要：{prefix}。",
    ]
    if baseline_text:
        summary_parts.append(f"心衰基本情况：{baseline_text}。")
    summary_parts.append(f"运动能力按{level_text}处理，{function_text}。")
    if risk_text:
        summary_parts.append(f"主要安全限制包括：{risk_text}。")
    summary_parts.append(f"总体策略为{dose_text}，逐式参数按动作受限和耐受情况调整。")
    return "".join(summary_parts)


def _function_level_text(function_layer: dict[str, Any]) -> str:
    labels = {"low": "运动能力低", "medium": "运动能力中", "high": "运动能力高"}
    return labels.get(function_layer.get("candidate_level"), "当前运动耐量")


def _risk_summary_text(context: dict[str, Any]) -> str:
    risks = ((context.get("tool_outputs") or {}).get("tool_3_risk_constraints") or {}).get("risks") or []
    priority = [
        "severe_valvular",
        "device_implant",
        "cad_or_af",
        "arrhythmia",
        "low_lvef",
        "high_bnp",
        "high_ve_vco2",
        "dyspnea_high",
        "low_hrr",
        "high_peak_sbp",
        "lower_limb_edema",
        "pad_lower_limb",
        "poor_balance",
        "low_self_efficacy",
        "baduanjin_newcomer",
    ]
    risk_by_id = {risk.get("risk_id"): risk for risk in risks if risk.get("risk_id")}
    ordered = [risk_by_id[key] for key in priority if key in risk_by_id]
    ordered.extend(risk for risk in risks if risk not in ordered)
    items = [
        _clean_reason(risk.get("detail") or risk.get("value_text"))
        for risk in ordered
    ]
    return "；".join(_dedupe([str(item) for item in items if item])[:5])


def _heart_failure_baseline_text(context: dict[str, Any]) -> str:
    profile = context.get("patient_profile") or {}
    data = profile.get("data") if isinstance(profile, dict) else {}
    if not isinstance(data, dict):
        return ""

    demo = data.get("DEMO") if isinstance(data.get("DEMO"), dict) else {}
    check = data.get("CHECK") if isinstance(data.get("CHECK"), dict) else {}
    echo = data.get("ECHO") if isinstance(data.get("ECHO"), dict) else {}
    labs = data.get("LABS") if isinstance(data.get("LABS"), dict) else {}

    parts: list[str] = []
    age = _as_number(demo.get("age"))
    if age is not None:
        parts.append(f"{age:g}岁")

    sex = _sex_text(demo.get("sex"))
    if sex:
        parts.append(sex)

    bmi = _as_number(check.get("bmi"))
    if bmi is not None:
        parts.append(f"BMI {bmi:g} kg/m²")

    lvef = _as_number(echo.get("lvef"))
    if lvef is not None:
        parts.append(f"LVEF {lvef:g}%")

    nt_pro_bnp = _as_number(labs.get("nt_pro_bnp"))
    if nt_pro_bnp is not None:
        parts.append(f"NT-proBNP {nt_pro_bnp:g} pg/mL")

    return "，".join(parts)


def _sex_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"male", "m", "1", "男", "男性"}:
        return "男性"
    if text in {"female", "f", "2", "女", "女性"}:
        return "女性"
    return str(value).strip() if value not in (None, "") else ""


def _dose_summary_text(global_plan: dict[str, Any]) -> str:
    frequency = global_plan.get("frequency_per_week")
    times_per_day = global_plan.get("times_per_day")
    session_minutes = global_plan.get("single_session_minutes")
    weekly_minutes = global_plan.get("weekly_minutes")
    return (
        f"每周 {frequency} 天、每天 {times_per_day} 次、单次约 {session_minutes} 分钟、"
        f"周总量约 {weekly_minutes} 分钟"
    )


def _as_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_risk(context: dict[str, Any], risk_id: str) -> bool:
    risks = ((context.get("tool_outputs") or {}).get("tool_3_risk_constraints") or {}).get("risks") or []
    return any(risk.get("risk_id") == risk_id for risk in risks)


def _clean_reason(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("Tool 4", "").replace("soft_preferences", "").replace("precaution", "注意事项")
    return text.rstrip("。；;")


def _join_reasons(reasons: list[str]) -> str:
    return "；".join(reason for reason in reasons if reason)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_form_plan(form_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM form-stage output to the volume-allocation input shape."""

    if "prescription" in form_plan and isinstance(form_plan["prescription"], dict):
        forms = form_plan["prescription"].get("forms", [])
    else:
        forms = form_plan.get("forms", [])
    return {
        "status": form_plan.get("status", "generated"),
        "global": {"sets_per_session": 1},
        "forms": forms,
        "annotations": form_plan.get("annotations", []),
    }


def generate_mock_prescription(context: dict[str, Any]) -> dict[str, Any]:
    """Local deterministic generator for tests and offline development."""

    feasible_region = context["feasible_region"]
    volume_options = context["volume_options"]
    plan = deepcopy(
        context.get("selected_form_plan")
        or select_default_form_plan(feasible_region, _context_soft_preferences(context))
    )
    selected_volume = deepcopy(volume_options["feasible_combinations"][0])
    global_plan = {
        "sets_per_session": selected_volume["sets_per_session"],
        "frequency_per_week": selected_volume["frequency_per_week"],
        "times_per_day": selected_volume["times_per_day"],
        "single_session_minutes": volume_options["single_session_minutes"],
        "weekly_minutes": selected_volume["weekly_minutes"],
        "selected_volume_level": selected_volume["level"],
    }
    prescription_body = {
        "global": global_plan,
        "forms": plan.get("forms", []),
    }

    annotations = deepcopy(feasible_region.get("annotations") or [])
    annotations.extend(deepcopy(plan.get("annotations") or []))
    eligibility = context.get("tool_outputs", {}).get("tool_1_eligibility", {})
    confidence = "high"
    if annotations:
        confidence = "medium"
    if eligibility.get("data_incomplete"):
        confidence = "low"

    result = {
        "status": "generated",
        "confidence": confidence,
        "prescription": prescription_body,
        "annotations": annotations,
    }
    return enrich_prescription_for_clinicians(result, context)


def _merge_patient_ids(
    patient_id: str | None,
    patient_ids: list[str] | set[str] | tuple[str, ...] | None,
) -> list[str] | None:
    merged: list[str] = []
    if patient_id:
        merged.append(patient_id)
    if patient_ids:
        merged.extend(str(item) for item in patient_ids)
    return _normalize_patient_ids(merged) if merged else None


def _normalize_patient_ids(values: list[str] | set[str] | tuple[str, ...] | None) -> list[str] | None:
    if values is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            patient_id = _normalize_patient_id(item)
            if patient_id is None or patient_id in seen:
                continue
            seen.add(patient_id)
            normalized.append(patient_id)
    return normalized


def _normalize_patient_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _format_patient_ids(values: list[str] | set[str] | tuple[str, ...] | None) -> str:
    if values is None:
        return "all"
    items = [str(item) for item in values]
    if not items:
        return "none"
    if len(items) <= 5:
        return ",".join(items)
    return f"{','.join(items[:5])}...(+{len(items) - 5})"


def _build_prescription_attempt_record(
    *,
    attempt_index: int,
    guardrail_feedback: str | None,
    selected_form_plan: dict[str, Any],
    volume_options: dict[str, Any],
    p_raw: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    passed = bool(validation.get("passed"))
    return {
        "call_count": attempt_index + 1,
        "attempt_index": attempt_index,
        "is_regeneration": attempt_index > 0,
        "guardrail_feedback": guardrail_feedback,
        "selected_form_plan": deepcopy(selected_form_plan),
        "volume_options": deepcopy(volume_options),
        "P_raw": deepcopy(p_raw),
        "P_valid": deepcopy(p_raw) if passed else None,
        "prescription": deepcopy(p_raw),
        "tool_5_validation": deepcopy(validation),
        "tool5_validation": deepcopy(validation),
    }


def _build_llm_skill_call_records(
    prescription_attempts: list[dict[str, Any]],
    *,
    skill: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fallback_index, attempt in enumerate(prescription_attempts, start=1):
        records.append(
            {
                "call_count": attempt.get("call_count", fallback_index),
                "skill": skill,
                "is_regeneration": bool(attempt.get("is_regeneration")),
                "guardrail_feedback": attempt.get("guardrail_feedback"),
                "selected_form_plan": deepcopy(attempt.get("selected_form_plan")),
                "volume_options": deepcopy(attempt.get("volume_options")),
                "P_raw": deepcopy(attempt.get("P_raw", attempt.get("prescription"))),
                "P_valid": deepcopy(attempt.get("P_valid")),
                "tool5_validation": deepcopy(
                    attempt.get("tool5_validation", attempt.get("tool_5_validation"))
                ),
            }
        )
    return records


def _prefer(values: list[str], preference: list[str]) -> str:
    for item in preference:
        if item in values:
            return item
    if values:
        return values[0]
    return preference[-1]
