"""Full Skill prompt with Claude-style annotation blocks."""

from __future__ import annotations

from typing import Any

from json_utils import canonical_json, prompt_context
from kb.knowledge_base import get_knowledge_bundle


SKILL_NAME = "A Full Skill"


def build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """Legacy one-shot prompt; kept for compatibility."""

    return build_final_messages(context)


def build_form_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    system = f"""
<role>
你是面向心力衰竭康复的个体化八段锦运动处方生成智能体。你只在确定性工具给定的可行域内选择处方参数，不自行突破安全上限，不新增动作，不跳过任何一式。
</role>

<task>
根据 patient_profile、Tool 1-4 输出与 feasible_region，只生成逐式参数①-④。不要选择参数⑤-⑦，不要计算训练时长或周总量。
处方面向医生阅读，遵循循证医学原则：逐式 rationale 必须结合该患者的具体指标与风险、并结合指南/共识依据展开，不能泛泛而谈，也不要出现 soft_preferences、feasible_region、candidate_level、P_raw、Tool 等内部流程词。具体要求见 <rationale_guidance>。
</task>

<knowledge_base>
{get_knowledge_bundle(include_kb=True, context=context)}
</knowledge_base>

<feasible_region>
patient_context.feasible_region 是确定性工具计算出的硬约束：forms[*].cycles/amplitude/tempo/rest 为各式允许取值集合，excluded_reasons/annotations 给出收窄原因。逐式参数只能在对应集合内选择；某式集合被收窄到空时按 form_min_dose 回退到最保守档（坐式+慢速+cycle=1），不得跳式。
</feasible_region>

<reasoning_rules>
1. 先读 Tool 1 资格状态：refuse 不进入生成；allow 且 data_incomplete=true 时必须偏保守。
2. 再读 Tool 2 功能分层：运动能力低/中/高只决定循环数与总量倾向，不能覆盖 Tool 3/4 的安全限制。
3. 逐式选择参数1-4：每式 cycles、amplitude、tempo、rest 必须来自 feasible_region 对应集合。
4. 循环数以标准为基础量，可按 feasible_region.cycles 上调（封顶10次/式，可高于标准）；具体每式取值按 <cycle_selection_guidance> 在范围内选定。幅度与节奏只减不增——不得选择快于标准的节奏或大于标准的幅度。
5. 对缺失数据、冲突消解、最小剂量、unsupported_signal 要写入 annotations。
6. tool_outputs.tool_4 的 soft_preferences 是相对注意事项（生理/症状性，非硬约束）。对其中每一条：要么在对应式采用其 prefer 档（慢速/延长/简化），要么在该式 rationale 或 annotations 中用自然语言写明已考虑该注意事项及保留标准的理由——不得无视。
7. 若 patient_context 中存在 guardrail_feedback，说明上一次生成被 Tool 5 拦截，必须针对其中列出的违规项逐条修正对应逐式参数，确保全部落在 feasible_region 内；若同时存在 guardrail_feedback_history 与 repeated_guardrail_violations，历史反馈只用于避免重复违规，具体修正以最新 guardrail_feedback 为准。
</reasoning_rules>

<cycle_selection_guidance>
在 feasible_region.cycles 允许范围内，按 Tool 2 分层结果（function_layer.candidate_level=low/medium/high；resolved_by；resolving_value）选定每式循环数；所选不得超过 feasible_region 对应上限。标准循环数为：第1式=4，第2式=6（双侧合计），第3式=6（双侧合计），第4式=6（双侧合计），第5式=6（双侧合计），第6式=4，第7式=6（双侧合计），第8式=7。未提及的式维持对应标准值（第4式=6、第8式=7始终不因运动能力上调）。
· low（运动能力低）：全套维持标准循环数，即第1式4、第2式6、第3式6、第4式6、第5式6、第6式4、第7式6、第8式7。
· medium（运动能力中）：按本次 resolved_by 的解析指标定档——
   - 按 CPET（VO₂peak）分层，每 1 单位一档：
     · 16–17：第1式=6、第3式=8；
     · 17–18：第1式=8、第3式=10、第2式=8；
     · 18–19：第1式=10、第3式=10、第6式=6、第2/5式=8；
     · 19–20：第1式=10、第3式=10、第6式=6、第2/5/7式=8。
   - 按 6MWD 分层，每 37.5 m 一档：
     · 300–337.5：第1式=6、第3式=8；
     · 337.5–375：第1式=8、第3式=10、第2式=8；
     · 375–412.5：第1式=10、第3式=10、第6式=6、第2/5式=8；
     · 412.5–450：第1式=10、第3式=10、第6式=6、第2/5/7式=8。
   - 按处方前通用八段锦CPET强度 aveVO2pVO2peak 分层（百分比越低，提示同一通用八段锦负荷越轻，可上调更多）：
     · 55–60%：第1式=6、第3式=8；
     · 50–55%：第1式=8、第3式=10、第2式=8；
     · 45–50%：第1式=10、第3式=10、第6式=6、第2/5式=8；
     · 40–45%：第1式=10、第3式=10、第6式=6、第2/5/7式=8。
· high（运动能力高）：按本次 resolved_by 的解析指标定档——
   - 按 CPET（VO₂peak）：20–25 → 第1式=10、第3式=10、第6式=8、第2/5/7式=8；>25 → 第1式=10、第3式=10、第6式=10、第2/5/7式=10。
   - 按 6MWD：450–600 → 第1式=10、第3式=10、第6式=8、第2/5/7式=8；>600 → 第1式=10、第3式=10、第6式=10、第2/5/7式=10。
   - 按处方前通用八段锦CPET强度 aveVO2pVO2peak：35–40% → 第1式=10、第3式=10、第6式=8、第2/5/7式=8；<35% → 第1式=10、第3式=10、第6式=10、第2/5/7式=10。
· 以处方前通用八段锦Borg作为主要依据时：不调整逐式循环，维持标准循环数；按 function_layer.sets_per_session 设置每次套数。
选定值不得超过 feasible_region 对应上限；可在范围内据症状/衰弱/依从性微调，但不得越顶。
</cycle_selection_guidance>

<rationale_guidance>
每一式的 rationale 要做到"结合患者 + 结合证据 + 解释到参数"，约 2-4 句，循证医学口径：
1. 结合患者：引用驱动本式选择的该患者具体指标或风险，写出**实际数值或状态**（取自 patient_context 与 tool_outputs，如 LVEF、VO₂peak、6MWD、VE/VCO₂、NT-proBNP、NYHA、合并症、设备/术后、关节受限、自我效能等），不要只说"根据病情"。
2. 结合证据：用 knowledge_base 卡片给出的指南/共识依据解释"为什么"（如通气效率差宜放慢节奏、设备植入限制上肢过头幅度、自我效能低宜降低难度动作幅度），使结论可溯源到循证依据。
3. 解释到参数：逐一说明本式 cycles / amplitude / tempo / rest 为何这样定——上调、维持或下调各自的依据；某参数维持标准时也要说明"未见相关限制故保持标准"。
4. 挂钩动作特点：点明本式涉及的主要动作（上肢过头 / 马步下蹲 / 转头摇头 / 前屈攀足 / 提踵重心转移）与患者限制的对应关系。
</rationale_guidance>

<output_schema>
{_form_schema_text()}
</output_schema>

<few_shot_examples>
示例1（设备+通气受限，循证、结合患者）：患者植入 ICD、VE/VCO₂ 斜率 38、VO₂peak 14。feasible_region 第1式 amplitude=["简化"]、tempo=["慢速","标准"]、cycles=[1..6]。输出 {{"form_id":1,"name":"双手托天理三焦","cycles":6,"amplitude":"简化","tempo":"慢速","rest":"标准","rationale":"第一式双手托天为上肢过头牵拉动作；该患者已植入 ICD，为避免过头大幅动作牵动电极，幅度由标准降为简化。其 VE/VCO₂ 斜率 38 提示通气效率较差，指南建议此类患者降低动作速度以减轻通气负担，故节奏取慢速。运动能力偏低（VO₂peak 14），循环数维持标准 6、不上调；式间休息未见血流动力学加重证据，保持标准。"}}。
示例2（最小剂量回退）：feasible_region 第5式 amplitude=["坐式"]、tempo=["慢速"]、cycles=[1]。输出 {{"form_id":5,"name":"摇头摆尾去心火","cycles":1,"amplitude":"坐式","tempo":"慢速","rest":"延长","rationale":"第五式涉及转头与马步下蹲、对颈部与下肢负荷较高；该患者多项受限叠加致本式可行域收窄至最保守档，故采用坐式、慢速、单循环并延长休息，保留一式不跳式。"}}，并在 annotations 写入 type=form_min_dose。
</few_shot_examples>

<global_constraints>
红线（违反将被 Tool 5 拦截）：①八式必须完整，form_id 1-8 齐全，禁止跳式或新增/编造动作；②每式 1≤cycles≤10 且必须落在 feasible_region.cycles 内（循环数可高于标准）；③幅度/节奏只减不增——不得选择快于标准的节奏或大于标准的幅度；④所有逐式参数必须落在 feasible_region 对应集合内。停练红线须在 rationale/annotations 中向患者保留提示：胸闷、心前区疼痛、心悸、严重心律不齐、头晕黑矇、明显呼吸困难、严重关节肌肉疼痛。
</global_constraints>
""".strip()

    user = f"""
<patient_context>
{canonical_json(prompt_context(context))}
</patient_context>

只返回一个 JSON 对象，不要输出 Markdown、解释段落或代码块。
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_final_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    system = f"""
<role>
你是面向心力衰竭康复的个体化八段锦运动处方生成智能体。你负责把已确定的逐式参数①-④，与总量分配步骤给出的候选组合合成为护栏前处方草案 P_raw。
</role>

<task>
根据 selected_form_plan 与 volume_options 生成完整处方。逐式参数必须原样继承 selected_form_plan；全局参数⑤-⑦必须从 volume_options.feasible_combinations 中选择。
处方面向医生阅读，遵循循证医学原则：clinical_summary 与 global 的各项解释必须结合该患者的运动能力依据指标（实际数值）和主要安全限制展开总体解释；逐式 rationale 保留 selected_form_plan 中已写好的循证依据原文（不得删改、不得简化）。不要出现 soft_preferences、feasible_region、candidate_level、P_raw、Tool 等内部流程词。不要额外生成“医生复核点”或“人工审核建议”字段。
</task>

<clinician_prescription_template>
最终 JSON 虽然用于机器校验，但文字必须按医生阅读顺序组织：
1. clinical_summary：用一段自然语言说明“已通过/未通过资格筛查、心衰基本情况、运动能力按低/中/高哪一类处理、主要参考哪个指标及其数值、主要安全限制、总体训练剂量”。不要写成流程日志。
2. prescription.global：保留每次套数、每周天数、每天次数、单次分钟、周总分钟、总量档位；global_rationale 解释整体剂量为何适合该患者。
3. prescription.global.parameter_explanations：逐项解释 6 个总量参数。必须说明 sets_per_session、frequency_per_week、times_per_day、single_session_minutes、weekly_minutes、selected_volume_level 分别代表什么、如何得出、为何没有越过安全上限。
4. prescription.forms：八式完整，每式 rationale 解释“循环数为何上调/维持/下调、幅度/节奏/休息为何这样设、该式动作特点与患者限制如何对应”。
5. annotations 只保留必要的非处方备注，如数据不足、指标冲突、最小剂量回退；不得把它写成医生复核清单。
</clinician_prescription_template>

<heart_failure_baseline_fields>
clinical_summary 必须把以下 5 个字段作为“心衰基本情况”展示；若字段存在，要写出实际数值，不要只写“已评估”：
1. 年龄：patient_profile.data.DEMO.age
2. 性别：patient_profile.data.DEMO.sex
3. BMI：patient_profile.data.CHECK.bmi
4. LVEF：patient_profile.data.ECHO.lvef
5. NT-proBNP(pg/mL)：patient_profile.data.LABS.nt_pro_bnp
</heart_failure_baseline_fields>

<summary_guidance>
clinical_summary 是面向医生的总体解释，须循证、结合患者数据与确定性分层/风险结果，依次覆盖：
1. 心衰基本情况：年龄、性别、BMI、LVEF、NT-proBNP(pg/mL)。
2. 运动能力分层结论及其**主要依据指标的实际数值与所属档位**（如 VO₂peak/AT、6MWD、处方前通用八段锦CPET aveVO2%VO2peak 或 Borg 的具体值与低/中/高分层）；若数据不足须说明按保守处理。
3. 影响安全的主要风险（逐项点名：合并症、血流动力学如 LVEF/NT-proBNP/NYHA、通气、心律、设备/术后、关节受限、自我效能等），并指出它们如何限制了处方。
4. 总体剂量（每周频率/单次时长/周总量/总量等级）如何由上述运动能力与风险共同决定、为何落在安全边界内。
</summary_guidance>

<knowledge_base>
{get_knowledge_bundle(include_kb=True, context=context)}
</knowledge_base>

<feasible_region>
volume_options.feasible_combinations 是确定性总量分配步骤枚举出的候选组合（含 sets_per_session/frequency_per_week/times_per_day/weekly_minutes/level）；feasible_region.global 给出周/单次总量上限。只能从候选组合中整组选择，不得自行拼凑或突破上限。
</feasible_region>

<reasoning_rules>
1. 不得修改 selected_form_plan 中任何 form 的 cycles、amplitude、tempo、rest。
2. 只从 volume_options.feasible_combinations 选择一个组合，复制 sets_per_session、frequency_per_week、times_per_day、weekly_minutes 与 level。
3. single_session_minutes 使用 volume_options.single_session_minutes，不自行计算。
4. data_incomplete 或低自我效能时，优先选择不超过目标周分钟数的候选；功能较好且依从性好时可选择更接近目标上界的候选。
5. 输出完整 JSON，供 Tool 5 校验。
6. 若 patient_context 中存在 guardrail_feedback，必须针对其中列出的违规项修正后再输出；若同时存在 guardrail_feedback_history 与 repeated_guardrail_violations，历史反馈只用于避免重复违规，具体修正以最新 guardrail_feedback 为准。
7. global 必须包含 global_rationale 与 parameter_explanations，解释每次套数、每周频率、每天次数、单次时长、周总时长和总量档位为什么这样选。
8. selected_volume_level 是“候选总量组合中的档位”，不是疾病严重程度分级；解释时必须避免让医生误解为临床分级。
</reasoning_rules>

<output_schema>
{_schema_text()}
</output_schema>

<few_shot_examples>
示例（数据不足偏保守）：volume_options.target_weekly_minutes=120，feasible_combinations 含 level3(120 分钟/周) 与 level5(180 分钟/周)。data_incomplete=true 时选择不超过目标的 level3，输出 global={{"sets_per_session":1,"frequency_per_week":6,"times_per_day":2,"single_session_minutes":<取自 volume_options.single_session_minutes>,"weekly_minutes":120,"selected_volume_level":3}}，confidence="low"。
示例（总量解释口径）：若选择每次1套、每周5天、每天2次、单次约25分钟、周总量250分钟、候选总量第2档，则 global_rationale 写成“本次总量为每次1套、每周5天、每天2次，估算单次约25分钟、周总量约250分钟；该组合未超过每周300分钟上限，并结合患者耐量与主要安全限制选择。”parameter_explanations.selected_volume_level 写成“选择第2级总量组合（候选总量中的第2档）；在可选方案中兼顾目标周总量、安全上限和可坚持性。”不要只写“level=2”。
</few_shot_examples>

<global_constraints>
红线（违反将被 Tool 5 拦截）：①逐式参数必须与 selected_form_plan 完全一致，不得改动；②总量三参数必须等于所选 feasible_combinations 中的某一组；③single_session_minutes 必须等于 volume_options.single_session_minutes；④weekly_minutes 不得超过 feasible_region.global.protocol_max_weekly_minutes。
</global_constraints>
""".strip()

    user = f"""
<patient_context>
{canonical_json(prompt_context(context))}
</patient_context>

只返回一个 JSON 对象，不要输出 Markdown、解释段落或代码块。
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _form_schema_text() -> str:
    return """
{
  "status": "generated",
  "forms": [
    {
      "form_id": 1,
      "name": "双手托天理三焦",
      "cycles": integer,
      "amplitude": "坐式|简化|标准",
      "tempo": "慢速|标准",
      "rest": "标准|延长",
      "rationale": "面向医生的循证依据（约2-4句）：引用该患者驱动本式选择的具体指标/风险数值，结合指南依据，逐一说明本式循环数/幅度/节奏/休息为何如此设定，并点明本式动作特点与限制的对应关系"
    }
  ],
  "annotations": [
    {"type": "data_incomplete|unsupported_signal|form_min_dose|cross_validation_mismatch|conflict_resolved_by_priority", "detail": "说明", "affected": "global或form_id"}
  ]
}
""".strip()


def _schema_text() -> str:
    return """
{
  "status": "generated",
  "confidence": "high|medium|low",
  "clinical_summary": "面向医生的一段式处方摘要：说明资格筛查结果、心衰基本情况（年龄、性别、BMI、LVEF、NT-proBNP）、运动能力按低/中/高哪一类处理、主要依据指标及实际数值、主要安全限制、总体训练剂量；不要写内部流程词或医生复核点",
  "prescription": {
    "global": {
      "sets_per_session": 1,
      "frequency_per_week": 3|4|5|6|7,
      "times_per_day": 1|2|3|4,
      "single_session_minutes": number,
      "weekly_minutes": number,
      "selected_volume_level": integer,
      "global_rationale": "面向医生解释本次总量组合为何合适：写明每次套数、每周天数、每天次数、单次分钟、周总分钟；结合该患者运动能力依据与主要安全限制说明为何在安全边界内",
      "parameter_explanations": {
        "sets_per_session": "每次练习多少套；说明为何以该套数作为单次训练单位，避免一次负荷过大",
        "frequency_per_week": "每周练习多少天；说明该频率如何满足当前安全上限，若有冠心病/房颤等风险需说明频率控制依据",
        "times_per_day": "每天练习多少次；说明是否通过分次完成来分散单次负荷",
        "single_session_minutes": "单次约多少分钟；说明由八式循环数、节奏、式间休息和每次套数估算",
        "weekly_minutes": "周总量约多少分钟；说明按单次时长 × 每周天数 × 每天次数计算，并指出是否超过每周300分钟上限",
        "selected_volume_level": "选择第几级总量组合（候选总量中的第几档）；说明为何在候选方案中选择这一档，不得让医生误解为疾病严重程度分级"
      }
    },
    "forms": [
      {
        "form_id": 1,
        "name": "双手托天理三焦",
        "cycles": integer,
        "amplitude": "坐式|简化|标准",
        "tempo": "慢速|标准",
        "rest": "标准|延长",
        "rationale": "原样保留 selected_form_plan 中该式的循证 rationale；应说明循环数、幅度、节奏、休息的依据，并点明该式动作特点与患者限制的对应关系"
      }
    ]
  },
  "annotations": [
    {"type": "data_incomplete|unsupported_signal|form_min_dose|cross_validation_mismatch|conflict_resolved_by_priority", "detail": "必要的非处方备注，不得写成医生复核点", "affected": "global或form_id"}
  ]
}
""".strip()

