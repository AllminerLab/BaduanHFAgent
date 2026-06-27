"""Offline tool: assist authoring of dynamic KB cards from guideline .docx.

为什么有它：动态卡片若全手写，知识一多维护成本线性上升。本脚本把"从几十页里
找相关句"自动化成确定性候选，医生只需在候选里勾选、定稿，再冻结进
kb.knowledge_base.DYNAMIC_CARDS。运行时不变（仍是冻结的卡 + 确定性信号匹配）。

流程：docx → 段落 → 句子 → 按 ANCHOR_MAP 关键词命中 → 清洗(去文献号/空格) →
打分排序 → 产出"候选卡"评审队列（含 suggested_trigger / candidate_text /
source_locator）。可选：若传入 LLM 客户端，对候选做润色/择优（默认不启用）。

注意：这是离线策展脚本，不被运行时 import；只处理 .docx（大 PDF 不在此处抽取）。
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Callable

_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

DEFAULT_QWEN_API_KEY = (
    os.environ.get("QWEN_API_KEY")
    or os.environ.get("DASHSCOPE_API_KEY")
)
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_QWEN_MODEL = "qwen3-max"
DEFAULT_QWEN_TEMPERATURE = 0.2
DEFAULT_QWEN_TIMEOUT_SECONDS = 90

DEFAULT_DOCX_PATHS = [
    "data/知识库/六分钟步行试验临床规范应用中国专家共识.docx",
    "data/知识库/慢性心力衰竭心脏康复中国专家共识.docx",
    "data/知识库/Exercise training in heart failure- from theory to practice.docx",
    "data/知识库/心肺运动试验的临床应用.docx",
]

CARD_SPECS: list[dict[str, Any]] = [
    {"id": "cpet_grounding", "triggers": ["has_cpet"]},
    {"id": "six_mwd_grounding", "triggers": ["has_6mwd"]},
    {"id": "borg_grounding", "triggers": ["resolved_borg"]},
    {"id": "data_incomplete_grounding", "triggers": ["data_incomplete"]},
    {"id": "low_function_grounding", "triggers": ["function_low"]},
    {"id": "mid_function_grounding", "triggers": ["function_mid"]},
    {"id": "high_function_grounding", "triggers": ["function_high"]},
    {"id": "beta_blocker_grounding", "triggers": ["beta_blocker"]},
    {"id": "ventilatory_grounding", "triggers": ["copd", "dyspnea_high", "high_ve_vco2"]},
    {"id": "rhythm_grounding", "triggers": ["arrhythmia"]},
    {"id": "device_sternal_grounding", "triggers": ["device_implant", "post_cabg"]},
    {"id": "severe_valvular_grounding", "triggers": ["severe_valvular"]},
    {"id": "cad_af_grounding", "triggers": ["cad_or_af"]},
    {
        "id": "hemodynamic_burden_grounding",
        "triggers": ["high_bnp", "high_peak_sbp", "high_rest_sbp", "high_rest_hr", "low_hrr", "low_lvef", "nyha_iii"],
    },
    {"id": "fatigue_grounding", "triggers": ["fatigue_high"]},
    {"id": "adherence_grounding", "triggers": ["low_self_efficacy"]},
]

# 触发信号 -> 关键词锚点。这是唯一需要维护的"小词表"，远小于手写每张卡。
# 信号名与 knowledge_base._patient_signals / DYNAMIC_CARDS.triggers 保持一致。
ANCHOR_MAP: dict[str, list[str]] = {
    "has_cpet": ["CPET", "心肺运动试验", "峰值摄氧", "VO2peak", "无氧阈", "VE/VCO2"],
    "has_6mwd": ["6MWD", "6 分钟步行", "六分钟步行", "6分钟步行"],
    "resolved_borg": ["Borg", "RPE", "自感劳累", "主观疲劳"],
    "data_incomplete": ["评估", "危险分层", "监护", "临床状态", "检查"],
    "function_low": ["低强度", "低起点", "慢推进", "Gradual mobilization", "start low", "severe HF"],
    "function_mid": ["中等强度", "moderate", "Borg RPE", "结构化运动", "规律体力活动"],
    "function_high": ["个体化", "individualized", "training intensity", "VO2peak", "VO2reserve", "HRR", "Borg RPE"],
    "low_lvef": ["LVEF", "射血分数", "左室收缩功能"],
    "high_bnp": ["利钠肽", "BNP", "NT-proBNP"],
    "high_peak_sbp": ["峰值血压", "收缩压", "systolic blood pressure"],
    "low_hrr": ["心率储备", "HRR", "heart rate reserve"],
    "copd": ["COPD", "慢性阻塞性肺"],
    "dyspnea_high": ["呼吸困难", "气促", "dyspnoea", "dyspnea"],
    "high_ve_vco2": ["VE/VCO2", "VE/VCO₂", "通气效率", "通气当量"],
    "arrhythmia": ["心律失常", "心房颤动", "房颤", "室性"],
    "cad_or_af": ["冠心病", "房颤", "心绞痛", "缺血", "ST 段", "atrial fibrillation", "ischaemia"],
    "beta_blocker": ["β受体阻滞", "β 受体阻滞", "受体阻滞剂"],
    "severe_valvular": ["瓣膜", "主动脉瓣狭窄", "二尖瓣"],
    "nyha_iii": ["NYHA", "心功能分级"],
    "device_implant": ["起搏器", "ICD", "CRT", "植入"],
    "post_cabg": ["CABG", "冠状动脉旁路", "心外科术后", "术后急性期"],
    "contraindication": ["禁忌"],
    "stop_rule": ["终止运动", "停止运动", "停练"],
    "fatigue_high": ["疲劳", "乏力", "fatigue", "Borg", "RPE"],
    "low_self_efficacy": ["依从性", "自我效能", "坚持", "adherence", "self-efficacy", "心理"],
}


LLM_REFINEMENT_INSTRUCTIONS = """
你在离线知识库策展流程中工作。请把候选原文整理成一张“通用医学知识卡”，用于给八段锦训练建议提供医学背景介绍。

硬性要求：
1. 只能使用候选原文和出处中的信息，不编造新的阈值、疾病结论或处方参数。
2. 卡片可以是一小段，不限 1-2 句，但要紧凑、可读、能直接放进 prompt。
3. 卡片只能做通用医学背景介绍，不要出现“Tool”“工具”“可行域”“确定性”“算法”“课题独创”“专项八段锦CPET”等内部框架或项目特异内容。
4. 若指南原文给出通用有氧训练剂量（如每周 3-5 天、每次 20-60 分钟、HRR/最大心率范围），不要把它改写成八段锦固定处方；只能概括为“通用训练需个体化并监测”。
5. 如果候选证据不足以支持该卡，请返回 insufficient_evidence=true，不要硬写。

返回 JSON：
{
  "id": "<card_id>",
  "text": "<接地卡正文>",
  "source_locator": "<保留最相关出处，多个用分号分隔>",
  "insufficient_evidence": false
}
""".strip()


def build_llm_refine_prompt(card_id: str, candidates: list[dict[str, Any]]) -> str:
    """Build the stricter prompt used by optional offline LLM refinement."""

    return "\n\n".join(
        [
            LLM_REFINEMENT_INSTRUCTIONS,
            f"card_id: {card_id}",
            "候选原文：",
            _json_for_prompt(candidates),
        ]
    )


def qwen_refine_card(
    card_id: str,
    candidates: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    temperature: float = DEFAULT_QWEN_TEMPERATURE,
    timeout_seconds: int = DEFAULT_QWEN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Use Qwen to turn candidate snippets into one curated card body."""

    prompt = build_llm_refine_prompt(card_id, candidates)
    return _qwen_json(
        prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def build_qwen_dynamic_cards(
    docx_paths: list[str],
    *,
    review_queue: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    temperature: float = DEFAULT_QWEN_TEMPERATURE,
    max_candidates_per_card: int = 12,
) -> list[dict[str, Any]]:
    """Build frozen dynamic-card JSON with Qwen refinement.

    The module-level defaults can be overridden by QWEN_API_KEY,
    DASHSCOPE_API_KEY, or a CLI argument for one-off local use.
    """

    queue = review_queue if review_queue is not None else build_review_queue(docx_paths)
    cards: list[dict[str, Any]] = []
    for spec in CARD_SPECS:
        triggers = set(spec["triggers"])
        candidates = [
            item for item in queue if item.get("suggested_trigger") in triggers
        ][:max_candidates_per_card]
        if not candidates:
            continue
        refined = qwen_refine_card(
            spec["id"],
            candidates,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
        if refined.get("insufficient_evidence"):
            continue
        text = str(refined.get("text") or "").strip()
        if not text:
            continue
        cards.append(
            {
                "id": spec["id"],
                "triggers": spec["triggers"],
                "source": _source_summary(candidates, refined.get("source_locator")),
                "text": text,
                "source_locator": str(refined.get("source_locator") or _join_locators(candidates)).strip(),
            }
        )
    return cards


def extract_candidates(
    docx_path: str,
    max_per_trigger: int = 3,
    llm_refine: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Return ranked candidate cards from one .docx (deterministic).

    If ``llm_refine`` is given (e.g. wrapping an LLM client), it post-processes the
    candidates (pick best / rephrase / refine triggers); otherwise the deterministic
    candidates are returned as-is for human review.
    """

    file_name = docx_path.rsplit("/", 1)[-1]
    paras = _paragraphs(docx_path)

    candidates: list[dict[str, Any]] = []
    for trigger, terms in ANCHOR_MAP.items():
        seen: set[str] = set()
        scored: list[tuple[float, dict[str, Any]]] = []
        for para_idx, para in enumerate(paras):
            for sentence in _sentences(para):
                if not any(term in sentence for term in terms):
                    continue
                text = _clean(sentence)
                if len(text) < 8 or text in seen or _looks_like_heading(text):
                    continue
                seen.add(text)
                matched = next(t for t in terms if t in sentence)
                scored.append(
                    (
                        _score(text, matched),
                        {
                            "suggested_trigger": trigger,
                            "matched_term": matched,
                            "candidate_text": text,
                            "source_file": file_name,
                            "source_locator": f"{file_name} · 第{para_idx + 1}段",
                            "char_len": len(text),
                        },
                    )
                )
        scored.sort(key=lambda item: item[0], reverse=True)
        # 只保留分数过线的（滤掉枚举/标题等噪声），每个 trigger 取前 N。
        candidates.extend(
            item for score_value, item in scored[:max_per_trigger] if score_value >= 0.6
        )

    if llm_refine is not None:
        candidates = llm_refine(candidates)
    return candidates


def _paragraphs(path: str) -> list[str]:
    root = ET.parse(zipfile.ZipFile(path).open("word/document.xml")).getroot()
    out: list[str] = []
    for p in root.iter(f"{_NS}p"):
        text = "".join(t.text or "" for t in p.iter(f"{_NS}t")).strip()
        if text:
            out.append(text)
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", text)
    return [s.strip() for s in parts if s.strip()]


def _clean(sentence: str) -> str:
    # 去文献号 ［79-80］/[81]，折叠多余（含全角）空格，去首尾标点噪声。
    sentence = re.sub(r"[\[［][0-9,\-–\s]+[\]］]", "", sentence)
    sentence = re.sub(r"[ 　]+", "", sentence) if _is_spaced_cjk(sentence) else re.sub(r"\s+", " ", sentence)
    return sentence.strip(" ；;。、")


def _is_spaced_cjk(s: str) -> bool:
    # 文档里常见"死 亡 风 险"这种逐字空格；检测 CJK 间空格占比高则全部去空格。
    cjk_space = len(re.findall(r"[一-鿿]\s[一-鿿]", s))
    return cjk_space >= 3


_ASSERT_WORDS = ["是", "应", "建议", "提示", "相关", "增加", "降低", "预后", "风险", "禁忌", "终止", "推荐", "可"]


def _score(text: str, matched_term: str) -> float:
    score = 1.0
    enum = text.count("、")
    if enum >= 4:
        score -= 1.0  # 适应证/疾病清单这类枚举句，几乎都是噪声
    elif enum >= 2:
        score -= 0.3
    if "系统疾病" in text or "等疾病" in text:
        score -= 0.6  # 典型枚举前缀
    if "专家共识" in text and len(text) < 50:
        score -= 1.0  # 文档标题/页眉
    if "特发性肺间质纤维化" in text or "IPF" in text:
        score -= 0.8  # 6MWT 共识里的旁支疾病段落，通常不适合作为心衰八段锦卡
    if len(text) > 120:
        score -= 0.4
    if len(text) > 200:
        score -= 0.6
    score -= 0.1 * len(re.findall(r"[0-9]{4,}", text))  # 年份/编号噪声
    if any(word in text for word in _ASSERT_WORDS):
        score += 0.3  # 偏好陈述性结论句
    if text.startswith(matched_term):
        score += 0.1
    return score


def _looks_like_heading(text: str) -> bool:
    # 短、且无内部标点 → 多半是标题/小节名，不是可引用的结论句。
    if len(text) < 16 and not re.search(r"[，、：；]", text):
        return True
    if "专家共识" in text and len(text) < 50:
        return True
    if re.match(r"^(图|表|Table|Figure)\s*\d+", text):
        return True
    return False


def build_review_queue(docx_paths: list[str]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for path in docx_paths:
        queue.extend(extract_candidates(path))
    return queue


def _json_for_prompt(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _qwen_json(
    prompt: str,
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    import json

    key = api_key or DEFAULT_QWEN_API_KEY
    if not key:
        raise RuntimeError(
            "Missing Qwen API key. Set QWEN_API_KEY or DASHSCOPE_API_KEY, "
            "or pass --qwen-api-key for one-off local use."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨的医学知识库策展助手，只返回合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen request failed before HTTP response: {exc.reason}") from exc

    response_payload = json.loads(raw)
    content = response_payload["choices"][0]["message"]["content"]
    try:
        return json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        preview = content[:500].replace("\n", "\\n")
        raise RuntimeError(f"Qwen response was not valid JSON: {preview}") from exc


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return text.strip()


def _source_summary(candidates: list[dict[str, Any]], source_locator: Any = None) -> str:
    files: list[str] = []
    locator_text = str(source_locator or "")
    if locator_text:
        for part in re.split(r"[;,；，]", locator_text):
            name = part.strip().split("·", 1)[0].strip()
            if name and name not in files:
                files.append(name)
    for item in candidates:
        name = str(item.get("source_file") or "").strip()
        if name and name not in files:
            files.append(name)
    return " / ".join(files[:3])


def _join_locators(candidates: list[dict[str, Any]]) -> str:
    locators: list[str] = []
    for item in candidates:
        locator = str(item.get("source_locator") or "").strip()
        if locator and locator not in locators:
            locators.append(locator)
    return "；".join(locators[:6])


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build offline dynamic-card review candidates from docx files.")
    parser.add_argument("paths", nargs="*", help="DOCX paths. Defaults to all configured knowledge-base docx files.")
    parser.add_argument(
        "--out",
        default="data/知识库/card_candidates.json",
        help="JSON output path for the review queue.",
    )
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "--qwen-refine",
        dest="qwen_refine",
        action="store_true",
        default=True,
        help="Use Qwen to generate dynamic cards from the review queue instead of only outputting candidates.",
    )
    refine_group.add_argument(
        "--no-qwen-refine",
        dest="qwen_refine",
        action="store_false",
        help="Only write the candidate review queue and skip Qwen refinement.",
    )
    parser.add_argument(
        "--approved-out",
        default="src/kb/dynamic_cards.json",
        help="Output path for Qwen-refined dynamic cards.",
    )
    parser.add_argument(
        "--qwen-api-key",
        default=None,
        help="One-off Qwen API key. Prefer QWEN_API_KEY/DASHSCOPE_API_KEY environment variables.",
    )
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-temperature", type=float, default=DEFAULT_QWEN_TEMPERATURE)
    args = parser.parse_args()

    paths = args.paths or DEFAULT_DOCX_PATHS
    queue = build_review_queue(paths)
    candidate_text = json.dumps(queue, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(candidate_text)
            handle.write("\n")
    elif not args.qwen_refine:
        print(candidate_text)

    if args.qwen_refine:
        cards = build_qwen_dynamic_cards(
            paths,
            review_queue=queue,
            api_key=args.qwen_api_key,
            base_url=args.qwen_base_url,
            model=args.qwen_model,
            temperature=args.qwen_temperature,
        )
        text = json.dumps(cards, ensure_ascii=False, indent=2)
        output_path = args.approved_out
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
        else:
            print(text)
