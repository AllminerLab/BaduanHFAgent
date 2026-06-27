"""Replaceable LLM client layer."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Callable
from urllib import request
from urllib.error import HTTPError, URLError

from config import LLMConfig
from json_utils import extract_json_object


class LLMClientError(RuntimeError):
    """Raised when the configured LLM endpoint cannot return a usable answer."""


class OpenAICompatibleLLMClient:
    """Minimal OpenAI-compatible chat-completions client.

    This client works with providers that expose a /chat/completions-compatible
    API. To support a non-compatible provider later, implement another client
    with the same chat/generate_json methods and inject it into BaduanjinAgent.
    """

    def __init__(self, config: LLMConfig, progress: Callable[[str], None] | None = None):
        self.config = config
        self.progress = progress
        self._last_finish_reason: str | None = None

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        api_key = self.config.resolved_api_key
        if not api_key:
            raise LLMClientError(
                "LLM api key is empty. Set api_key or api_key_env in the LLM config, "
                "or use provider='mock' for local deterministic runs."
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        endpoint = self._chat_endpoint()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **self.config.extra_headers,
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started_at = time.monotonic()
        self._progress(
            "LLM API 调用中: "
            f"provider={self.config.provider}, model={self.config.model}, "
            f"json_mode={json_mode}, timeout={self.config.timeout_seconds}s"
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                elapsed = time.monotonic() - started_at
                self._progress(
                    "LLM API 调用成功: "
                    f"status={getattr(response, 'status', 'unknown')}, "
                    f"elapsed={elapsed:.1f}s, response_chars={len(raw)}"
                )
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started_at
            self._progress(
                "LLM API 调用失败: "
                f"HTTP {exc.code}, elapsed={elapsed:.1f}s"
            )
            raise LLMClientError(f"LLM request failed: HTTP {exc.code}: {body}") from exc
        except (socket.timeout, TimeoutError) as exc:
            elapsed = time.monotonic() - started_at
            self._progress(
                "LLM API 调用超时: "
                f"elapsed={elapsed:.1f}s, timeout={self.config.timeout_seconds}s"
            )
            raise LLMClientError(
                "LLM request timed out after "
                f"{self.config.timeout_seconds} seconds. Increase timeout_seconds "
                "in the LLM config or retry later."
            ) from exc
        except URLError as exc:
            elapsed = time.monotonic() - started_at
            self._progress(
                "LLM API 调用失败: "
                f"network_error={exc.reason}, elapsed={elapsed:.1f}s"
            )
            raise LLMClientError(f"LLM request failed: {exc.reason}") from exc

        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            content = choice["message"]["content"]
            self._last_finish_reason = choice.get("finish_reason")
            if self._last_finish_reason == "length":
                self._progress(
                    "LLM API 输出被截断: finish_reason=length，"
                    f"已达 max_tokens={self.config.max_tokens}（推理模型时该上限同时覆盖『推理+内容』）"
                )
            self._progress(f"LLM API 内容提取成功: content_chars={len(str(content or ''))}")
            return content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self._progress("LLM API 响应解析失败: 返回结构不是预期的 chat-completions JSON")
            raise LLMClientError(f"Unexpected LLM response: {raw[:1000]}") from exc

    def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        max_retries = max(0, int(self.config.json_parse_retries))
        retry_messages = messages
        last_error = "LLM returned malformed JSON."
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            content = self.chat(retry_messages, json_mode=True)
            try:
                parsed = extract_json_object(content)
            except json.JSONDecodeError as exc:
                last_exc = exc
                # 截断（finish_reason=length）丢的是数据不是语法，重发"修语法"无效——直接给出可操作错误。
                if self._last_finish_reason == "length":
                    raise LLMClientError(
                        "LLM 输出在 max_tokens 处被截断（finish_reason=length），返回的 JSON 不完整、无法解析。"
                        f"当前 max_tokens={self.config.max_tokens}。请调大 LLM 配置的 max_tokens"
                        "（deepseek-v4-pro 等推理模型，该上限同时覆盖『推理+最终内容』，需更大余量）。"
                    ) from exc
                last_error = (
                    "LLM returned malformed JSON: "
                    f"{exc.msg} at line {exc.lineno} column {exc.colno} "
                    f"(char {exc.pos})."
                )
                self._progress(
                    "LLM JSON 解析失败: "
                    f"模型返回的内容不是合法 JSON，line={exc.lineno}, "
                    f"column={exc.colno}, char={exc.pos}"
                )
            else:
                if isinstance(parsed, dict):
                    suffix = f"，JSON 修复重试 {attempt} 次后成功" if attempt else ""
                    self._progress(f"LLM JSON 解析成功{suffix}")
                    return parsed
                last_error = "LLM returned JSON but not a JSON object."
                self._progress("LLM JSON 解析失败: 模型返回的 JSON 不是对象")

            if attempt >= max_retries:
                break
            self._progress(
                "LLM JSON 修复重试: "
                f"第 {attempt + 1}/{max_retries} 次；这是格式修复，不是 Tool5 护栏重生成"
            )
            retry_messages = _json_repair_messages(messages, last_error, content)

        raise LLMClientError(last_error) from last_exc

    def _chat_endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


def _json_repair_messages(
    messages: list[dict[str, str]],
    error: str,
    malformed_content: str,
) -> list[dict[str, str]]:
    repair_instruction = (
        "上一次回答无法被 JSON 解析器解析，错误为："
        f"{error}\n"
        "请基于下面的“上一次原始回答”修复 JSON 语法，并重新输出一个完整、合法的 JSON 对象。要求：\n"
        "1. 只输出 JSON 对象本身，不要输出 Markdown、代码块或解释文字。\n"
        "2. 所有对象成员之间必须有英文逗号。\n"
        "3. 字符串必须使用双引号，不能使用单引号。\n"
        "4. 不要出现尾随逗号、注释、NaN、Infinity 或省略号。\n"
        "5. 保持原任务要求和 schema 不变，不要新增字段，不要改动处方参数含义。\n\n"
        "<上一次原始回答>\n"
        f"{_clip_for_repair(malformed_content)}\n"
        "</上一次原始回答>"
    )
    return [*messages, {"role": "user", "content": repair_instruction}]


def _clip_for_repair(text: str, limit: int = 20000) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...（内容过长，后续已截断）"
