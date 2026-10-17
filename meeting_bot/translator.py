# -*- coding: utf-8 -*-
"""翻译与纪要生成：调用 OpenCode Go（OpenAI 兼容接口）。"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import openai

from .config import Settings


class TranslationError(Exception):
    """翻译接口不可用（重试后仍失败）。"""


def session_headers() -> dict[str, str]:
    """OpenCode Go 路由所需请求头。

    接口对缺少 ``x-opencode-session`` 的请求直接返回 400
    （``MissingSessionID``：「cannot be routed efficiently」），
    翻译/纪要会全线不可用。会话 ID 仅用于服务端路由，取进程级随机值即可；
    需要固定值（例如复现问题）时可用环境变量 ``OPENCODE_SESSION_ID`` 覆盖。
    """
    sid = (os.getenv("OPENCODE_SESSION_ID") or "").strip()
    return {"x-opencode-session": sid or f"meeting-bot-{uuid.uuid4().hex[:8]}"}


@dataclass
class ActionItem:
    task: str
    owner: str = ""
    due: str = ""


@dataclass
class MinutesResult:
    title: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


SYSTEM_TRANSLATE = (
    "你是会议实时翻译引擎。把用户发言翻译成简体中文。"
    "语境：医疗/商务会议口语。"
    "要求：1) 只输出译文本身，不要任何解释或前缀；"
    "2) 保留人名、产品名、数字；3) 口语化的表述译成自然流畅的中文；"
    "4) 如果输入本身就是中文，原样输出；"
    "5) 修正口语重复、语气词与识别错字（结合上下文推测本意）；"
    "6) 医学术语与药品/器械名称译准确（不确定时保留原文）；"
    "7) 数字、比例、金额、日期必须精确对应；"
    "8) 地名/公司名/品牌名译成通行的中文名称（无通行译名保留原文）。"
)

MINUTES_PROMPT = """你是资深会议记录助理。根据下面的会议逐字稿，生成会议纪要。
严格输出 JSON，不要输出其他内容，格式：
{"title": "会议主题", "summary": "3-5句概要",
 "key_points": ["讨论要点1"],
 "decisions": ["结论1"],
 "action_items": [{"task": "任务", "owner": "责任人", "due": "截止时间（无则留空）"}],
 "open_questions": ["待跟进问题1"]}

逐字稿：
{transcript}"""


class Translator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = openai.OpenAI(
            base_url=settings.opencode_base_url,
            api_key=settings.opencode_api_key,
            timeout=30.0,
            max_retries=0,
            default_headers=session_headers(),
        )
        self._thinking_disabled_ok: bool | None = None

    def _chat(self, messages: list[dict], max_tokens: int = 512, model: str | None = None) -> str:
        extra = {"thinking": {"type": "disabled"}} if self._thinking_disabled_ok is not False else None
        for attempt in range(3):
            try:
                kwargs = dict(model=model or self._settings.translate_model,
                              messages=messages, max_tokens=max_tokens, temperature=0.0)
                if extra is not None:
                    kwargs["extra_body"] = extra
                resp = self._client.chat.completions.create(**kwargs)
                if extra is not None:
                    self._thinking_disabled_ok = True
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise TranslationError("接口返回空内容")
                return content
            except TranslationError:
                raise
            except openai.APIError as e:
                if extra is not None:
                    # 接口可能不支持 thinking 参数 → 降级为普通请求重试
                    extra = None
                    self._thinking_disabled_ok = False
                    continue
                if attempt == 2:
                    raise TranslationError(f"翻译接口不可用：{e}") from e
                time.sleep(2 ** attempt)
            except (IndexError, AttributeError, ValueError, TypeError) as e:
                # 响应结构异常（choices 为空 / message 缺失 / 字段类型不合法）：
                # 非 APIError，不重试，直接降级为 TranslationError 交调用方处理。
                # TypeError 必须在内（最终审查 Critical-1）：choices=None 时
                # resp.choices[0] 抛 TypeError("'NoneType' object is not subscriptable")，
                # 漏掉它会穿透 stop()/worker 的 TranslationError 处理链。
                raise TranslationError(f"接口响应结构异常：{e}") from e
        raise TranslationError("翻译接口不可用")

    def translate(self, text: str, lang: str = "") -> str:
        src = f"（原文语种：{lang}）" if lang and lang != "zh" else ""
        messages = [
            {"role": "system", "content": SYSTEM_TRANSLATE},
            {"role": "user", "content": f"{src}\n{text}"},
        ]
        return self._chat(messages, max_tokens=512)

    def translate_stream(self, text: str, lang: str = "") -> Iterator[str]:
        """流式翻译：每次 yield **累积**译文（不是增量），供字幕逐字上屏。

        控制器实测（本机真 API，deepseek-v4-pro）：首 token 1.99s、全文 2.28s，
        非流式 4.2s —— 逐字上屏把「等整句」变成「听到一半就在看」。

        与 :meth:`_chat` 一致的两件状态机：
        * ``extra_body={"thinking": {"type": "disabled"}}`` 首次即带（不带则会先流
          144 块 reasoning_content，首字延迟翻倍）；接口不认该参数（400）时降级
          重试一次，并把结果记进 ``_thinking_disabled_ok`` 供后续请求复用。
        * 连接建立失败（create 抛错）沿用既有退避重试；**流中途断开不重试**——
          半句已在屏上，重试会重放前缀，由调用方把译文清空退回原文。

        生成器语义：调用时不发请求（首次 next() 才发）。流正常结束但 content
        全空 → 抛 :class:`TranslationError`。
        """
        src = f"（原文语种：{lang}）" if lang and lang != "zh" else ""
        messages = [
            {"role": "system", "content": SYSTEM_TRANSLATE},
            {"role": "user", "content": f"{src}\n{text}"},
        ]
        extra = ({"thinking": {"type": "disabled"}}
                 if self._thinking_disabled_ok is not False else None)
        stream = None
        for attempt in range(3):
            try:
                kwargs = dict(model=self._settings.translate_model,
                              messages=messages, max_tokens=512, temperature=0.0,
                              stream=True)
                if extra is not None:
                    kwargs["extra_body"] = extra
                stream = self._client.chat.completions.create(**kwargs)
                if extra is not None:
                    self._thinking_disabled_ok = True
                break
            except openai.APIError as e:
                if extra is not None:
                    # 接口不认 thinking 参数（400 类）→ 降级为普通流式请求重试
                    extra = None
                    self._thinking_disabled_ok = False
                    continue
                if attempt == 2:
                    raise TranslationError(f"翻译接口不可用：{e}") from e
                time.sleep(2 ** attempt)
            except (IndexError, AttributeError, ValueError, TypeError) as e:
                raise TranslationError(f"接口响应结构异常：{e}") from e
        if stream is None:
            raise TranslationError("翻译接口不可用")

        parts: list[str] = []
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue          # 心跳/空块：跳过，不打断累积
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                # thinking 未被禁用时接口先流 reasoning_content（实测 144 块）：
                # 那是模型的思考过程，绝不能混进译文（否则字幕先冒出半页推理）
                if getattr(delta, "reasoning_content", None):
                    continue
                piece = getattr(delta, "content", None)
                if not piece:
                    continue
                parts.append(piece)
                yield "".join(parts)
        except openai.APIError as e:
            # 流中途断开：已显示部分保留在调用方，不重试（重试会重放前缀）
            raise TranslationError(f"翻译流中断：{e}") from e
        except (IndexError, AttributeError, ValueError, TypeError) as e:
            raise TranslationError(f"接口响应结构异常：{e}") from e
        except Exception as e:
            # 其它边界异常（SDK 内部错误等）同样降级为 TranslationError：
            # recorder 只对 TranslationError 做「清空译文 + 告警」的确定性处理。
            raise TranslationError(f"翻译流中断：{e}") from e
        if not "".join(parts).strip():
            raise TranslationError("接口返回空内容")

    def minutes(self, transcript_md: str) -> MinutesResult:
        # 不能用 str.format：模板内的 JSON 字面量（{"title": …}）会被当成占位符，
        # 直接抛 KeyError: '"title"'，纪要功能全线不可用。用 replace 注入逐字稿。
        messages = [
            {"role": "system", "content": "你是资深会议记录助理，输出严格 JSON。"},
            {"role": "user", "content": MINUTES_PROMPT.replace("{transcript}",
                                                               transcript_md[-180000:])},
        ]
        raw = self._chat(messages, max_tokens=8000, model=self._settings.minutes_model)
        return parse_minutes(raw)


def parse_minutes(raw: str) -> MinutesResult:
    """解析纪要 JSON；失败降级为纯文本纪要。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        return MinutesResult(
            title=str(data.get("title") or "会议纪要"),
            summary=str(data.get("summary") or ""),
            key_points=[str(x) for x in data.get("key_points") or []],
            decisions=[str(x) for x in data.get("decisions") or []],
            action_items=[
                ActionItem(task=str(x.get("task", "")), owner=str(x.get("owner", "")),
                           due=str(x.get("due", "")))
                for x in data.get("action_items") or []
            ],
            open_questions=[str(x) for x in data.get("open_questions") or []],
        )
    except (json.JSONDecodeError, AttributeError):
        return MinutesResult(title="会议纪要", summary=raw)
