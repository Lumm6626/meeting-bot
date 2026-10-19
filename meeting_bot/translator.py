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
    "8) 地名/公司名/品牌名译成通行的中文名称（无通行译名保留原文）；"
    "9) 结合上文参考统一术语、人名、称呼的译法；口音/识别导致的错字按上下文推测修正。"
)

CONTEXT_HEADER = "【上文参考（用于统一术语/人名并修正识别错字）】"
TARGET_HEADER = "【待翻译】"


def build_user_message(text: str, lang: str = "", context: str = "") -> str:
    """构造 user 消息：有会议上文参考时用分节标记把「参考」与「待译」隔开。

    标记不是装饰：同一条消息里混两段文本时，模型很容易把上文参考当成待译内容
    一起译出来（译文开头冒出上一句）。显式的【上文参考】/【待翻译】抬头把两者
    划开，口音重的句子能稳定复用上文的术语 / 人名，并按上下文纠正识别错字。

    无 context 时与旧版完全一致（``（原文语种：xx）`` + 待译文本），保证不传
    context 的调用方零行为变化。
    """
    src = f"（原文语种：{lang}）" if lang and lang != "zh" else ""
    ctx = (context or "").strip()
    if not ctx:
        return f"{src}\n{text}"
    # 带上文参考时不再叠「原文语种」前缀：参考里的原文本身就是语种线索，
    # 多一行前缀只会把待译文本推远、增加首 token 延迟。
    return f"{CONTEXT_HEADER}\n{ctx}\n{TARGET_HEADER}\n{text}"


MINUTES_PROMPT = """你是资深会议记录助理。根据下面的会议逐字稿，生成会议纪要。
严格输出 JSON，不要输出其他内容，格式：
{"title": "会议主题", "summary": "3-5句概要",
 "key_points": ["讨论要点1"],
 "decisions": ["结论1"],
 "action_items": [{"task": "任务", "owner": "责任人", "due": "截止时间（无则留空）"}],
 "open_questions": ["待跟进问题1"]}

逐字稿：
{transcript}"""


# action_first：JSON 字段与 standard 完全相同，只调「顺序 + 侧重」——先输出行动项，
# 且行动项必须写成可执行动作（谁 + 做什么 + 何时），避免被讨论要点吞掉。
MINUTES_ACTION_FIRST_PROMPT = """你是资深会议记录助理。根据下面的会议逐字稿，生成会议纪要。
严格输出 JSON，不要输出其他内容，格式（注意字段顺序：先输出行动项）：
{"title": "会议主题",
 "action_items": [{"task": "任务", "owner": "责任人", "due": "截止时间（无则留空）"}],
 "summary": "3-5句概要",
 "key_points": ["讨论要点1"],
 "decisions": ["结论1"],
 "open_questions": ["待跟进问题1"]}

要求：
1) 先输出行动项（action_items），逐条写清做什么、谁负责（owner，逐字稿没提到的留空）、
   截止时间（due，没提到的留空）；
2) 行动项只写具体的可执行动作（谁做什么、何时要），不要把讨论要点或结论写进行动项；
3) summary 只概括结论与影响，不重复行动项；
4) 逐字稿没有行动项时输出空数组 []，不要编造。

逐字稿：
{transcript}"""


# bilingual：JSON 结构与 standard 相同（parse_minutes 不感知模板），但每个字段的
# 文本本身是「中文 / English」双语，方便外方参会人直接读纪要。
MINUTES_BILINGUAL_PROMPT = """你是资深会议记录助理。根据下面的会议逐字稿，生成中英双语（中文 / English）的会议纪要。
严格输出 JSON，不要输出其他内容，格式：
{"title": "会议主题 / Meeting topic",
 "summary": "3-5句概要 / 3-5 sentence summary",
 "key_points": ["讨论要点1 / Key point 1"],
 "decisions": ["结论1 / Decision 1"],
 "action_items": [{"task": "任务 / Task", "owner": "责任人", "due": "截止时间（无则留空）"}],
 "open_questions": ["待跟进问题1 / Open question 1"]}

要求：
1) 每个字段的内容都输出双语文本：先中文，再英文（English），用 " / " 分隔（例：
   "已确认下周发货 / Shipment confirmed for next week"）；
2) title 同样双语，标题里的分隔符也用 " / "；
3) action_items 的 task 用双语，owner 与 due 保持原语言（人名、日期不翻译）；
4) 中英文必须一一对应：英文是逐条对应的英文表达，不增删信息、不改写原意；
5) 英文用简洁自然的商务英语，专有名词保留原文写法。

逐字稿：
{transcript}"""


# 模板名 → prompt。未知模板回落 standard（配置层已校验，这里只做防御）。
MINUTES_PROMPTS = {
    "standard": MINUTES_PROMPT,
    "action_first": MINUTES_ACTION_FIRST_PROMPT,
    "bilingual": MINUTES_BILINGUAL_PROMPT,
}


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

    def translate(self, text: str, lang: str = "", context: str = "") -> str:
        """非流式翻译（stop() 收尾补译走这里）。

        ``context``：会议上文参考（前几句的原文/译文），见 :func:`build_user_message`。
        不传则与旧版完全一致。
        """
        messages = [
            {"role": "system", "content": SYSTEM_TRANSLATE},
            {"role": "user", "content": build_user_message(text, lang, context)},
        ]
        return self._chat(messages, max_tokens=512)

    def translate_stream(self, text: str, lang: str = "",
                         context: str = "") -> Iterator[str]:
        """流式翻译：每次 yield **累积**译文（不是增量），供字幕逐字上屏。

        控制器实测（本机真 API，deepseek-v4-pro）：首 token 1.99s、全文 2.28s，
        非流式 4.2s —— 逐字上屏把「等整句」变成「听到一半就在看」。

        与 :meth:`_chat` 一致的两件状态机：
        * ``extra_body={"thinking": {"type": "disabled"}}`` 首次即带（不带则会先流
          144 块 reasoning_content，首字延迟翻倍）；接口不认该参数（400）时降级
          重试一次，并把结果记进 ``_thinking_disabled_ok`` 供后续请求复用。
        * 连接建立失败（create 抛错）沿用既有退避重试；**流中途断开不重试**——
          半句已在屏上，重试会重放前缀，由调用方把译文清空退回原文。

        ``context``（可选）：前几句的原文/译文参考，让模型统一术语/人名并按上下文
        纠正口音识别错字；不传时 prompt 与旧版逐字节一致。

        生成器语义：调用时不发请求（首次 next() 才发）。流正常结束但 content
        全空 → 抛 :class:`TranslationError`。
        """
        messages = [
            {"role": "system", "content": SYSTEM_TRANSLATE},
            {"role": "user", "content": build_user_message(text, lang, context)},
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

    def minutes(self, transcript_md: str, template: str = "standard") -> MinutesResult:
        """生成纪要。``template`` 取 standard / action_first / bilingual（见 .env 的
        MINUTES_TEMPLATE）；未知值回落 standard。三种模板共用同一套 JSON 结构，
        parse_minutes 无需区分。
        """
        # 不能用 str.format：模板内的 JSON 字面量（{"title": …}）会被当成占位符，
        # 直接抛 KeyError: '"title"'，纪要功能全线不可用。用 replace 注入逐字稿。
        prompt = MINUTES_PROMPTS.get((template or "standard").strip().lower(),
                                     MINUTES_PROMPT)
        messages = [
            {"role": "system", "content": "你是资深会议记录助理，输出严格 JSON。"},
            {"role": "user", "content": prompt.replace("{transcript}",
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
