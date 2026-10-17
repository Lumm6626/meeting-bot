# -*- coding: utf-8 -*-
import openai
import pytest

from meeting_bot.translator import (ActionItem, MinutesResult, TranslationError,
                                    Translator, parse_minutes, session_headers)


def _settings():
    from meeting_bot.config import Settings
    return Settings(
        opencode_base_url="https://opencode.ai/zen/go/v1",
        opencode_api_key="sk-test",
        translate_model="deepseek-v4-flash",
        minutes_model="deepseek-v4-pro",
        whisper_model="medium",
        hf_token="",
        output_dir=None,
        root=None,
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


def test_translate_returns_text(monkeypatch):
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        return FakeResponse("  大家好，欢迎参加今天的会议。  ")

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    assert t.translate("Hello everyone.", "en") == "大家好，欢迎参加今天的会议。"
    assert calls[0]["max_tokens"] == 512


def test_translate_retries_then_raises(monkeypatch):
    def fake_create(self, **kwargs):
        raise openai.APIError("boom", request=None, body=None)

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    with pytest.raises(TranslationError):
        t.translate("hello", "en")


def test_thinking_disable_falls_back_on_400(monkeypatch):
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        if "extra_body" in kwargs:
            raise openai.APIError("unsupported", request=None, body=None)
        return FakeResponse("ok")

    # APIError 没有 status_code 属性时按 400 处理走降级路径
    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    assert t.translate("hi", "en") == "ok"
    assert len(calls) == 2
    assert "extra_body" not in calls[1]


# ---------------------------------------------------------------------------
# 流式翻译（translate_stream）
# ---------------------------------------------------------------------------

class FakeDelta:
    """流式 delta：content 与 reasoning_content 是两条独立通道。"""

    def __init__(self, content=None, reasoning=None):
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


class FakeStreamChoice:
    def __init__(self, delta):
        self.delta = delta


class FakeStreamChunk:
    def __init__(self, delta):
        self.choices = [FakeStreamChoice(delta)]


def _content_chunk(text):
    return FakeStreamChunk(FakeDelta(content=text))


def _reasoning_chunk(text):
    """thinking 未禁用时接口先流的 reasoning 块（实测 144 块）。"""
    return FakeStreamChunk(FakeDelta(content=None, reasoning=text))


def test_translate_stream_accumulates(monkeypatch):
    """每次 yield 的是**累积**译文（不是增量）→ 面板可以拿它原地替换同一句。"""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        return [_content_chunk("沙特"), _content_chunk("阿拉伯"), _content_chunk("市场")]

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    assert list(t.translate_stream("Saudi Arabia market", "en")) == [
        "沙特", "沙特阿拉伯", "沙特阿拉伯市场"]
    # 流式请求必须显式 stream=True，且首次就带 thinking 禁用（否则白等 144 块推理）
    assert calls[0]["stream"] is True
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert calls[0]["max_tokens"] == 512


def test_translate_stream_skips_reasoning(monkeypatch):
    """delta.reasoning_content 一律跳过：思考过程绝不能混进译文。"""

    def fake_create(self, **kwargs):
        return [_reasoning_chunk("让我先想想"),
                _reasoning_chunk("地名要保持原文"),
                _content_chunk("沙特"),
                _content_chunk("市场")]

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    parts = list(t.translate_stream("Saudi market", "en"))
    assert parts == ["沙特", "沙特市场"]          # 只有 content 触发了 yield
    assert "想想" not in "".join(parts)
    assert "地名" not in "".join(parts)


def test_translate_stream_empty_raises(monkeypatch):
    """流正常结束但 content 全空（只有 reasoning / 空串）→ TranslationError。"""

    def fake_create(self, **kwargs):
        return [_reasoning_chunk("只有思考没有答案"), _content_chunk(""), _content_chunk(None)]

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    with pytest.raises(TranslationError) as excinfo:
        list(t.translate_stream("hello", "en"))
    assert "空内容" in str(excinfo.value)


def test_translate_stream_connection_error_retries(monkeypatch):
    """create 调用抛 APIError（接口不认 thinking）→ 降级重试一次并成功。"""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        if "extra_body" in kwargs:
            raise openai.APIError("unsupported parameter", request=None, body=None)
        return [_content_chunk("沙特"), _content_chunk("市场")]

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    assert list(t.translate_stream("Saudi market", "en")) == ["沙特", "沙特市场"]
    assert len(calls) == 2
    assert "extra_body" not in calls[1]          # 第二次降级为普通请求
    assert calls[1]["stream"] is True
    assert t._thinking_disabled_ok is False      # 状态位复用：后续请求不再带 thinking


def test_translate_stream_midstream_error_no_retry(monkeypatch):
    """流中途断开不重试（半句已上屏，重试会重放前缀）→ 抛 TranslationError。"""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)

        def _broken_stream():
            yield _content_chunk("沙特")
            raise openai.APIError("connection reset", request=None, body=None)

        return _broken_stream()

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = False              # 跳过降级路径，只看流中断语义
    got = []
    with pytest.raises(TranslationError) as excinfo:
        for part in t.translate_stream("Saudi market", "en"):
            got.append(part)
    assert got == ["沙特"]                        # 已产出的部分由调用方决定去留
    assert len(calls) == 1                        # 未重试
    assert "流中断" in str(excinfo.value)


def test_translate_stream_with_context(monkeypatch):
    """带会议上下文：user 消息用【上文参考】/【待翻译】分节，且上下文原样可见。

    分节标记是两个需求的落点：
    * 上文参考给模型做术语 / 人名统一 + 口音错字的上下文纠错；
    * 【待翻译】把待译文本明确隔开，避免模型把上文一起翻出来。
    """
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return [_content_chunk("沙特"), _content_chunk("市场")]
        return FakeResponse("沙特市场")

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    ctx = "原文：We met in Riad last week\n译文：上周我们在利雅得见过"

    assert list(t.translate_stream("Saudi market", "en", context=ctx)) == ["沙特", "沙特市场"]
    user = calls[0]["messages"][1]["content"]
    assert "【上文参考" in user
    assert "【待翻译】" in user
    assert ctx in user                       # 上下文原样传入（不截断/不改写）
    assert user.endswith("Saudi market")      # 待译文本在最后，紧贴【待翻译】
    # 系统提示词第 9 条：统一术语/人名 + 按上下文纠口音错字
    system = calls[0]["messages"][0]["content"]
    assert "口音" in system and "术语" in system

    # 非流式同一条构造路径（stop() 补译也用 context）
    assert t.translate("Saudi market", "en", context=ctx) == "沙特市场"
    plain_user = calls[1]["messages"][1]["content"]
    assert ctx in plain_user and "【待翻译】" in plain_user

    # 不传 context：与旧版完全一致（不带分节标记，也不留空壳标题）
    t.translate("Saudi market", "en")
    no_ctx = calls[2]["messages"][1]["content"]
    assert "【上文参考" not in no_ctx and "【待翻译】" not in no_ctx
    assert no_ctx.endswith("Saudi market")


def test_system_translate_prompt_covers_meeting_context(monkeypatch):
    """提示词升级：会议语境 + 识别错字/医学术语/数字/品牌名四条要求必须在系统提示里。"""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        return FakeResponse("ok")

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t.translate("hello", "en")
    system = calls[0]["messages"][0]["content"]
    assert "只输出译文" in system                 # 核心要求不得退化
    for keyword in ("医疗", "识别错字", "医学术语", "数字、比例、金额、日期", "品牌名"):
        assert keyword in system, keyword


def test_parse_minutes_valid_json():
    raw = ('{"title": "周会", "summary": "讨论了产品发布", "key_points": ["定价"],'
    '"decisions": ["下月发布"], "action_items": [{"task": "写方案", "owner": "张三", "due": "10/20"}],'
    '"open_questions": ["渠道确认"]}')
    m = parse_minutes(raw)
    assert isinstance(m, MinutesResult)
    assert m.title == "周会"
    assert m.action_items[0] == ActionItem(task="写方案", owner="张三", due="10/20")


def test_parse_minutes_falls_back_on_bad_json():
    m = parse_minutes("这不是 JSON，是纯文本纪要。")
    assert m.title == "会议纪要"
    assert m.summary == "这不是 JSON，是纯文本纪要。"


def test_translate_empty_choices_raises_translation_error(monkeypatch):
    """响应结构异常（choices 为空）必须降级为 TranslationError，不泄漏 IndexError。"""

    class EmptyChoicesResponse:
        choices = []

    def fake_create(self, **kwargs):
        return EmptyChoicesResponse()

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    with pytest.raises(TranslationError) as excinfo:
        t.translate("hello", "en")
    # 保留 from e 因果链
    assert isinstance(excinfo.value.__cause__, IndexError)


def test_translate_none_choices_raises_translation_error(monkeypatch):
    """choices=None 必须降级为 TranslationError，不得泄漏 TypeError（Critical-1 的一半）。

    接口字段缺失时 ``resp.choices[0]`` 抛
    TypeError("'NoneType' object is not subscriptable")。修复前 translator 的异常包裹
    元组不含 TypeError → 异常直穿 stop() 的 ``except TranslationError`` →
    export_all 整段被跳过（整场会议零产物）。本用例锁死该包裹。
    """

    class NoneChoicesResponse:
        choices = None

    def fake_create(self, **kwargs):
        return NoneChoicesResponse()

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    with pytest.raises(TranslationError) as excinfo:
        t.translate("hello", "en")
    assert isinstance(excinfo.value.__cause__, TypeError)   # 保留 from e 因果链


def test_translator_sets_opencode_session_header(monkeypatch):
    """OpenCode Go 缺 x-opencode-session 头会直接 400（MissingSessionID），客户端必须带上。"""
    monkeypatch.delenv("OPENCODE_SESSION_ID", raising=False)
    t = Translator(_settings())
    headers = {k.lower(): v for k, v in t._client.default_headers.items()}
    assert headers.get("x-opencode-session")


def test_session_id_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_SESSION_ID", "fixed-session-123")
    assert session_headers() == {"x-opencode-session": "fixed-session-123"}


def test_session_headers_ignores_blank_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_SESSION_ID", "   ")
    assert session_headers()["x-opencode-session"].startswith("meeting-bot-")


def test_minutes_embeds_transcript_without_format_crash(monkeypatch):
    """回归：模板含 JSON 花括号，用 str.format 会抛 KeyError('"title"')，纪要曾全线不可用。"""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        return FakeResponse('{"title": "周会", "summary": "讨论了定价", "key_points": [],'
                            '"decisions": [], "action_items": [], "open_questions": []}')

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    m = t.minutes("# 会议逐字稿：周会\n原文：hello\n译文：你好")
    assert m.title == "周会"
    prompt = calls[0]["messages"][1]["content"]
    assert "{transcript}" not in prompt
    assert "原文：hello" in prompt
    assert '"key_points"' in prompt      # JSON 示例模板原样保留
    assert calls[0]["model"] == "deepseek-v4-pro"


def test_minutes_truncates_long_transcript(monkeypatch):
    """超长逐字稿只取尾部 18 万字符，避免请求体过大。"""
    calls = []

    def fake_create(self, **kwargs):
        calls.append(kwargs)
        return FakeResponse('{"title": "t", "summary": "s"}')

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    t = Translator(_settings())
    t.minutes("A" * 200000 + "TAIL")
    prompt = calls[0]["messages"][1]["content"]
    assert len(prompt) < 200000
    assert prompt.endswith("TAIL")
