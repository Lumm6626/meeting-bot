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
