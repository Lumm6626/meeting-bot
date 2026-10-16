# -*- coding: utf-8 -*-
import openai
import pytest

from meeting_bot.translator import (ActionItem, MinutesResult, TranslationError,
                                    Translator, parse_minutes)


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
