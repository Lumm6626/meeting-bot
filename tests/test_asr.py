# -*- coding: utf-8 -*-
import numpy as np

from meeting_bot.asr import ASREngine


def _settings():
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None)


class FakeSeg:
    start, end, text = 0.0, 1.5, " Hello world. "


class FakeTranscribe:
    def __init__(self):
        self.segments = [FakeSeg()]
        self.language = "en"


class FakeModel:
    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, **kwargs):
        assert kwargs["language"] is None, "必须开启自动语种检测"
        return [FakeSeg()], type("Info", (), {"language": "en"})()


def test_transcribe_maps_segments(monkeypatch):
    monkeypatch.setattr("meeting_bot.asr.WhisperModel", FakeModel)
    engine = ASREngine(_settings())
    segs = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert len(segs) == 1
    assert segs[0].text == "Hello world."
    assert segs[0].lang == "en"


# ---------------------------------------------------------------------------
# 云端识别（Groq whisper-large-v3-turbo）：全部 mock，不真调接口
# ---------------------------------------------------------------------------
import io
import types
import wave

import httpx
import openai
import pytest

from meeting_bot.asr import CloudASREngine, CloudASRError
from meeting_bot.config import ConfigError


def _cloud_settings(**overrides):
    """云端模式 Settings（关键字构造：新增字段带默认值，不影响既有位置参数用法）。"""
    from meeting_bot.config import Settings
    base = dict(opencode_base_url="https://x", opencode_api_key="sk-test",
                translate_model="m1", minutes_model="m2", whisper_model="medium",
                hf_token="", output_dir=None, root=None,
                groq_api_key="gsk_test", asr_provider="groq")
    base.update(overrides)
    return Settings(**base)


class FakeTranscriptions:
    """假 audio.transcriptions 资源：记录调用参数，按预设返回或抛错。"""

    def __init__(self, result=None, error: BaseException | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _install_fake_openai(monkeypatch, transcriptions: FakeTranscriptions) -> list[dict]:
    """把 openai.OpenAI 换成假 client，返回构造参数列表（校验 base_url/timeout 等）。"""
    created: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.audio = types.SimpleNamespace(transcriptions=transcriptions)

    monkeypatch.setattr("meeting_bot.asr.openai.OpenAI", FakeOpenAI)
    return created


def _verbose_json(language="en", segments=None):
    return types.SimpleNamespace(language=language, segments=segments)


def _api_error() -> openai.APIError:
    return openai.APIError("groq down", httpx.Request("POST", "https://api.groq.com"),
                           body=None)


def test_cloud_engine_requires_key(monkeypatch):
    """a. 没有 GROQ_API_KEY（环境变量与 settings 都空）→ ConfigError，不得构造 client。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("meeting_bot.asr.openai.OpenAI",
                        lambda **kw: pytest.fail("缺 key 时不得创建 client"))
    with pytest.raises(ConfigError):
        CloudASREngine(_cloud_settings(groq_api_key=""))


def test_cloud_engine_reads_env_key(monkeypatch):
    """key 可来自环境变量 GROQ_API_KEY（settings 里为空也行）。"""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    created = _install_fake_openai(monkeypatch, FakeTranscriptions(result=_verbose_json()))
    CloudASREngine(_cloud_settings(groq_api_key=""))
    assert created[0]["api_key"] == "gsk_from_env"
    assert created[0]["base_url"] == "https://api.groq.com/openai/v1"
    assert created[0]["max_retries"] == 0
    assert created[0]["timeout"] == 30.0


def test_cloud_transcribe_maps_segments(monkeypatch):
    """b. verbose_json → Segment(start/end/text/lang)，空文本过滤。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tr = FakeTranscriptions(result=_verbose_json("en", [
        # dict 与 attr 两种返回形态都要吃得下（fake client 可能给任一种）
        {"start": 0.0, "end": 1.5, "text": "  Hello world.  "},
        types.SimpleNamespace(start=1.5, end=3.25, text="Second line"),
        {"start": 3.25, "end": 3.5, "text": "   "},          # 空文本 → 过滤
    ]))
    _install_fake_openai(monkeypatch, tr)
    engine = CloudASREngine(_cloud_settings())

    segs = engine.transcribe(np.zeros(32000, dtype=np.float32))

    assert [(s.start, s.end, s.text, s.lang) for s in segs] == [
        (0.0, 1.5, "Hello world.", "en"),
        (1.5, 3.25, "Second line", "en"),
    ]
    assert tr.calls[0]["model"] == "whisper-large-v3-turbo"
    assert tr.calls[0]["response_format"] == "verbose_json"


def test_cloud_transcribe_retries_then_raises(monkeypatch):
    """c1. APIError 连续 3 次（首次 + 重试 2 次）→ CloudASRError。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("meeting_bot.asr.time.sleep", lambda s: None)   # 不真等退避
    tr = FakeTranscriptions(error=_api_error())
    _install_fake_openai(monkeypatch, tr)
    engine = CloudASREngine(_cloud_settings())

    with pytest.raises(CloudASRError) as ei:
        engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert len(tr.calls) == 3
    assert "云端识别不可用" in str(ei.value)


def test_cloud_transcribe_retries_then_succeeds(monkeypatch):
    """重试机制真的有用：前 2 次失败、第 3 次成功 → 正常返回。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("meeting_bot.asr.time.sleep", lambda s: None)
    tr = FakeTranscriptions(result=_verbose_json("en", [{"start": 0, "end": 1, "text": "ok"}]))
    calls = {"n": 0}
    original = tr.create

    def flaky_create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            tr.calls.append(kwargs)
            raise _api_error()
        return original(**kwargs)

    tr.create = flaky_create
    _install_fake_openai(monkeypatch, tr)
    segs = CloudASREngine(_cloud_settings()).transcribe(np.zeros(16000, dtype=np.float32))
    assert [s.text for s in segs] == ["ok"]
    assert calls["n"] == 3


@pytest.mark.parametrize("bad", [
    types.SimpleNamespace(language="en"),                       # 缺 segments 字段
    _verbose_json("en", None),                                  # segments=None
    _verbose_json("en", []),                                    # 空 segments
    _verbose_json("en", [{"start": "oops", "end": 1, "text": "x"}]),   # 字段类型非法
])
def test_cloud_transcribe_wraps_bad_structure(monkeypatch, bad):
    """c2. 结构异常（缺字段/空 segments/类型非法）也被包装成 CloudASRError。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tr = FakeTranscriptions(result=bad)
    _install_fake_openai(monkeypatch, tr)
    with pytest.raises(CloudASRError):
        CloudASREngine(_cloud_settings()).transcribe(np.zeros(16000, dtype=np.float32))


def test_cloud_writes_16k_wav(monkeypatch):
    """d. 上传的 file 是 16k 单声道 16bit WAV（wave 解析上传内容）。"""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tr = FakeTranscriptions(result=_verbose_json("en", [{"start": 0, "end": 1, "text": "x"}]))
    _install_fake_openai(monkeypatch, tr)
    CloudASREngine(_cloud_settings()).transcribe(np.zeros(3200, dtype=np.float32))

    name, data, mime = tr.calls[0]["file"]
    assert mime == "audio/wav"
    assert name.endswith(".wav")
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 3200


def test_encode_wav_clips_out_of_range():
    """超出 [-1,1] 的浮点样本必须削顶（int16 溢出会变成刺耳噪声/负值回绕）。"""
    from meeting_bot.asr import encode_wav
    data = encode_wav(np.array([2.0, -2.0, 0.0], dtype=np.float32))
    with wave.open(io.BytesIO(data), "rb") as wf:
        frames = np.frombuffer(wf.readframes(3), dtype="<i2")
    assert frames.tolist() == [32767, -32767, 0]


def test_cloud_transcribe_real_sdk_shapes(monkeypatch):
    """真实 openai SDK + MockTransport（不联网）：验证请求能被编成 multipart、
    响应 verbose_json 解析出的对象形状与工程里 _field 的取值方式一致。

    这是「不装 key 也能验证 SDK 用法没写错」的护栏：file 三元组形状、response_format、
    端点路径、segments 属性名任何一处写错，本用例都会失败（真调接口才会暴露的错，
    在没有 Groq key 的环境里由它兜住）。
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.read()
        return httpx.Response(200, json={
            "task": "transcribe", "language": "en", "duration": 1.25, "text": " Hello ",
            "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": 1.25, "text": " Hello ",
                          "tokens": [1], "temperature": 0.0, "avg_logprob": -0.1,
                          "compression_ratio": 1.0, "no_speech_prob": 0.01}],
        })

    real_openai = openai.OpenAI
    monkeypatch.setattr("meeting_bot.asr.openai.OpenAI",
                        lambda **kw: real_openai(http_client=httpx.Client(
                            transport=httpx.MockTransport(handler)), **kw))

    segs = CloudASREngine(_cloud_settings()).transcribe(np.zeros(16000, dtype=np.float32))

    assert captured["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert "multipart/form-data" in captured["content_type"]
    body = captured["body"]
    assert b"verbose_json" in body and b"whisper-large-v3-turbo" in body
    assert b"RIFF" in body                     # 上传的确实是 wav
    assert [(s.start, s.end, s.text, s.lang) for s in segs] == [(0.0, 1.25, "Hello", "en")]
