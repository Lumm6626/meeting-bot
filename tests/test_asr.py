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
