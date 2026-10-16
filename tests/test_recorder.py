# -*- coding: utf-8 -*-
import queue
import time

import numpy as np
import pytest

from meeting_bot.recorder import Recorder
from meeting_bot.transcript import Segment


def _settings():
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None)


class FakeCapture:
    """喂入 3 块静音后结束。"""
    def __init__(self):
        self.chunks = [np.zeros(8000, dtype=np.float32)] * 3
        self._i = 0
        self.started = False

    def start(self):
        self.started = True

    def read(self, timeout=1.0):
        if self._i < len(self.chunks):
            c = self.chunks[self._i]
            self._i += 1
            return c
        return None

    def stop(self):
        pass


class FakeASR:
    def transcribe(self, audio, sample_rate=16000):
        return [Segment(0.5, 2.5, "hello world", lang="en")]


class FakeTranslator:
    def translate(self, text, lang=""):
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeDiarizer:
    def label(self, audio, sample_rate=16000):
        return [(0.0, 10.0, "SPEAKER_00")]


class FakeVAD:
    """替换真实 silero：第一次 push 返回一条语句。"""
    def __init__(self):
        self.pushed = 0

    def push(self, chunk):
        self.pushed += 1
        if self.pushed == 1:
            return [(0, np.zeros(16000, dtype=np.float32))]
        return []

    def flush(self):
        return []


def test_full_pipeline(monkeypatch):
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.1)
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASR(),
                 translator=FakeTranslator(), diarizer=FakeDiarizer())
    events = []
    r.on_segment(events.append)
    r.on_status(lambda m: None)
    r.start("测试会议")
    time.sleep(0.6)  # 等 diarize worker 跑一轮
    r.stop()

    segs = r.store.segments()
    assert len(segs) >= 1
    seg = segs[0]
    assert seg.text == "hello world"
    assert seg.lang == "en"
    assert seg.translation == "你好世界"
    assert seg.speaker == "SPEAKER_00"
    assert len(events) >= 1


# ---------------------------------------------------------------------------
# Task 7 审查 N1 转来的前置项：覆盖「capture 结束哨兵 → 主循环退出 → 尾部
# flush」路径，并顺带覆盖 read() 超时抛 queue.Empty 时主循环继续读取的契约。
# ---------------------------------------------------------------------------

class FakeCaptureWithSentinel:
    """首次 read 抛 queue.Empty（模拟超时），随后喂入 3 块音频，最后返回 None。"""
    def __init__(self):
        self.chunks = [np.zeros(16000, dtype=np.float32)] * 3
        self._i = 0
        self._timeout_left = 1
        self.started = False

    def start(self):
        self.started = True

    def read(self, timeout=1.0):
        if self._timeout_left:
            self._timeout_left -= 1
            raise queue.Empty()
        if self._i < len(self.chunks):
            c = self.chunks[self._i]
            self._i += 1
            return c
        return None

    def stop(self):
        pass


class FakeVADFlushOnly:
    """push 只缓存不产出，flush 时一次性吐出每块音频（尾部 flush 路径）。"""
    def __init__(self):
        self._chunks = []

    def push(self, chunk):
        self._chunks.append(chunk)
        return []

    def flush(self):
        out = [(i * 16000, c) for i, c in enumerate(self._chunks)]
        self._chunks = []
        return out


class FakeASRPerChunk:
    """每次 transcribe 调用产出 1 句。"""
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        return [Segment(0.1, 0.9, f"line {self.calls}", lang="en")]


def test_capture_end_flushes_tail_and_translates_all(monkeypatch):
    """capture 结束哨兵 → 主循环退出 → 尾部 flush 的 3 条语句全部入库并补译。"""
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADFlushOnly)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.1)
    asr = FakeASRPerChunk()
    r = Recorder(_settings(), capture=FakeCaptureWithSentinel(), asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)
    r.start("尾部 flush 会议")
    r.stop()

    assert asr.calls == 3
    segs = r.store.segments()
    assert len(segs) == 3
    assert [s.text for s in segs] == ["line 1", "line 2", "line 3"]
    assert all(s.translation == "你好世界" for s in segs)
