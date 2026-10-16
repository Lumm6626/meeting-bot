# -*- coding: utf-8 -*-
"""Task 9 录制编排集成测试：全部注入 fake，不加载任何重型依赖。"""
import queue
import threading
import time

import numpy as np

from meeting_bot.recorder import Recorder
from meeting_bot.transcript import Segment
from meeting_bot.translator import TranslationError


def _settings():
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None)


# ---------------------------------------------------------------------------
# fake 组件
# ---------------------------------------------------------------------------

class FakeCapture:
    """喂入 n 块静音后返回结束哨兵（chunks=1 用于单句场景）。"""

    def __init__(self, chunks: int = 3):
        self.chunks = [np.zeros(8000, dtype=np.float32)] * chunks
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


class FakeASR:
    def transcribe(self, audio, sample_rate=16000):
        return [Segment(0.5, 2.5, "hello world", lang="en")]


class FakeASRZh:
    def transcribe(self, audio, sample_rate=16000):
        return [Segment(0.5, 2.5, "你好世界", lang="zh")]


class FakeASRPerChunk:
    """每次 transcribe 调用产出 1 句。"""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        return [Segment(0.1, 0.9, f"line {self.calls}", lang="en")]


class FakeASRSlowTail:
    """前 2 次快返回；第 3 次（尾部语句）慢 tail_delay 秒才返回。"""

    def __init__(self, tail_delay: float = 0.3):
        self.calls = 0
        self.tail_delay = tail_delay
        self.third_started = threading.Event()

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        if self.calls < 3:
            return [Segment(0.1, 0.9, f"head {self.calls}", lang="en")]
        self.third_started.set()
        time.sleep(self.tail_delay)
        return [Segment(0.1, 0.9, "tail 3", lang="en")]


class FakeTranslator:
    def __init__(self):
        self.calls = 0

    def translate(self, text, lang=""):
        self.calls += 1
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeTranslatorFailOnce:
    """第一次 translate 抛 TranslationError，之后成功（覆盖 stop() 补译路径）。"""

    def __init__(self):
        self.calls = 0

    def translate(self, text, lang=""):
        self.calls += 1
        if self.calls == 1:
            raise TranslationError("模拟翻译接口失败")
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeDiarizer:
    def label(self, audio, sample_rate=16000):
        return [(0.0, 10.0, "SPEAKER_00")]


class FakeVAD:
    """替换真实 silero：第一次 push 返回一条偏移 16000 样本（1 秒）的语句。"""

    def __init__(self):
        self.pushed = 0

    def push(self, chunk):
        self.pushed += 1
        if self.pushed == 1:
            return [(16000, np.zeros(16000, dtype=np.float32))]
        return []

    def flush(self):
        return []


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


class FakeVADPerChunk:
    """每块音频产出一条语句（多句流水线场景）。"""

    def __init__(self):
        self._n = 0

    def push(self, chunk):
        offset = self._n * 16000
        self._n += 1
        return [(offset, np.zeros(16000, dtype=np.float32))]

    def flush(self):
        return []


# ---------------------------------------------------------------------------
# 基础流水线
# ---------------------------------------------------------------------------

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
    # 时间戳：VAD 偏移 16000 样本（1s）+ ASR 段 0.5~2.5 → 1.5 ~ 3.5
    assert seg.start == 1.5
    assert seg.end == 3.5


def test_capture_end_flushes_tail_and_translates_all(monkeypatch):
    """capture 结束哨兵 → 主循环退出 → 尾部 flush 的 3 条语句全部入库并补译。

    覆盖 Task 7 审查 N1 转来的前置项，并顺带覆盖 read() 超时抛
    queue.Empty 时主循环继续读取的契约。
    """
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


# ---------------------------------------------------------------------------
# 修复轮 1：producer-done 模式，防尾部语句丢失
# ---------------------------------------------------------------------------

def test_stop_waits_for_tail_utterance(monkeypatch):
    """stop() 时主循环仍卡在尾段 ASR（0.3s）：尾部语句必须照样翻译并 emit。

    旧实现 worker 循环用 _stop 判定，队列一空即退出；
    stop() 置位 _stop 后 worker 退出、尾段稍后才入队 → 永不翻译、永不 emit。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.1)
    asr = FakeASRSlowTail(tail_delay=0.3)
    r = Recorder(_settings(), capture=FakeCapture(), asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    events = []
    r.on_segment(events.append)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("尾部会议",), daemon=True)
    thread.start()
    try:
        assert asr.third_started.wait(timeout=5), "第 3 次 transcribe（尾段）未开始"
        time.sleep(0.15)  # 前两句已翻译 emit，此刻翻译队列为空
        r.stop()
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()

    segs = r.store.segments()
    assert [s.text for s in segs] == ["head 1", "head 2", "tail 3"]
    assert all(s.translation == "你好世界" for s in segs)
    assert [s.text for s in events] == ["head 1", "head 2", "tail 3"]


# ---------------------------------------------------------------------------
# 修复轮 1：补 spec 3/4/7 测试覆盖
# ---------------------------------------------------------------------------

def test_lazy_asr_engine_created_once(monkeypatch):
    """spec 3：未注入 asr 时惰性构造 ASREngine，且只构造一次。"""
    created = []

    class _RecordingASREngine:
        def __init__(self, settings):
            created.append(settings)

        def transcribe(self, audio, sample_rate=16000):
            return [Segment(0.5, 2.5, "lazy ok", lang="en")]

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _RecordingASREngine)
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    r = Recorder(_settings(), capture=FakeCapture(), translator=FakeTranslator(),
                 diarizer=None)
    r.on_status(lambda m: None)
    r.start("惰性 ASR 会议")
    r.stop()

    assert len(created) == 1
    segs = r.store.segments()
    assert len(segs) == 3
    assert all(s.text == "lazy ok" and s.translation == "你好世界" for s in segs)


def test_translation_error_segment_is_retried_on_stop(monkeypatch):
    """spec 4：TranslationError 的句子在 stop() 时统一补译，且不抛异常。"""
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    tr = FakeTranslatorFailOnce()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    statuses = []
    r.on_status(statuses.append)
    r.start("补译会议")
    r.stop()   # 补译失败句不得向外抛异常

    segs = r.store.segments()
    assert len(segs) == 1
    assert segs[0].translation == "你好世界"
    assert tr.calls == 2                      # 1 次失败 + stop() 补译成功
    assert any("翻译失败" in m for m in statuses)


def test_zh_segment_skips_translation(monkeypatch):
    """spec 7：lang == 'zh' 的句子不调翻译接口，译文等于原文。"""
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    tr = FakeTranslator()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASRZh(),
                 translator=tr, diarizer=None)
    events = []
    r.on_segment(events.append)
    r.on_status(lambda m: None)
    r.start("中文会议")
    r.stop()

    segs = r.store.segments()
    assert len(segs) == 1
    assert segs[0].lang == "zh"
    assert segs[0].translation == segs[0].text == "你好世界"
    assert tr.calls == 0
    assert len(events) == 1
