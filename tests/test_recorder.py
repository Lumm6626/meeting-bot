# -*- coding: utf-8 -*-
"""Task 9 录制编排集成测试：全部注入 fake，不加载任何重型依赖。"""
import queue
import re
import threading
import time

import numpy as np

from meeting_bot.recorder import Recorder, assign_speakers, map_local_labels
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


# ---------------------------------------------------------------------------
# 修复轮 2：diarize 滑动窗口 + 重叠区标签映射（消除 O(T²) 全量重算）
# ---------------------------------------------------------------------------

class FakeCaptureOneLongChunk:
    """单块长音频（默认 2 秒 = 32000 样本）后结束。

    只有一块音频 → 有效 diarize 只可能跑一次，窗口切片与 base 都能确定性断言。
    """

    def __init__(self, samples: int = 32000):
        self.chunk = np.zeros(samples, dtype=np.float32)
        self._done = False

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._done:
            return None
        self._done = True
        return self.chunk

    def stop(self):
        pass


class FakeCaptureStaged:
    """分阶段喂音频：第 i 块要等 gates[i] 置位才返回，否则按超时抛 queue.Empty。"""

    def __init__(self, chunks: int, chunk_samples: int = 16000):
        self.chunks = [np.zeros(chunk_samples, dtype=np.float32) for _ in range(chunks)]
        self.gates = [threading.Event() for _ in range(chunks)]
        self.gates[0].set()          # 第一块立即可读
        if chunks > 1:
            self.gates[1].set()      # 第一轮 diarize 用「前两块」跑
        self._i = 0

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._i >= len(self.chunks):
            return None
        if not self.gates[self._i].wait(timeout=timeout):
            raise queue.Empty()      # 未到放行阶段：模拟采集超时
        chunk = self.chunks[self._i]
        self._i += 1
        return chunk

    def stop(self):
        pass

    def release(self, i: int):
        self.gates[i].set()


class FakeVADTwoUtterances:
    """一次 push 吐出两条语句（偏移 0 与 16000 样本）→ 单块音频内首尾各一句。"""

    def __init__(self):
        self.pushed = False

    def push(self, chunk):
        if self.pushed:
            return []
        self.pushed = True
        return [(0, np.zeros(16000, dtype=np.float32)),
                (16000, np.zeros(16000, dtype=np.float32))]

    def flush(self):
        return []


class FakeDiarizerWindow:
    """返回覆盖整窗的固定 span（局部标签可指定），记录每次收到的窗口样本数。"""

    def __init__(self, local_label: str = "SPK_A"):
        self.local_label = local_label
        self.calls = 0
        self.window_lens = []

    def label(self, audio, sample_rate=16000):
        self.calls += 1
        self.window_lens.append(len(audio))
        return [(0.0, len(audio) / sample_rate, self.local_label)]


class FakeDiarizerSequence:
    """窗口未填满返回空；每次「有效」调用按顺序返回 labels 里的下一个局部标签。"""

    def __init__(self, labels, full_window_samples: int):
        self.labels = list(labels)
        self.full_window_samples = full_window_samples
        self.calls = 0
        self.window_lens = []
        self._returned = 0

    def label(self, audio, sample_rate=16000):
        self.calls += 1
        self.window_lens.append(len(audio))
        if len(audio) < self.full_window_samples:
            return []                       # 窗口还没填满：结果不可靠，空手而归
        label = self.labels[min(self._returned, len(self.labels) - 1)]
        self._returned += 1
        return [(0.0, len(audio) / sample_rate, label)]


def _wait_until(pred, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_map_local_labels():
    """纯函数：重叠投票（正常映射 / 平局取先出现 / 新说话人无映射）。"""
    # 场景 1：按重叠时长取最大者；overlap_start 之前的旧标注不参与投票
    prev = [(200.0, 290.0, "SPEAKER_07"),      # 整条在 overlap_start 之前
            (300.0, 310.0, "SPEAKER_00"),
            (310.0, 320.0, "SPEAKER_01")]
    new = [(200.0, 260.0, "LOCAL_C"),
           (300.0, 308.0, "LOCAL_A"),          # 与 SPEAKER_00 重叠 8s → SPEAKER_00
           (308.0, 320.0, "LOCAL_B")]          # SPEAKER_00 2s vs SPEAKER_01 10s → SPEAKER_01
    got = map_local_labels(prev, new, 300.0)
    assert got == {"LOCAL_A": "SPEAKER_00", "LOCAL_B": "SPEAKER_01"}
    assert "LOCAL_C" not in got

    # 场景 2：与两个全局标签重叠时长相等 → 平局取「先出现者」（prev_labeled 顺序）
    new_tie = [(0.0, 10.0, "LOCAL_X")]
    assert map_local_labels([(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")],
                            new_tie, 0.0) == {"LOCAL_X": "SPEAKER_00"}
    assert map_local_labels([(5.0, 10.0, "SPEAKER_01"), (0.0, 5.0, "SPEAKER_00")],
                            new_tie, 0.0) == {"LOCAL_X": "SPEAKER_01"}

    # 场景 3：全新说话人（无任何重叠）→ 不出现在映射里，由调用方分配新标签
    assert map_local_labels([(0.0, 5.0, "SPEAKER_00")],
                            [(20.0, 25.0, "LOCAL_NEW")], 0.0) == {}


def test_diarize_windowed(monkeypatch):
    """滑动窗口：只把最近 DIARIZE_WINDOW 秒喂给 pyannote，窗口内语句拿到全局 SPEAKER_xx。

    单块 2 秒音频 + DIARIZE_WINDOW=1.0s（16000 样本）→ 窗口 = 末尾 16000 样本
    （base=1.0s）：只有落在窗口内的第 2 句能拿到标签。若实现漏加 base 偏移，
    拿到标签的会是窗口外的第 1 句 —— 该断言同时锁死 base 正确性。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADTwoUtterances)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.05)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_WINDOW", 1.0)     # 16000 样本

    seen = []
    real_assign = assign_speakers

    def _spy(segments, spans, min_overlap=0.3):
        seen.append([tuple(s) for s in spans])
        return real_assign(segments, spans, min_overlap)

    monkeypatch.setattr("meeting_bot.recorder.assign_speakers", _spy)

    diarizer = FakeDiarizerWindow(local_label="SPK_A")
    r = Recorder(_settings(), capture=FakeCaptureOneLongChunk(32000), asr=FakeASRPerChunk(),
                 translator=FakeTranslator(), diarizer=diarizer)
    r.on_status(lambda m: None)
    r.start("窗口会议")     # capture 只有一块，主循环很快结束
    assert _wait_until(lambda: any(s.speaker for s in r.store.segments())), "diarize 未产出标签"
    r.stop()

    # 1) 只喂窗口：pyannote 每次拿到的音频都不超过 DIARIZE_WINDOW（不再全量重算）
    assert diarizer.window_lens == [16000]
    # 2) base 偏移 + 标签替换：传给 assign_speakers 的是绝对时间轴上的全局标签
    assert seen == [[(1.0, 2.0, "SPEAKER_00")]]
    # 3) 窗口内语句拿到 SPEAKER_00（局部 SPK_A 未映射 → 新分配）；窗口外语句不标注
    segs = r.store.segments()
    assert len(segs) == 2
    assert segs[0].speaker == ""              # [0.1, 0.9] 在窗口外
    assert segs[1].speaker == "SPEAKER_00"    # [1.1, 1.9] 在窗口 [1.0, 2.0] 内
    assert re.fullmatch(r"SPEAKER_\d{2}", segs[1].speaker)
    # 4) stop() 收尾：标注线程已被 join
    assert not r._diarize_thread.is_alive()


def test_diarize_maps_local_labels_across_windows(monkeypatch):
    """跨窗口标签连续性：第二个窗口的新局部标签经重叠区投票映射回同一全局标签。

    第一轮：前两块音频（窗口=32000 样本，base=0）→ 局部 SPK_A 首次出现 → SPEAKER_00；
    第二轮：放行第三块后窗口 = 最近 32000 样本（base=1.0，局部 SPK_B），
    重叠区里的第 2 句 [1.1, 1.9] 把 SPK_B 投票映射回 SPEAKER_00；
    若映射缺失，SPK_B 会被当成新说话人 → 冒出 SPEAKER_01。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.05)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_WINDOW", 2.0)     # 32000 样本

    diarizer = FakeDiarizerSequence(["SPK_A", "SPK_B"], full_window_samples=32000)
    capture = FakeCaptureStaged(chunks=3, chunk_samples=16000)
    r = Recorder(_settings(), capture=capture, asr=FakeASRPerChunk(),
                 translator=FakeTranslator(), diarizer=diarizer)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("跨窗口会议",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: len(r.store.segments()) == 2
                           and r.store.segments()[1].speaker == "SPEAKER_00"), "第一轮未标注"
        capture.release(2)                    # 放行第三块 → 触发第二轮滑窗
        assert _wait_until(lambda: len(r.store.segments()) == 3
                           and bool(r.store.segments()[2].speaker)), "第二轮未标注"
    finally:
        r.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()

    segs = r.store.segments()
    assert [s.speaker for s in segs] == ["SPEAKER_00"] * 3
    assert [sp.label for sp in r.store.speakers()] == ["SPEAKER_00"]   # 没多出 SPEAKER_01
    assert max(diarizer.window_lens) <= 32000                          # 从不喂全量音频
