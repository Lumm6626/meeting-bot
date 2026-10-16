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


# ---------------------------------------------------------------------------
# 修复轮 3：幽灵说话人标签 + 翻译 worker 异常免疫
# ---------------------------------------------------------------------------

class FakeTranslatorUnexpectedErrorOnce:
    """第 1 次 translate 抛非 TranslationError 的意外异常（模拟 openai 库边界异常）。"""

    def __init__(self):
        self.calls = 0

    def translate(self, text, lang=""):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("模拟翻译接口意外异常")
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeDiarizerNoOverlap:
    """返回与所有语句零重叠的 span（语句都在 0~3s，span 在 10~12s）。"""

    def __init__(self):
        self.calls = 0

    def label(self, audio, sample_rate=16000):
        self.calls += 1
        return [(10.0, 12.0, "SPK_GHOST")]


def test_worker_survives_unexpected_error(monkeypatch):
    """N2 回归：翻译接口抛非 TranslationError 的意外异常时，worker 不得猝死。

    旧实现只捕 TranslationError：第 1 段 translate 抛 RuntimeError 时 worker 线程当场
    死亡，后续语句留在队列里 —— 只入库、永不翻译、永不 emit。新实现捕 Exception →
    告警后继续，该段照常 emit 原文，后续段落正常处理；失败段由 stop() 补译补齐。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.1)
    tr = FakeTranslatorUnexpectedErrorOnce()
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRPerChunk(),
                 translator=tr, diarizer=None)
    events = []
    statuses = []
    r.on_segment(events.append)
    r.on_status(statuses.append)

    r.start("意外异常会议")        # 主循环同步跑完：3 块音频 → 3 条语句
    try:
        # 关键断言（变异自查点）：第 1 段抛异常后 worker 必须继续干完剩下 2 段。
        # 去掉 worker 里「except Exception」保护时 worker 猝死 → events 恒为 0 → 此处超时失败。
        assert _wait_until(lambda: len(events) == 3), (
            f"worker 未在意外异常后继续处理：events={[s.text for s in events]}")
    finally:
        r.stop()                   # 不得向外抛异常

    segs = r.store.segments()
    assert len(segs) == 3                                     # store 条目齐全
    assert [s.text for s in segs] == ["line 1", "line 2", "line 3"]
    assert len(events) == 3                                   # 每段都 emit，且未重复
    assert [s.text for s in events] == ["line 1", "line 2", "line 3"]
    assert all(s.translation for s in segs)                   # 失败段由 stop() 补译补齐
    assert tr.calls == 4                                      # 1 失败 + 2 正常 + 1 补译
    assert any("翻译异常" in m for m in statuses)             # 告警文案区别于 TranslationError


def test_no_ghost_speaker(monkeypatch):
    """N1 回归：与所有语句零重叠的 turn 不得注册出「幽灵说话人」。

    语句落在 0~3s，pyannote 却返回 10~12s 的 turn（真实场景：短 turn 与语句重叠
    <30% 时 assign_speakers 不标注任何语句）。旧实现 _globalize_spans 对未映射局部
    标签无条件 register_label → store.speakers() 多出一个无人使用的标签（纪要列出
    幽灵发言人、编号被顶高）。修复后只注册真正落到语句上的标签。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.05)
    diarizer = FakeDiarizerNoOverlap()
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRPerChunk(),
                 translator=FakeTranslator(), diarizer=diarizer)
    statuses = []
    r.on_status(statuses.append)
    r.start("幽灵说话人会议")
    assert _wait_until(lambda: diarizer.calls >= 1), "diarize 一轮都没跑，断言无意义"
    r.stop()                       # 收尾 join 标注线程，确保这一轮跑完

    segs = r.store.segments()
    assert len(segs) == 3
    assert all(s.speaker == "" for s in segs)      # 零重叠 → 无语句被标注
    assert r.store.speakers() == []                # 未映射标签不注册 → 无幽灵说话人
    assert any("已识别说话人：0" in m for m in statuses)   # 该轮确实跑完（而非从没跑）


# ---------------------------------------------------------------------------
# 修复轮 4：emit 记账原子化（at-most-once）+ diarize 标注段异常保护
# ---------------------------------------------------------------------------

class FakeTranslatorBlocking:
    """每次 translate 阻塞 delay 秒后返回（模拟慢接口，制造 stop() 与 worker 的交叠窗口）。"""

    def __init__(self, delay: float = 0.3):
        self.delay = delay
        self.calls = 0
        self.first_started = threading.Event()
        self._lock = threading.Lock()

    def translate(self, text, lang=""):
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            self.first_started.set()
        time.sleep(self.delay)
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeDiarizerAssignBoom:
    """label() 正常返回；配合「assign_speakers 抛异常」验证标注线程不猝死。"""

    def __init__(self):
        self.calls = 0

    def label(self, audio, sample_rate=16000):
        self.calls += 1
        return [(0.0, max(len(audio) / sample_rate, 0.1), "SPK_A")]


def test_stop_backfill_does_not_duplicate_events(monkeypatch):
    """Minor-1 回归：stop() 兜底补发与 worker 并发时，同一 seq 不得 emit 两次。

    构造：翻译接口每次阻塞 0.3s + TRANSLATE_DRAIN_TIMEOUT=0.05 → stop() 必然等不到
    worker 排空，于是走补译 / 兜底补发循环，与 worker 并发对同一 seq 做「检查 → 发送」。

    on_segment 回调故意 sleep 0.25s，把「检查 → 记账」窗口拉宽，让 stop() 的并发检查
    必然落在窗口内：旧实现 check-then-act 非原子 → 同一 seq 两条事件（本用例失败）；
    修复后先把 seq 记入 _emitted 再发送 → 至多一次。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.05)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_DRAIN_TIMEOUT", 0.05)
    tr = FakeTranslatorBlocking(delay=0.3)
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRPerChunk(),
                 translator=tr, diarizer=None)
    events = []

    def _slow_append(seg):
        time.sleep(0.25)          # 拉宽「检查 → 记账」窗口（见 docstring）
        events.append(seg)

    r.on_segment(_slow_append)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("兜底补发会议",), daemon=True)
    thread.start()
    try:
        assert tr.first_started.wait(timeout=5), "翻译 worker 未开始第一段翻译"
        assert _wait_until(lambda: len(r.store.segments()) == 3), "三句未全部入库"
        r.stop()                  # drain 超时 → 走补译 + 兜底补发路径
    finally:
        thread.join(timeout=5)    # 主循环结束（producer_done 已置位）
        if r._translate_thread is not None:
            r._translate_thread.join(timeout=15)   # 等 worker 收尾，消除断言竞态
    assert not thread.is_alive()

    segs = r.store.segments()
    assert len(segs) == 3
    assert all(s.translation for s in segs), "stop() 兜底后仍有缺译文的语句"
    seqs = [s.seq for s in events]
    assert len(seqs) == 3, f"事件数应为 3（每 seq 一次），实际 {[s.text for s in events]}"
    assert len(set(seqs)) == 3, f"同一 seq 被重复 emit：{seqs}"
    assert set(seqs) == {s.seq for s in segs}


def test_segment_callback_exception_does_not_kill_worker(monkeypatch):
    """Minor-1/2 回归：on_segment 回调抛 RuntimeError 时线程不死、异常不外冒。

    正常回调注册在前、抛异常回调注册在后（_emit 按注册顺序投递：抛异常回调会中断其后的
    回调，但不得杀死翻译 worker、不得中断后续语句的翻译与投递）。
    关键在 stop() 之前等满 3 条事件：若 _emit_once 不捕回调异常，worker 处理第 1 段时
    就猝死，第 2、3 段永远不会 emit → 此处超时失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.1)

    def _boom(seg):
        raise RuntimeError("模拟事件回调异常")

    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRPerChunk(),
                 translator=FakeTranslator(), diarizer=None)
    events = []
    statuses = []
    r.on_segment(events.append)
    r.on_segment(_boom)
    r.on_status(statuses.append)

    r.start("回调异常会议")       # 主循环同步跑完；回调异常不得冒出
    try:
        assert _wait_until(lambda: len(events) == 3), (
            f"回调异常后事件未继续投递：{[s.text for s in events]}")
    finally:
        r.stop()                  # 不得向外抛异常
    if r._translate_thread is not None:
        r._translate_thread.join(timeout=10)

    segs = r.store.segments()
    assert len(segs) == 3
    assert [s.text for s in events] == ["line 1", "line 2", "line 3"]
    assert all(s.translation for s in segs)
    warnings = [m for m in statuses if "语句事件回调异常" in m]
    assert len(warnings) == 3, f"每段一条告警，实际 {warnings}"


def test_diarize_worker_survives_post_label_error(monkeypatch):
    """Minor-3 回归：label() 之后的映射 / 分配 / 注册抛异常不得杀死标注线程。

    在 assign_speakers 处注入 RuntimeError：旧实现异常直穿 while 循环 → 标注线程猝死，
    此后整场会议再无说话人标注；修复后告警并进入下一轮（本用例断言第二轮照样开跑）。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.05)

    def _boom(segments, spans, min_overlap=0.3):
        raise RuntimeError("模拟标注分配失败")

    monkeypatch.setattr("meeting_bot.recorder.assign_speakers", _boom)

    diarizer = FakeDiarizerAssignBoom()
    capture = FakeCaptureStaged(chunks=3, chunk_samples=16000)
    r = Recorder(_settings(), capture=capture, asr=FakeASRPerChunk(),
                 translator=FakeTranslator(), diarizer=diarizer)
    statuses = []
    r.on_status(statuses.append)

    thread = threading.Thread(target=r.start, args=("标注异常会议",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: diarizer.calls >= 1), "第一轮标注未开跑"
        capture.release(2)        # 放行第三块 → 触发第二轮
        assert _wait_until(lambda: diarizer.calls >= 2), "标注线程在异常后已猝死"
    finally:
        r.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()

    assert len(r.store.segments()) == 3
    assert all(s.speaker == "" for s in r.store.segments())   # 分配失败 → 无人被标注
    assert any("说话人标注失败" in m for m in statuses)


# ---------------------------------------------------------------------------
# Task 12 修复轮 1：stop-before-start 守卫（Major-3）
# ---------------------------------------------------------------------------

def test_start_after_stop_does_not_open_capture(monkeypatch):
    """Major-3 回归：stop() 先于 start() 落地时，start() 不得打开采集。

    场景：用户点「开始记录」后立刻点「结束并生成纪要」——start() 在工作线程里
    晚于 GUI 线程的 stop() 执行。旧实现照常 _capture.start() 并跑主循环（_stop
    已置位 → 主循环立即退出），但此后无人再调用 stop()：采集被打开却永不关闭，
    loopback 通道一直占用到进程退出。守卫必须在 _capture.start() 之前返回。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    capture = FakeCapture()
    r = Recorder(_settings(), capture=capture, asr=FakeASR(),
                 translator=FakeTranslator(), diarizer=None)
    statuses = []
    r.on_status(statuses.append)

    r.stop()                    # stop-before-start：先到者把 _stop 置位
    r.start("竞态会议")          # 晚到的 start 必须被取消

    assert capture.started is False          # 采集从未被打开（旧实现此处为 True）
    assert statuses[-1] == "录制已取消"
    assert r.title == ""                     # 取消后不接受标题
    assert r.started_at is None              # 未进入录制
    assert r.store.segments() == []          # 无语句入库
    assert r._translate_thread is None       # 翻译 worker 未启动
    assert r._diarize_thread is None


# ---------------------------------------------------------------------------
# Task 12 修复轮 2：守卫窗口的 TOCTOU 闭环（New-1）
# ---------------------------------------------------------------------------

class FakeCaptureSlowStart:
    """start() 阻塞到测试放行；start() 返回前 stop() 是空操作（建模真实采集）。

    真实 AudioCapture.start() 只负责起线程、stream 在采集线程里才真正打开，因此
    「stop() 早于 start() 返回」时它关不掉任何东西（空操作）。本 fake 精确复现这一点：
    opened 只在 start() 返回时置位，未 opened 时的 stop() 不记账、不关闭。
    """

    def __init__(self):
        self.opened = False
        self.closed = False
        self.stop_calls = 0
        self.enter_start = threading.Event()          # start() 已进入（正在打开采集）
        self.release_start = threading.Event()        # 测试放行：start() 此刻才返回
        self.stop_seen_unopened = threading.Event()   # stop() 在「采集未打开」时到达

    def start(self):
        self.enter_start.set()
        assert self.release_start.wait(timeout=10), "测试未放行 capture.start()"
        self.opened = True

    def read(self, timeout=1.0):
        return None          # 结束哨兵：主循环立即退出（本用例只关心「打开/关闭」配对）

    def stop(self):
        if not self.opened:
            self.stop_seen_unopened.set()    # 采集还没打开 → 空操作（同真实实现）
            return
        self.stop_calls += 1
        self.closed = True
        self.opened = False


def test_forced_interleave_stop_wins_capture_is_still_closed(monkeypatch):
    """New-1 回归：_stop 守卫与 _capture.start() 之间的亚毫秒窗口也必须闭环。

    强制交错（不靠 sleep 撞运气）：
      1. start() 线程通过 _stop 守卫后阻塞在 capture.start() 内（采集尚未打开）；
      2. 另一线程抢先 stop()：_stop 置位 + capture.stop() 空操作（此刻没得关）；
      3. 放行 capture.start() → 采集此刻才真正打开，主循环因 _stop 已置位立即退出；
      4. 主循环 finally 补一次 capture.stop() → 采集被关闭。

    旧实现（只在 stop() 里关采集）第 4 步什么都不做 → stop_calls == 0，
    loopback 通道一路占用到进程退出。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    capture = FakeCaptureSlowStart()
    r = Recorder(_settings(), capture=capture, asr=FakeASR(),
                 translator=FakeTranslator(), diarizer=None)
    statuses = []
    r.on_status(statuses.append)
    errors = []

    def _start():
        try:
            r.start("交错会议")
        except BaseException as e:   # 线程内异常默认只打日志，带回主线程断言
            errors.append(e)

    starter = threading.Thread(target=_start, name="start-thread", daemon=True)
    starter.start()
    assert capture.enter_start.wait(timeout=5), "start() 未进入 capture.start()"

    stopper = threading.Thread(target=r.stop, name="stop-thread", daemon=True)
    stopper.start()                  # stop() 抢先：此时采集还没打开
    try:
        assert capture.stop_seen_unopened.wait(timeout=5), "stop() 未在采集打开前落地"
        capture.release_start.set()  # 放行：采集在此刻才真正打开
        starter.join(timeout=10)
        stopper.join(timeout=10)
    finally:
        capture.release_start.set()  # 断言失败也要放行，避免线程卡到 10s 超时

    assert not starter.is_alive() and not stopper.is_alive()
    assert errors == []
    assert capture.stop_calls >= 1, "采集被打开后无人关闭（TOCTOU 残留）"
    assert capture.closed is True and capture.opened is False
    # 守卫已放行才算真的走进了 TOCTOU 窗口；若守卫提前挡下，本用例就退化成
    # 上一个用例（start_after_stop），不再覆盖「先开后关」这条路径。
    assert "录制已取消" not in statuses
    assert r.store.segments() == []
    assert r._translate_thread is not None    # start() 确实跑到了主循环之后


class FakeCaptureWithError(FakeCapture):
    """采集线程中途出错（error 已记录）+ 背压丢块，用于验证上层上报。"""

    def __init__(self):
        super().__init__(chunks=2)
        from meeting_bot.capture import AudioCapture
        self.error = RuntimeError("WASAPI stream broken")
        self.dropped_chunks = 4


def test_capture_failure_is_reported_to_status():
    """采集中断/丢块必须冒到状态回调：否则用户只看到「录音正常结束」而音频已缺。"""
    logs: list[str] = []
    rec = Recorder(_settings(), capture=FakeCaptureWithError())
    rec.on_status(logs.append)
    rec.start("测试")
    rec.stop()

    joined = "\n".join(logs)
    assert "声音采集中断" in joined
    assert "WASAPI stream broken" in joined
    assert "丢弃 4 块" in joined


def test_no_warning_when_capture_healthy():
    """采集正常（无 error / 无丢块）时不能误报。"""
    logs: list[str] = []
    rec = Recorder(_settings(), capture=FakeCapture(chunks=2))
    rec.on_status(logs.append)
    rec.start("测试")
    rec.stop()

    joined = "\n".join(logs)
    assert "声音采集中断" not in joined
    assert "丢弃" not in joined


# ---------------------------------------------------------------------------
# 最终审查修复波：补译免疫 + diarize 缓冲裁剪
# ---------------------------------------------------------------------------

class FakeTranslatorChoicesNone:
    """translate 全程抛 TypeError（复现 choices=None 的结构异常链）。

    translator._chat 的异常包裹（修复前不含 TypeError）接不住这个异常，它会直穿
    stop() 的 ``except TranslationError`` —— 这正是 Critical-1 的故障链。
    """

    def __init__(self):
        self.calls = 0

    def translate(self, text, lang=""):
        self.calls += 1
        raise TypeError("'NoneType' object is not subscriptable")

    def minutes(self, md):
        raise NotImplementedError


class FakeCaptureNSamples:
    """喂入 n 块固定样本数的静音后返回结束哨兵（缓冲裁剪用）。"""

    def __init__(self, chunks: int, samples: int):
        self.chunks = [np.zeros(samples, dtype=np.float32) for _ in range(chunks)]
        self._i = 0

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._i < len(self.chunks):
            c = self.chunks[self._i]
            self._i += 1
            return c
        return None

    def stop(self):
        pass


def test_stop_backfill_survives_unexpected_error(monkeypatch, tmp_path):
    """Critical-1 回归：非 TranslationError 的补译异常不得逃逸出 stop()。

    故障链：接口返回 choices=None → translate 抛 TypeError → stop() 的补译循环只捕
    TranslationError 接不住 → 异常逃逸到 app 的 finish() → export_all 整段被跳过 →
    整场会议零产物（无逐字稿、无纪要）。

    修复后：stop() 不抛异常、失败段保持空译文、每条都有告警、连续失败 3 次后放弃补译
    （有界重试，不再逐句重试拖长收尾），且 export_all 照常产出两份文件。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.1)
    tr = FakeTranslatorChoicesNone()
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRPerChunk(),
                 translator=tr, diarizer=None)
    events = []
    statuses = []
    r.on_segment(events.append)
    r.on_status(statuses.append)

    r.start("补译免疫会议")     # 主循环同步跑完：3 块音频 → 3 条语句
    r.stop()                  # 变异自查点：改回 except TranslationError，TypeError 逃逸 → 用例失败

    segs = r.store.segments()
    assert [s.text for s in segs] == ["line 1", "line 2", "line 3"]
    assert all(s.translation == "" for s in segs)      # 失败段保持空译文，未被伪造成原文
    assert tr.calls == 6                               # 3 次 worker + 3 次补译（3 次后放弃）
    assert sum(1 for m in statuses if "收尾补译失败" in m) == 3
    assert any("放弃本场补译" in m for m in statuses)   # 有界重试：不再逐句重试
    assert len(events) == 3                            # 放弃补译不影响兜底补发

    # 关键收益：stop() 正常返回 → 调用方 export_all 照常执行 → 两份产物落盘
    from meeting_bot.config import Settings
    from meeting_bot.export import export_all
    settings = Settings("https://x", "sk-test", "m1", "m2", "small", "",
                        tmp_path / "会议记录", tmp_path)
    transcript_path, minutes_path = export_all(r.store, settings, "补译免疫会议",
                                              r.started_at, translator=None)
    assert transcript_path.exists() and minutes_path.exists()


def test_diarize_buffer_is_trimmed(monkeypatch):
    """I2 回归：_diarize_audio 有上限（不再无界增长），裁剪量记入 _diarize_base。

    DIARIZE_BUFFER_SECONDS=0.05（800 样本），每块 400 样本 × 10 块喂入 → 缓冲只留
    800 样本，_diarize_base 记下裁剪掉的 3200。不变量：缓冲样本 + 裁剪样本 = 总样本，
    窗口起点换算回绝对时间轴。变异自查点：去掉 _trim_diarize_buffer 调用 → 缓冲仍是
    4000 样本 → 本用例失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_BUFFER_SECONDS", 0.05)
    chunk_samples, chunks = 400, 10
    r = Recorder(_settings(),
                 capture=FakeCaptureNSamples(chunks=chunks, samples=chunk_samples),
                 asr=FakeASR(), translator=None, diarizer=None)
    r.on_status(lambda m: None)
    r.start("缓冲裁剪会议")
    r.stop()

    total = chunks * chunk_samples                 # 4000
    limit = int(0.05 * 16000)                      # 800
    kept = sum(len(a) for a in r._diarize_audio)
    assert kept <= limit + chunk_samples, f"缓冲未裁剪：{kept} 样本"
    assert kept + r._diarize_base == total, "缓冲样本 + 裁剪样本必须守恒（时间轴不漂移）"
    assert r._diarize_base == total - kept

    audio, base = r._take_diarize_window()
    assert len(audio) == kept                      # 窗口不超过缓冲
    assert base == r._diarize_base / 16000.0       # 窗口起点 = 裁剪量换算的绝对秒数
