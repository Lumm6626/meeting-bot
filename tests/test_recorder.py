# -*- coding: utf-8 -*-
"""Task 9 录制编排集成测试：全部注入 fake，不加载任何重型依赖。"""
import queue
import re
import threading
import time

import numpy as np
import pytest

from meeting_bot.config import Settings
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


class FakeASRBatchChunks:
    """按批内每 1 秒切一条语句（模拟 whisper 对整批音频的分段）。

    识别改批处理前，每个 transcribe 调用恰好对应一句；改后一次调用对应一批。
    返回值按「批内 1 秒粒度」还原语句，使断言与「这批是怎么切的」完全解耦：
    无论批里有 1 句还是 3 句，产出的语句条数与文本序列都确定。
    """

    def __init__(self):
        self.calls = 0
        self.batch_lens: list[int] = []      # 每次 transcribe 收到的样本数（批大小证据）
        self._produced = 0                   # 已产出的语句数（跨批连续编号）

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        self.batch_lens.append(len(audio))
        n = max(1, len(audio) // 16000)      # 每条语句恰好 1 秒 → n = 批内语句数
        segs = [Segment(i + 0.1, i + 0.9, f"line {self._produced + i + 1}", lang="en")
                for i in range(n)]
        self._produced += n                  # 跨批连续编号：文本与批如何切分无关
        return segs


class FakeASRSlowBatch:
    """每次 transcribe 阻塞 delay 秒才返回（模拟收尾批的完整识别开销）。"""

    def __init__(self, delay: float = 0.3):
        self.calls = 0
        self.delay = delay
        self.started = threading.Event()

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        self.started.set()
        time.sleep(self.delay)
        n = max(1, len(audio) // 16000)
        return [Segment(i + 0.1, i + 0.9, f"line {i + 1}", lang="en")
                for i in range(n)]


def _stream_via_translate(self, text, lang="", context=""):
    """把「非流式」fake 适配成流式：一次 yield 完整译文（等于「译完才上屏」）。

    生产实现按 delta 累积逐字 yield；对 worker 而言两者只差通知次数，
    翻译结果 / 异常 / 调用计数完全一致，故既有用例只需这一行适配。
    """
    yield self.translate(text, lang)


class FakeTranslator:
    def __init__(self):
        self.calls = 0

    translate_stream = _stream_via_translate

    def translate(self, text, lang="", context=""):
        self.calls += 1
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeTranslatorThreads(FakeTranslator):
    """记录每次 translate 发生在哪个线程（分辨「worker 翻译」与「stop() 补译」）。"""

    def __init__(self):
        super().__init__()
        self.threads: list[threading.Thread] = []

    def translate(self, text, lang="", context=""):
        self.threads.append(threading.current_thread())
        return super().translate(text, lang)


class FakeTranslatorFailOnce:
    """第一次 translate 抛 TranslationError，之后成功（覆盖 stop() 补译路径）。"""

    def __init__(self):
        self.calls = 0

    translate_stream = _stream_via_translate

    def translate(self, text, lang="", context=""):
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
    # 批处理改造后默认阈值 30s：测试里把它压到「有语句就出批」，
    # 让语句在录制中途入库（diarize 才有一轮机会标注它）
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASR(),
                 translator=FakeTranslator(), diarizer=FakeDiarizer())
    events = []
    updates = []
    r.on_segment(events.append)
    r.on_update(updates.append)
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
    # 两阶段显示：原文事件每 seq 一次（去重），译文经 on_update 送达并带译文
    assert [s.seq for s in events] == [seg.seq]
    assert [s.seq for s in updates] == [seg.seq]
    assert updates[0].translation == "你好世界"
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
    asr = FakeASRBatchChunks()
    r = Recorder(_settings(), capture=FakeCaptureWithSentinel(), asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)
    r.start("尾部 flush 会议")
    r.stop()

    # 批处理：3 条语句（各 16000 样本）在收尾时拼成一批、只识别一次
    assert asr.calls == 1
    assert asr.batch_lens == [3 * 16000]
    segs = r.store.segments()
    assert len(segs) == 3
    assert [s.text for s in segs] == ["line 1", "line 2", "line 3"]
    assert all(s.translation == "你好世界" for s in segs)


# ---------------------------------------------------------------------------
# 修复轮 1：producer-done 模式，防尾部语句丢失
# ---------------------------------------------------------------------------

def test_stop_waits_for_tail_utterance(monkeypatch):
    """收尾批（stop 之后才触发的识别）必须由翻译 worker 消化，不能丢给 stop() 补译。

    批处理下最末一批语句在 _producer_done 置位**之后**才产出：翻译 worker 若仍只按
    「生产者完结 + 队列空」判定退出，会在那一瞬退出 → 尾批语句入队时已无人消费。
    断言译文的执行线程不是调用 stop() 的线程，即证明 worker 等到了 _asr_finished
    （走 stop() 补译路径时译文会由调用线程产出 → 本用例失败）。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.1)
    asr = FakeASRSlowBatch(delay=0.3)
    tr = FakeTranslatorThreads()
    r = Recorder(_settings(), capture=FakeCapture(), asr=asr,
                 translator=tr, diarizer=None)
    events = []
    r.on_segment(events.append)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("尾部会议",), daemon=True)
    thread.start()
    try:
        # 主循环（生产者）先跑完：3 条语句已入批但都没触发识别（阈值 30s）
        assert _wait_until(lambda: r._producer_done.is_set()), "主循环未结束"
        assert r.store.segments() == [], "阈值未到却已出批，用例前提不成立"
        assert asr.calls == 0
        r.stop()          # stop() 才触发收尾批 → 必须等它识别完再排空翻译
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()

    segs = r.store.segments()
    assert [s.text for s in segs] == ["line 1", "line 2", "line 3"]
    assert all(s.translation == "你好世界" for s in segs)
    assert [s.text for s in events] == ["line 1", "line 2", "line 3"]
    # 收尾批译自 worker 线程（而非 stop() 补译的调用线程）
    assert len(tr.threads) == 3
    assert all(t is not threading.current_thread() for t in tr.threads)
    assert not r._asr_thread.is_alive()        # stop() 已 join 识别线程


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
            # 批处理下一次调用对应整批音频 → 按批内 1 秒一句还原（3 句 = 3 秒）
            n = max(1, len(audio) // 16000)
            return [Segment(i + 0.5, i + 0.9, "lazy ok", lang="en") for i in range(n)]

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
    updates = []
    r.on_segment(events.append)
    r.on_update(updates.append)
    r.on_status(lambda m: None)
    r.start("中文会议")
    r.stop()

    segs = r.store.segments()
    assert len(segs) == 1
    assert segs[0].lang == "zh"
    assert segs[0].translation == segs[0].text == "你好世界"
    assert tr.calls == 0
    assert len(events) == 1
    # zh 直填同样走「原地更新」通道（第二阶段），否则面板上中文行不会刷新副标签状态
    assert [s.seq for s in updates] == [segs[0].seq]
    assert updates[0].translation == "你好世界"


# ---------------------------------------------------------------------------
# 流式翻译：逐字上屏 + 重绘节流 + 流中断回退
# ---------------------------------------------------------------------------

class FakeTranslatorStreamChunks:
    """流式 fake：逐块吐字符，块间可插延时（节流用例的核心道具）。

    ``delay=0`` → 10 块在毫秒级吐完（必然落在同一节流窗口内）；
    ``delay=0.6`` → 每块间隔均 > STREAM_UPDATE_INTERVAL（每块都该通知）。
    """

    def __init__(self, chunks="0123456789", delay: float = 0.0):
        self.chunks = chunks
        self.delay = delay
        self.stream_calls = 0
        self.translate_calls = 0

    def translate_stream(self, text, lang="", context=""):
        self.stream_calls += 1
        acc = ""
        for ch in self.chunks:
            if self.delay:
                time.sleep(self.delay)
            acc += ch
            yield acc

    def translate(self, text, lang="", context=""):
        """stop() 收尾补译走非流式（收尾阶段无人看实时性）。"""
        self.translate_calls += 1
        return "".join(self.chunks)

    def minutes(self, md):
        raise NotImplementedError


class FakeTranslatorStreamBreaks:
    """先正常吐 2 块译文，随后流中断（连接断开）；补译走非流式。"""

    def __init__(self):
        self.stream_calls = 0
        self.translate_calls = 0

    def translate_stream(self, text, lang="", context=""):
        self.stream_calls += 1
        yield "你"
        yield "你好"
        raise TranslationError("模拟流中断")

    def translate(self, text, lang="", context=""):
        self.translate_calls += 1
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


def test_streaming_updates_throttled(monkeypatch):
    """快流：10 块在毫秒级吐完（同一节流窗口）→ 通知被节流（不是每块都推 Qt），且末次必刷。

    变异自查点：去掉 _stream_translate 的节流判断 → 通知次数变成 10，断言失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorStreamChunks(chunks="0123456789", delay=0.0)
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    updates = []          # 回调里立刻快照译文：Segment 是原地改写的同一对象
    r.on_update(lambda s: updates.append(s.translation))
    r.on_status(lambda m: None)

    r.start("快流会议")
    try:
        assert _wait_until(lambda: r.store.segments()
                           and r.store.segments()[0].translation == "0123456789"), "译文未完成"
        # 首块必刷 + 尾块必刷 → 至少 2 次；10 块全发说明没节流
        assert 2 <= len(updates) < 10, f"节流失效：{len(updates)} 次通知 / 10 块"
        # 末次必刷：面板拿到的是完整译文（不是被节流挡住的倒数第二块）
        assert updates[-1] == "0123456789"
        assert r.store.segments()[0].translation == "0123456789"
        assert tr.stream_calls == 1                 # worker 只流一次，没退回非流式
        assert tr.translate_calls == 0
    finally:
        r.stop()


def test_streaming_updates_every_chunk_when_slow(monkeypatch):
    """慢流：块间隔 > 0.5s → 每块都通知（稀疏 token 不会被节流吞掉）。"""
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorStreamChunks(chunks="0123", delay=0.6)
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    updates = []          # 回调里立刻快照译文（同一 Segment 对象会被原地改写）
    r.on_update(lambda s: updates.append(s.translation))
    r.on_status(lambda m: None)

    r.start("慢流会议")
    try:
        assert _wait_until(lambda: r.store.segments()
                           and r.store.segments()[0].translation == "0123",
                           timeout=10), "译文未完成"
        # 每块间隔 0.6s > 0.5s 节流阈值 → 每块都通知，逐字上屏逐字可见
        assert updates == ["0", "01", "012", "0123"]
    finally:
        r.stop()


def test_stream_failure_reverts_to_original_then_backfill(monkeypatch):
    """流中断 → 译文清空（面板退回原文）+ 既有告警；stop() 补译补上最终译文。

    变异自查点：去掉 _stream_translate 的「回退时清空」→ seg.translation 停在半句
    「你好」，stop() 会认为「已有译文」跳过补译 → 断言失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorStreamBreaks()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    updates = []          # 回调里立刻快照译文（同一 Segment 对象会被原地改写）
    statuses = []
    r.on_update(lambda s: updates.append(s.translation))
    r.on_status(statuses.append)

    r.start("流中断会议")
    try:
        assert _wait_until(lambda: any("翻译失败" in m for m in statuses),
                           timeout=10), "未观察到流中断告警"
        seg = r.store.segments()[0]
        assert seg.translation == "", "流中断后译文必须清空（否则字幕停在残句）"
        # 半句确实上过屏（流式正常路径），随后被回退通知覆盖为空。
        # 只断言首块：第 2 块「你好」落在节流窗口内（未通知），回退通知已把它抹掉。
        assert updates[0] == "你"
        assert updates[-1] == ""
    finally:
        r.stop()                                    # 收尾补译（非流式）
    assert r.store.segments()[0].translation == "你好世界"
    assert updates[-1] == "你好世界"                  # 面板拿到最终译文
    assert tr.translate_calls == 1


class FakeTranslatorStreamPaced:
    """1.5 秒内匀速吐 30 块（模拟真 API 的 token 节奏）：验证刷新是匀速的。

    30 块 / 0.05s 间隔 = 1.5s。若「每块一刷」，面板会被推 30 次（用户反馈的
    拼接感正来自这里）；按 0.5s 节奏只该推 3~4 次。
    """

    def __init__(self, chunks: int = 30, interval: float = 0.05):
        self.chunks = chunks
        self.interval = interval
        self.full = "*" * chunks
        self.stream_calls = 0
        self.translate_calls = 0

    def translate_stream(self, text, lang="", context=""):
        self.stream_calls += 1
        acc = ""
        for _ in range(self.chunks):
            time.sleep(self.interval)
            acc += "*"
            yield acc

    def translate(self, text, lang="", context=""):
        self.translate_calls += 1
        return self.full

    def minutes(self, md):
        raise NotImplementedError


def test_streaming_updates_evenly_paced(monkeypatch):
    """匀速刷新：1.5 秒内 30 块 → 通知 ≤ 6 次（≈每 0.5s 一次），末次译文完整。

    变异自查点：
    * STREAM_UPDATE_INTERVAL 改成 0（每块都刷）→ 30 次通知 → 断言失败；
    * 改回 0.25（仍按 token 碎片刷）→ 约 7 次 → 断言失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorStreamPaced()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    updates = []          # 回调里立刻快照译文（同一 Segment 对象会被原地改写）
    r.on_update(lambda s: updates.append(s.translation))
    r.on_status(lambda m: None)

    r.start("匀速会议")
    try:
        assert _wait_until(lambda: updates and updates[-1] == tr.full,
                           timeout=15), "最终译文未上屏"
        # 30 块摊成 1.5 秒：0.5s 节奏下 3~4 次；≤6 给调度抖动留余量
        assert len(updates) <= 6, f"刷新过密（拼接感）：{len(updates)} 次 / 30 块"
        assert len(updates) >= 3, f"刷新太稀（看不出进度）：{len(updates)} 次 / 30 块"
        assert updates[-1] == tr.full, "末次通知必须是完整译文"
        assert r.store.segments()[0].translation == tr.full
    finally:
        r.stop()


class FakeASRFourLines:
    """一次 transcribe 吐 4 条语句：让第 4 句有 3 条已翻译的前文。"""

    def transcribe(self, audio, sample_rate=16000):
        return [Segment(0.5 + i, 0.9 + i, f"line {i + 1}", lang="en") for i in range(4)]


class FakeTranslatorRecordsContext:
    """记录每次翻译收到的 context；译文带前缀以验证「前文译文」进入了上下文。"""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []      # (原文, context)

    def translate_stream(self, text, lang="", context=""):
        self.calls.append((text, context))
        yield f"译:{text}"

    def translate(self, text, lang="", context=""):
        self.calls.append((text, context))
        return f"译:{text}"

    def minutes(self, md):
        raise NotImplementedError


def test_context_passed_to_translate(monkeypatch):
    """翻译带会议上下文：第 4 句吃前 3 句的原文/译文，且绝不含自己。

    变异自查点：把 worker 里的 ``context=self._recent_context(seg)`` 去掉
    → 第 4 句 context 变成空串 → 「line 1..3 in ctx4」断言失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorRecordsContext()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASRFourLines(),
                 translator=tr, diarizer=None)
    r.on_status(lambda m: None)

    r.start("上下文会议")
    try:
        assert _wait_until(lambda: len(r.store.segments()) == 4
                           and all(s.translation for s in r.store.segments()),
                           timeout=10), "4 条语句未全部翻译"
    finally:
        r.stop()

    assert [s.text for s in r.store.segments()] == [f"line {i}" for i in range(1, 5)]
    ctx_of = dict(tr.calls)          # 同一句只翻一次 → 用原文当 key 安全
    assert ctx_of["line 1"] == ""                  # 首句没有上文
    ctx4 = ctx_of["line 4"]
    for prev in ("line 1", "line 2", "line 3"):
        assert prev in ctx4, prev
    assert "line 4" not in ctx4, "待译句自己不得进上文（模型会照抄自己）"
    assert "原文：" in ctx4 and "译文：" in ctx4
    assert "译:line 1" in ctx4                     # 优先带已有译文（术语锚点）


class FakeTranslatorFailsInWorker:
    """流式阶段非首句一律失败（留给 stop() 补译），并记录补译收到的 context。"""

    def __init__(self):
        self.stream_calls = 0
        self.backfill: list[tuple[str, str]] = []      # (原文, context)

    def translate_stream(self, text, lang="", context=""):
        self.stream_calls += 1
        if text != "line 1":
            raise TranslationError("模拟流中断")
        yield f"译:{text}"

    def translate(self, text, lang="", context=""):
        self.backfill.append((text, context))
        return f"译:{text}"

    def minutes(self, md):
        raise NotImplementedError


def test_stop_backfill_passes_context(monkeypatch):
    """stop() 补译同样带上文（此刻 store 更全）：口音句在收尾时才有足够前文纠错。

    变异自查点：把 stop() 补译里的 ``context=self._recent_context(seg)`` 去掉
    → 补译收到的 context 全为空 → 「line 1 in backfill["line 2"]」断言失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorFailsInWorker()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASRFourLines(),
                 translator=tr, diarizer=None)
    statuses = []
    r.on_status(statuses.append)

    r.start("补译上下文会议")
    try:
        assert _wait_until(lambda: sum(1 for s in r.store.segments() if s.translation) == 1
                           and len(r.store.segments()) == 4, timeout=10), "只有首句该译成功"
    finally:
        r.stop()                                # 补译 line 2/3/4（带上下文）

    backfill = dict(tr.backfill)
    assert set(backfill) == {"line 2", "line 3", "line 4"}, f"补译集合异常：{set(backfill)}"
    assert "line 1" in backfill["line 2"]
    assert "line 2" not in backfill["line 2"], "补译句自己不得进上文"
    assert "line 3" in backfill["line 4"]     # 收尾时前文的译文也已补齐
    assert all(s.translation for s in r.store.segments())


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
    # 批处理阈值压到「有语句就出批」：语句必须在录制中途入库，diarize 才有机会标注
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)

    seen = []
    real_assign = assign_speakers

    def _spy(segments, spans, min_overlap=0.3):
        seen.append([tuple(s) for s in spans])
        return real_assign(segments, spans, min_overlap)

    monkeypatch.setattr("meeting_bot.recorder.assign_speakers", _spy)

    diarizer = FakeDiarizerWindow(local_label="SPK_A")
    r = Recorder(_settings(), capture=FakeCaptureOneLongChunk(32000), asr=FakeASRBatchChunks(),
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
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)  # 逐句入库（见上）

    diarizer = FakeDiarizerSequence(["SPK_A", "SPK_B"], full_window_samples=32000)
    capture = FakeCaptureStaged(chunks=3, chunk_samples=16000)
    r = Recorder(_settings(), capture=capture, asr=FakeASRBatchChunks(),
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

    translate_stream = _stream_via_translate

    def translate(self, text, lang="", context=""):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("模拟翻译接口意外异常")
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeDiarizerNoOverlap:
    """返回与所有语句零重叠的 span（语句都在 0~3s，span 在 10~12s）。

    seg_counts 记录每次 label() 调用那一刻「已入库语句数」：用来证明这一轮 diarize
    tick 真的看到了语句（否则「没有语句被标注 / 无幽灵说话人」是恒真的空断言，审查 N2）。
    """

    def __init__(self, seg_count=None):
        self.calls = 0
        self.seg_count = seg_count          # 可用 Recorder 建好后回填（lambda: len(...)）
        self.seg_counts: list[int] = []

    def label(self, audio, sample_rate=16000):
        self.calls += 1
        if self.seg_count is not None:
            self.seg_counts.append(self.seg_count())
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
    # 批处理阈值压到「有语句就出批」：本用例要求在 stop() 之前就观察到语句/事件
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRBatchChunks(),
                 translator=tr, diarizer=None)
    events = []
    statuses = []
    r.on_segment(events.append)
    r.on_status(statuses.append)

    r.start("意外异常会议")        # 主循环同步跑完：3 块音频 → 3 条语句
    try:
        # 关键断言（变异自查点）：第 1 段抛异常后 worker 必须继续干完剩下 2 段。
        # 两阶段显示下「原文事件」由识别线程发出（不再由 worker 发），所以存活证据
        # 改用 worker 侧的 translate 调用次数：去掉「except Exception」保护时 worker
        # 在第 1 段猝死 → tr.calls 恒为 1 → 此处超时失败。
        assert _wait_until(lambda: tr.calls >= 3), (
            f"worker 未在意外异常后继续处理：tr.calls={tr.calls}")
        assert len(events) == 3                  # 原文阶段：3 条都已上屏
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
    # 批处理改造后必须压小批量阈值：默认 30s/35s 下 diarize tick 会跑在语句入库之前，
    # 「没有语句被标注 / 无幽灵说话人」就变成恒真断言（测试退化，审查 M3）。
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    diarizer = FakeDiarizerNoOverlap()
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRBatchChunks(),
                 translator=FakeTranslator(), diarizer=diarizer)
    # Recorder 建好后再回填取数器：label() 在标注线程里调用时才读 store（审查 N2）
    diarizer.seg_count = lambda: len(r.store.segments())
    statuses = []
    r.on_status(statuses.append)
    r.start("幽灵说话人会议")
    assert _wait_until(lambda: diarizer.calls >= 1), "diarize 一轮都没跑，断言无意义"
    # 关键断言（N2）：这一轮 tick 必须「已经看到语句」。若 diarize 跑在语句入库之前，
    # 下面的「无语句被标注 / 无幽灵说话人」是恒真断言（测试退化，审查 M3/N2）。
    assert _wait_until(lambda: max(diarizer.seg_counts or [0]) >= 1), (
        f"diarize tick 从未看到语句，断言无意义：seg_counts={diarizer.seg_counts}")
    r.stop()                       # 收尾 join 标注线程，确保这一轮跑完
    assert max(diarizer.seg_counts) >= 1     # 收尾后再确证一次（防 stop 前竞态）

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

    translate_stream = _stream_via_translate

    def translate(self, text, lang="", context=""):
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
    """Minor-1 回归：stop() 兜底补发与识别线程并发时，同一 seq 不得 emit 两次。

    构造：翻译接口每次阻塞 0.3s + TRANSLATE_DRAIN_TIMEOUT=0.05 → stop() 必然等不到
    worker 排空，于是走补译 / 兜底补发循环，与识别线程并发对同一批 seq 做
    「检查 → 发送」。

    on_segment 回调故意 sleep 0.25s，把「检查 → 记账」窗口拉宽，让 stop() 的并发检查
    必然落在窗口内：旧实现 check-then-act 非原子 → 同一 seq 两条事件（本用例失败）；
    修复后先把 seq 记入 _emitted 再发送 → 至多一次。

    两阶段改造后两条通道各自去重：原文事件每 seq 一次（识别完即发），译文只走
    on_update（每条语句都拿到一次通知、且都带译文）。update 允许重复投递（补译与
    worker 迟到译文可能各通告一次，面板 update_segment 按 seq 幂等），故只断言覆盖。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.05)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_DRAIN_TIMEOUT", 0.05)
    # 批处理阈值压到「有语句就出批」：本用例要求在 stop() 之前就观察到语句/事件
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorBlocking(delay=0.3)
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRBatchChunks(),
                 translator=tr, diarizer=None)
    events = []
    updates = []

    def _slow_append(seg):
        time.sleep(0.25)          # 拉宽「检查 → 记账」窗口（见 docstring）
        events.append(seg)

    r.on_segment(_slow_append)
    r.on_update(updates.append)
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
    # 译文通道：每条语句都收到过更新通知，且通知里带译文（否则面板停在原文）
    up_seqs = [s.seq for s in updates]
    assert set(up_seqs) == {s.seq for s in segs}, f"有语句的译文未通知面板：{up_seqs}"
    assert all(s.translation for s in updates), "update 通知里不得缺译文"


def test_segment_callback_exception_does_not_kill_worker(monkeypatch):
    """Minor-1/2 回归：on_segment 回调抛 RuntimeError 时线程不死、异常不外冒。

    正常回调注册在前、抛异常回调注册在后（_emit 按注册顺序投递：抛异常回调会中断其后的
    回调，但不得杀死识别线程、不得中断后续语句的识别与投递）。
    关键在 stop() 之前等满 3 条事件：原文事件在识别线程里发出（两阶段第一阶段），
    若 _emit_once 不捕回调异常，识别线程处理第 1 段时就猝死，第 2、3 段永远不会
    emit → 此处超时失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.TRANSLATE_POLL_TIMEOUT", 0.1)

    # 批处理阈值压到「有语句就出批」：本用例要求在 stop() 之前就观察到语句/事件
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)

    def _boom(seg):
        raise RuntimeError("模拟事件回调异常")

    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRBatchChunks(),
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
    r = Recorder(_settings(), capture=capture, asr=FakeASRBatchChunks(),
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

    translate_stream = _stream_via_translate

    def translate(self, text, lang="", context=""):
        self.calls += 1
        raise TypeError("'NoneType' object is not subscriptable")

    def minutes(self, md):
        raise NotImplementedError


class FakeCaptureNSamples:
    """喂入 n 块固定样本数的静音后返回结束哨兵（缓冲裁剪用）。

    delay > 0 时每块之前 sleep（模拟实时采集节奏），供标注线程在喂入期间穿插跑轮。
    """

    def __init__(self, chunks: int, samples: int, delay: float = 0.0):
        self.chunks = [np.zeros(samples, dtype=np.float32) for _ in range(chunks)]
        self.delay = delay
        self._i = 0

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._i < len(self.chunks):
            if self.delay:
                time.sleep(self.delay)
            c = self.chunks[self._i]
            self._i += 1
            return c
        return None

    def stop(self):
        pass


class FakeDiarizerCounter:
    """只统计 label 调用次数（音频内容无关），用于断言标注线程是否持续跑轮。"""

    def __init__(self):
        self.calls = 0

    def label(self, audio, sample_rate=16000):
        self.calls += 1
        return []


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
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASRBatchChunks(),
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


def test_diarize_keeps_running_after_trim(monkeypatch):
    """最终复审 N-1 回归：缓冲进入裁剪稳态后，说话人标注不得永久停摆。

    故障链：DIARIZE_BUFFER_SECONDS 很小 → 几块后每轮「append 1 块 + trim 丢 1 块」，
    len(_diarize_audio) 恒定 → 旧门控（与上次块数比较）从首个稳态 tick 起永远判定
    「无新音频」→ 标注永久停摆（真实会议里录制 6-7 分钟、缓冲填满后开始）。
    修复后门控用单调递增的 _diarize_appended。

    变异自查点：门控改回 len(self._diarize_audio) 比较 → label 只被调 1 次 → 本用例失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.05)
    # 800 样本 ≈ 2 块：喂到第 3 块起进入「每 append 必 trim」的稳态
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_BUFFER_SECONDS", 0.05)

    diarizer = FakeDiarizerCounter()
    # 每块间隔 0.02s（共 30 块 ≈ 0.6s）：喂入比标注周期慢，稳态下每轮 tick 都能看到新块
    capture = FakeCaptureNSamples(chunks=30, samples=400, delay=0.02)
    r = Recorder(_settings(), capture=capture, asr=FakeASR(),
                 translator=None, diarizer=diarizer)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("裁剪稳态会议",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: diarizer.calls >= 3), (
            f"裁剪稳态后说话人标注停摆：label 只被调 {diarizer.calls} 次")
        assert r._diarize_base > 0, "用例前提不成立：缓冲尚未进入裁剪稳态"
    finally:
        r.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# 音频源：audio_source 只在「未注入 capture」时转发给 AudioCapture
# ---------------------------------------------------------------------------

def test_audio_source_forwarded_to_capture(monkeypatch):
    """Recorder(settings, audio_source="mic") → AudioCapture(source="mic")。

    默认值必须是 both（自动双路，线上/线下免切换）；显式注入 capture 时不得
    再构造 AudioCapture —— 上层已探测可用设备并复用它，重复构造会白打开一次设备。
    """
    created: list[dict] = []

    class _SpyAudioCapture:
        def __init__(self, *args, **kwargs):
            created.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr("meeting_bot.recorder.AudioCapture", _SpyAudioCapture)

    # 1) 显式 mic
    Recorder(_settings(), audio_source="mic")
    assert created[-1]["kwargs"] == {"source": "mic"}
    assert created[-1]["args"] == ()

    # 2) 不传：默认自动双路（扬声器 loopback + 麦克风）
    Recorder(_settings())
    assert created[-1]["kwargs"] == {"source": "both"}

    # 3) 显式注入 capture：audio_source 被忽略，不再构造新捕获
    before = len(created)
    injected = FakeCapture()
    r = Recorder(_settings(), capture=injected, audio_source="mic")
    assert len(created) == before, "注入 capture 时不应构造 AudioCapture"
    assert r._capture is injected
    assert r._audio_source == "mic"      # 参数仍记录（便于诊断），但不生效


# ---------------------------------------------------------------------------
# 识别批处理（ASR_BATCH_SECONDS / ASR_BATCH_MAX_WAIT / 收尾批）
# ---------------------------------------------------------------------------

class FakeCaptureHeld:
    """喂完给定块后一直「读超时」（不返回结束哨兵），录制持续到 stop() 置位。"""

    def __init__(self, chunks: list[np.ndarray]):
        self._chunks = list(chunks)
        self._fed = 0
        self._done = threading.Event()

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._fed < len(self._chunks):
            c = self._chunks[self._fed]
            self._fed += 1
            return c
        if self._done.is_set():
            return None                 # 收尾哨兵：只有 stop() 之后才出现
        time.sleep(0.005)               # 真实 read 会阻塞 timeout 秒，这里只做节流
        raise queue.Empty()

    def stop(self):
        self._done.set()


class FakeVADOffsets:
    """flush 时吐出指定样本偏移的语句（用于构造确定的批起点）。"""

    def __init__(self, offsets, samples: int = 16000):
        self.offsets = list(offsets)
        self.samples = samples

    def push(self, chunk):
        return []

    def flush(self):
        out = [(o, np.zeros(self.samples, dtype=np.float32)) for o in self.offsets]
        self.offsets = []
        return out


class FakeASRRelative:
    """总返回批内相对段 (0.5,1.5) / (1.5,2.5)：验证「批内相对 → 会议绝对」换算。"""

    def __init__(self):
        self.calls = 0
        self.batch_lens: list[int] = []

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        self.batch_lens.append(len(audio))
        return [Segment(0.5, 1.5, "r1", lang="en"),
                Segment(1.5, 2.5, "r2", lang="en")]


def test_batch_triggers_on_audio_length(monkeypatch):
    """长度触发：积压音频达 ASR_BATCH_SECONDS 立即出批（录制仍在进行，非收尾触发）。

    变异自查点：去掉 _asr_batch_ready 里的长度判定 → 只能等 stop() 收尾 → 本用例超时失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADTwoUtterances)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.05)    # 800 样本即达标
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_MAX_WAIT", 999.0)  # 排除 max-wait 干扰

    asr = FakeASRBatchChunks()
    capture = FakeCaptureHeld([np.zeros(16000, dtype=np.float32)])
    r = Recorder(_settings(), capture=capture, asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("按长度出批",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: len(r.store.segments()) == 2), \
            f"积压 2s 远超阈值 0.05s 却未出批：batch_lens={asr.batch_lens}"
        # 一批 = 两条语句拼接（2 秒音频），不是逐句识别
        assert asr.calls == 1
        assert asr.batch_lens == [2 * 16000]
        assert r._stop.is_set() is False          # 出批发生在录制中途
        assert [s.start for s in r.store.segments()] == [0.1, 1.1]
    finally:
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()


def test_batch_flushes_on_max_wait(monkeypatch):
    """静默会议兜底：音频长度永远够不到阈值时，最老语句等够墙钟也必须出批。

    变异自查点：去掉 max-wait 判定 → 只有 stop() 才出批 → 本用例超时失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 999.0)   # 长度永不达标
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_MAX_WAIT", 0.05)   # 只需等 50ms

    asr = FakeASRBatchChunks()
    capture = FakeCaptureHeld([np.zeros(16000, dtype=np.float32)])
    r = Recorder(_settings(), capture=capture, asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("静默会议",), daemon=True)
    thread.start()
    try:
        # 轮询间隔 0.5s → 首次醒来即判定 waited >= 0.05s
        assert _wait_until(lambda: len(r.store.segments()) == 1), "max-wait 兜底未触发"
        assert asr.calls == 1 and asr.batch_lens == [16000]
        assert r._stop.is_set() is False          # 同样发生在录制中途
        assert r.store.segments()[0].start == 1.1  # 偏移 16000 样本（1s）+ 0.1
    finally:
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()


def test_final_batch_on_stop(monkeypatch):
    """收尾批：录制结束时残余积压（两个阈值都没到）必须被识别，一条不丢。

    变异自查点：把 worker 的 force 参数（stop 置位即强制出批）改成 False → 残余批要等
    max-wait（默认 35s）才肯出，stop() 耗时断言失败（收尾被无谓拖长）。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_MAX_WAIT", 999.0)  # 排除 max-wait

    asr = FakeASRBatchChunks()
    capture = FakeCaptureHeld([np.zeros(16000, dtype=np.float32)] * 3)
    r = Recorder(_settings(), capture=capture, asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("收尾批会议",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: len(r._asr_batch) == 3), "3 条语句未全部入批"
        assert asr.calls == 0 and r.store.segments() == []   # 阈值未到：一条都没识别
        t0 = time.monotonic()
        r.stop()                                              # 收尾批在此触发
        elapsed = time.monotonic() - t0
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()

    # 收尾是「立即带走」（force），不是等 max-wait 35 秒才出批
    assert elapsed < 5.0, f"stop() 等了 {elapsed:.1f}s 才收尾（force 未生效？）"

    assert asr.calls == 1
    assert asr.batch_lens == [3 * 16000]        # 残余 3 条拼成一批带走
    segs = r.store.segments()
    assert [s.text for s in segs] == ["line 1", "line 2", "line 3"]
    assert all(s.translation == "你好世界" for s in segs)


def test_offsets_preserved_in_batch(monkeypatch):
    """绝对时间轴：语句绝对时间 = 批起点（批内最小偏移）+ whisper 相对时间。

    两个场景批起点不同（0 / 1s），ASR 返回的相对段完全一样 → 绝对时间必须整体平移 1s。
    变异自查点：忽略 base（直接用 s.start / s.end）→ 场景 B 断言失败（两场景结果相同）。
    """
    def _run(offsets):
        monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer",
                            lambda: FakeVADOffsets(offsets))
        asr = FakeASRRelative()
        rec = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=asr,
                       translator=None, diarizer=None)
        rec.on_status(lambda m: None)
        rec.start("偏移会议")
        rec.stop()
        return asr, rec.store.segments()

    # 场景 A：批起点 = 0 样本 → 绝对时间 = 相对时间
    asr_a, segs_a = _run([0, 16000])
    assert asr_a.batch_lens == [2 * 16000]
    assert [(s.start, s.end) for s in segs_a] == [(0.5, 1.5), (1.5, 2.5)]

    # 场景 B：批起点 = 16000 样本（1s）→ 绝对时间整体 +1s
    asr_b, segs_b = _run([16000, 32000])
    assert asr_b.batch_lens == [2 * 16000]
    assert [(s.start, s.end) for s in segs_b] == [(1.5, 2.5), (2.5, 3.5)]


# ---------------------------------------------------------------------------
# 修复轮 1：批内时间轴回填（保语句间隙）+ emit 异常免疫
# ---------------------------------------------------------------------------

class FakeASRGapTwoSegs:
    """模拟 whisper 对「拼接缓冲」分段：段相对时间按拼接缓冲连续 1 秒切。

    输入是两条各 1 秒、中间隔 10 秒静音的语句，拼接后缓冲仍只有 2 秒 →
    返回 (0,1)/(1,2)。正确实现必须把它们回填到 0~1s 与 11~12s。
    """

    def __init__(self):
        self.calls = 0
        self.batch_lens: list[int] = []

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        self.batch_lens.append(len(audio))
        return [Segment(0.0, 1.0, "first", lang="en"),
                Segment(1.0, 2.0, "second", lang="en")]


def test_batch_preserves_inter_utterance_gaps(monkeypatch):
    """M1 回归：批内语句之间的静音不得被吃掉（时间轴压缩会让后续语句系统性偏早）。

    VAD 产出两条语句：偏移 0 与 11*16000（中间 10 秒静音）。拼接后缓冲只有 2 秒，
    whisper 返回 (0,1)/(1,2)。旧实现 base=0 直接加相对时间 → 第二条落在 1.0~2.0s
    （比真实时间早 10 秒），逐字稿时间戳错误 + 与 pyannote 时间轴错位（说话人漏标/错标）。

    变异自查点：去掉 to_abs()/spans 映射（改回 base + s.start）→ 本用例断言失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer",
                        lambda: FakeVADOffsets([0, 11 * 16000]))
    asr = FakeASRGapTwoSegs()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=asr,
                 translator=None, diarizer=None)
    r.on_status(lambda m: None)
    r.start("间隙会议")
    r.stop()

    # 两条语句拼成一批（2 秒音频），间距 10s 不出现在拼接音频里
    assert asr.calls == 1
    assert asr.batch_lens == [2 * 16000]
    segs = r.store.segments()
    assert [s.text for s in segs] == ["first", "second"]
    assert segs[0].start == 0.0
    # 关键断言（N1）：第一条的终点恰在语句边界（1.0s），必须留在 1.0 而不是跳到
    # 第二条语句起点 11.0 —— 半开区间 [rs, re) 用在终点上会让时长膨胀 11 倍，
    # 说话人重叠比例随之错乱（漏标）。
    assert segs[0].end == 1.0, f"起点边界开闭错：end={segs[0].end}"
    # 关键断言：第二条必须回到 11.0~12.0，而不是 1.0~2.0
    assert abs(segs[1].start - 11.0) < 1e-6, f"间隙被吃掉：start={segs[1].start}"
    assert abs(segs[1].end - 12.0) < 1e-6, f"间隙被吃掉：end={segs[1].end}"


class FakeASRBoundarySegs:
    """返回终点/起点恰好压在语句边界上的段（(0.5,1.0) / (1.0,1.5)）。"""

    def __init__(self, segs):
        self.segs = list(segs)
        self.calls = 0
        self.batch_lens: list[int] = []

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        self.batch_lens.append(len(audio))
        return [Segment(a, b, t, lang="en") for a, b, t in self.segs]


def test_batch_segment_end_on_boundary(monkeypatch):
    """N1 回归：段终点压在语句边界上时，必须归属「前一条语句的终点」，不得跳到后一句起点。

    配置：两条 1 秒语句，偏移 0 与 11*16000（中间 10 秒静音）。拼接缓冲只有 2 秒，
    缓冲位置 1.0s 同时是「第 1 条语句的终点」和「第 2 条语句的起点」，但两者的会议
    绝对时间差 10 秒（1.0 vs 11.0）—— 边界点归属哪一侧直接决定时间戳对错。

    终点（is_end）：按 (rs, re] 闭右 → (0.5, 1.0)，时长 0.5s；
    旧实现（终点也用半开 [rs, re)）：落到第 2 条语句 → (0.5, 11.0)，时长膨胀 11 倍
    → assign_speakers 的重叠比例错乱（说话人漏标/错标）。

    三场景：相邻（0/16000，两侧同值，只看形状）、有间隙（0/11*16000，判据）、
    起点压边界（0/11*16000，起点仍按 [rs, re) 闭左 → 归后一句，与终点相反）。

    变异自查点：删掉 to_abs 的 is_end 分支（终点退回半开判定）→ 场景 B 得到 11.0 → 失败。
    """
    def _run(offsets, segs):
        monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer",
                            lambda: FakeVADOffsets(offsets, samples=16000))
        asr = FakeASRBoundarySegs(segs)
        rec = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=asr,
                       translator=None, diarizer=None)
        rec.on_status(lambda m: None)
        rec.start("边界会议")
        rec.stop()
        return asr, rec.store.segments()

    # 场景 A：两语句相邻（偏移 0 / 16000）→ 边界两侧同为 1.0，形状正确且不膨胀
    asr_a, segs_a = _run([0, 16000], [(0.5, 1.0, "a")])
    assert asr_a.batch_lens == [2 * 16000]
    assert [(s.start, s.end) for s in segs_a] == [(0.5, 1.0)], segs_a

    # 场景 B（判据）：中间 10 秒静音 → 终点必须留在 1.0，不能跳到 11.0
    asr_b, segs_b = _run([0, 11 * 16000], [(0.5, 1.0, "b")])
    assert asr_b.batch_lens == [2 * 16000]
    assert (segs_b[0].start, segs_b[0].end) == (0.5, 1.0), (
        f"终点跳过边界：{segs_b[0].start}~{segs_b[0].end}（续长 {segs_b[0].end - segs_b[0].start:.1f}s）")

    # 场景 C：起点压在边界上 → 起点按 [rs, re) 闭左，归后一条语句（11.0 起）
    _, segs_c = _run([0, 11 * 16000], [(1.0, 1.5, "c")])
    assert (segs_c[0].start, segs_c[0].end) == (11.0, 11.5), (
        f"起点边界归属错：{segs_c[0].start}~{segs_c[0].end}")


def test_emit_exception_does_not_kill_asr_worker(monkeypatch):
    """M2 回归：on_segment 回调抛异常不得杀死识别线程（否则此后全场无字幕）。

    translator=None 路径旧实现直接 _emit(seg)：回调异常直穿 _transcribe_batch →
    _asr_worker 当场死亡、无告警，后续批次永久不出库。改为 _emit_once 后每个 seq
    照常记账、异常只降级为状态栏告警。

    变异自查点：把 _emit_once 改回 _emit → 识别线程在第一条语句后死亡 →
    store 只有 1 条、is_alive() 为 False → 本用例失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)   # 有语句就出批
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_MAX_WAIT", 999.0)
    capture = FakeCaptureHeld([np.zeros(16000, dtype=np.float32)] * 3)
    # FakeASRBatchChunks 跨批连续编号：批如何切分都不影响 "line 1..3" 的文本序列
    r = Recorder(_settings(), capture=capture, asr=FakeASRBatchChunks(),
                 translator=None, diarizer=None)      # translator=None → 走 _emit_once 路径
    statuses: list[str] = []
    emitted: list[Segment] = []

    def boom(seg):
        emitted.append(seg)
        raise RuntimeError("模拟 UI 回调崩溃")

    r.on_segment(boom)
    r.on_status(statuses.append)

    thread = threading.Thread(target=r.start, args=("回调崩溃会议",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: len(r.store.segments()) == 3), (
            f"识别线程疑似猝死：store={[s.text for s in r.store.segments()]}")
        # 入库早于 emit（store.add 先于 _emit_once）→ 必须再等告警落地，否则竞态
        assert _wait_until(
            lambda: sum("语句事件回调异常" in m for m in statuses) == 3), (
            f"回调异常未被逐条告警：statuses={statuses}")
        # 线程仍存活：还能继续处理后续批次（此处验证未因异常退出）
        assert r._asr_thread is not None and r._asr_thread.is_alive()
        assert [s.text for s in r.store.segments()] == ["line 1", "line 2", "line 3"]
        assert len(emitted) == 3                        # 每条都尝试发送（不因异常跳过）
        assert not any("识别失败" in m for m in statuses)   # 不是「整批丢弃」
    finally:
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(r.store.segments()) == 3                 # 收尾未丢语句


# ---------------------------------------------------------------------------
# 云端识别（ASR_PROVIDER=groq）：每条语句到达即识别，不等攒批阈值
# ---------------------------------------------------------------------------

def _groq_settings():
    """云端模式 Settings（新增字段带默认值，追加在既有位置参数之后）。"""
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None,
                    groq_api_key="gsk_test", asr_provider="groq")


class FakeCloudASR:
    """云端 ASR：单条语句一次调用，记录每次收到的样本数（批大小证据）。"""

    def __init__(self):
        self.calls = 0
        self.batch_lens: list[int] = []
        self._produced = 0          # 跨批连续编号：文本与「批次怎么切」解耦

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        self.batch_lens.append(len(audio))
        n = max(1, len(audio) // 16000)
        segs = [Segment(i + 0.1, i + 0.9, f"line {self._produced + i + 1}", lang="en")
                for i in range(n)]
        self._produced += n
        return segs


class FakeCaptureGated:
    """每喂一块前先等上一块被识别完（calls 追平已喂块数）。

    没有这道闸门，主循环会把 3 条语句一股脑塞进积压，worker 醒来时取走全部 3 条
    （批大小 48000）—— 那样测的是「取批时机」而不是「云端逐句识别」，且存在竞态。
    """

    def __init__(self, asr: FakeCloudASR, chunks: int = 3, samples: int = 16000):
        self._asr = asr
        self._chunks = [np.zeros(samples, dtype=np.float32)] * chunks
        self._fed = 0
        self._done = threading.Event()

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._fed < len(self._chunks):
            deadline = time.monotonic() + 10.0
            while self._asr.calls < self._fed and time.monotonic() < deadline:
                time.sleep(0.005)      # 等上一条语句被识别（批内恒为 1 条）
            c = self._chunks[self._fed]
            self._fed += 1
            return c
        if self._done.is_set():
            return None                # 收尾哨兵：只有 stop() 之后才出现
        time.sleep(0.005)
        raise queue.Empty()

    def stop(self):
        self._done.set()


def test_groq_provider_processes_each_utterance(monkeypatch):
    """e. provider=groq：每条语句各自触发一次 transcribe（批大小恒为 1 条）。

    阈值故意压到「永不达标」（999s）：只有 _asr_batch_ready 的 groq 分支
    （有积压就出批）能让字幕在录制中途逐条出现。
    变异自查点：去掉 groq 即时出批分支 → 本用例的 calls==3 等待超时失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVADPerChunk)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 999.0)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_MAX_WAIT", 999.0)

    asr = FakeCloudASR()
    capture = FakeCaptureGated(asr, chunks=3)
    r = Recorder(_groq_settings(), capture=capture, asr=asr,
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("云端逐句",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: asr.calls == 3), \
            f"云端模式未逐句识别：calls={asr.calls} batch_lens={asr.batch_lens}"
        assert asr.batch_lens == [16000, 16000, 16000], asr.batch_lens
        assert [s.text for s in r.store.segments()] == ["line 1", "line 2", "line 3"]
        assert r._asr_provider == "groq"
        assert r._stop.is_set() is False          # 逐句出字幕发生在录制中途
        # 缺口补测：FakeCloudASR 每批返回批内相对 (0.1, 0.9)，to_abs 必须按该语句的
        # 绝对样本偏移回填（第 i 条语句偏移 i*16000）→ 0.1/1.1/2.1…，不是三条都在 0.1。
        spans = [(round(s.start, 3), round(s.end, 3)) for s in r.store.segments()]
        assert spans == [(0.1, 0.9), (1.1, 1.9), (2.1, 2.9)], spans
    finally:
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert asr.calls == 3
    assert all(s.translation == "你好世界" for s in r.store.segments())


def test_groq_provider_lazy_init_uses_cloud_engine(monkeypatch):
    """provider=groq 且未注入 asr 时，惰性构造的是 CloudASREngine（不是本地模型）。"""
    built: list[str] = []

    class _FakeCloudEngine:
        def __init__(self, settings):
            built.append("cloud")

        def transcribe(self, audio, sample_rate=16000, should_stop=None):
            n = max(1, len(audio) // 16000)
            return [Segment(i + 0.5, i + 0.9, "cloud ok", lang="en") for i in range(n)]

    monkeypatch.setattr("meeting_bot.recorder.CloudASREngine", _FakeCloudEngine)
    monkeypatch.setattr("meeting_bot.recorder.ASREngine",
                        lambda settings: pytest.fail("groq 模式不得构造本地 ASREngine"))
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)

    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=1),
                 translator=FakeTranslator(), diarizer=None)
    r.on_status(lambda m: None)
    r.start("云端惰性")
    r.stop()

    assert built == ["cloud"]
    assert [s.text for s in r.store.segments()] == ["cloud ok"]


def test_groq_provider_cloud_failure_drops_only_that_batch(monkeypatch):
    """云端失败 → 回退本地；本地也失败时只丢该批 + 告警，识别线程存活。

    回退改造后「云端失败」不再直接丢批（先切本地重试），本用例固定住回退后的
    兜底：本地引擎同样失败 → 丢这一批、告警可见、线程不猝死（后续批次仍能跑）。
    """
    from meeting_bot.asr import CloudASRError

    class FailingCloudASR:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("云端识别不可用（模拟）")

    class FailingLocalASR:
        """回退目标：本地引擎也起不来/也失败（模型未下载、CPU 打满等）。"""

        def __init__(self, settings):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("本地识别不可用（模拟）")

    built: list[FailingLocalASR] = []

    def _make_local(settings):
        eng = FailingLocalASR(settings)
        built.append(eng)
        return eng

    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _make_local)
    asr = FailingCloudASR()
    # FakeCaptureHeld：喂完一块后持续「读超时」，录制不结束 —— 让回退发生在录制
    # 中途（停止信号置位后不再回退，见 test_fallback_skipped_after_stop）。
    r = Recorder(_groq_settings(), capture=FakeCaptureHeld([np.zeros(16000, np.float32)]),
                 asr=asr, translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)
    thread = threading.Thread(target=r.start, args=("云端失败",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: built and built[0].calls >= 1),             f"回退 + 本地重试未发生：cloud={asr.calls} built={len(built)}"
        assert r._stop.is_set() is False               # 回退发生在录制中途
        assert r._asr_thread.is_alive(), "本地也失败不得杀掉识别线程（否则再无字幕）"
    finally:
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()

    assert asr.calls == 1
    assert len(built) == 1 and built[0].calls == 1     # 回退 + 同一批立即重试
    assert r.store.segments() == []                    # 本地也失败 → 该批丢弃
    assert any("识别失败" in m and "本地识别不可用" in m for m in statuses), statuses


def test_batch_thresholds_come_from_settings(monkeypatch):
    """阈值可由 .env（Settings）覆盖；未配置时回落模块常量（既有测试依赖后者）。"""
    import meeting_bot.recorder as rec

    s = Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None,
                 asr_batch_seconds=0.05, asr_batch_max_wait=999.0)
    r = Recorder(s, capture=FakeCapture(chunks=0), asr=FakeASR(), translator=None,
                 diarizer=None)
    r._enqueue_asr(0, np.zeros(16000, dtype=np.float32))   # 1 秒音频 ≥ 0.05s 阈值
    assert r._asr_batch_ready(False) is True

    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_MAX_WAIT", 42.0)
    d = Recorder(_settings(), capture=FakeCapture(chunks=0), asr=FakeASR(),
                 translator=None, diarizer=None)
    assert d._asr_batch_seconds == rec.ASR_BATCH_SECONDS == 0.01
    assert d._asr_batch_max_wait == rec.ASR_BATCH_MAX_WAIT == 42.0


def test_asr_config_error_reported_once(monkeypatch):
    """本地引擎构造持续失败（模型损坏/磁盘满等）：告警只报一次，不能刷满状态栏。

    注意本用例用 provider=local：groq 的配置类错误（缺 key/key 无效）在新实现里
    会先回退本地（见 test_asr_cloud_config_error_falls_back_to_local），不再走
    「本场不再有字幕」的一次性告警；一次性告警保留给「本地也起不来」这种无路可退
    的情形 —— 它是持续性的，每批都会复现。
    """
    from meeting_bot.config import ConfigError

    def _boom(settings):
        raise ConfigError("本地模型不可用（模拟）")

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _boom)
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    r = Recorder(_settings(), capture=FakeCapture(chunks=0),
                 translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)

    for _ in range(3):    # 三批：每批都会再撞一次同样的构造失败
        r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert sum("识别引擎不可用" in m for m in statuses) == 1, statuses
    assert sum("本地模型不可用" in m for m in statuses) == 1, statuses
    assert r.store.segments() == []


def test_asr_cloud_config_error_falls_back_to_local(monkeypatch):
    """M1 + 回退。key 无效/无权限（CloudASRConfigError）也走回退：不整场没字幕。

    这类错误每次识别都必然复现（key 不会自己变好），旧实现干脆整场放弃字幕。
    现在先切本地（本地识别不需要 key），并把「云端不可用」只报一次：
    既不刷满状态栏，也不用让用户重开录制。
    """
    from meeting_bot.asr import CloudASRConfigError

    class _BadKeyCloud:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRConfigError("GROQ_API_KEY 无效或无权限（HTTP 401）")

    class _LocalOk:
        def __init__(self, settings):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            return [Segment(0.5, 0.9, "本地兜底", lang="zh")]

    built: list[_LocalOk] = []

    def _make_local(settings):
        eng = _LocalOk(settings)
        built.append(eng)
        return eng

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _make_local)
    asr = _BadKeyCloud()
    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=0),
                 asr=asr, translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)

    for _ in range(3):    # 三批：回退后全部由本地引擎接手
        r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert asr.calls == 1, "回退后不再调用云端引擎（key 已确定不可用）"
    assert len(built) == 1, "只回退一次（不得每批重建本地模型）"
    assert built[0].calls == 3, "回退后每批都走本地（含回退当场的那次重试）"
    assert sum("已自动切换本地识别" in m for m in statuses) == 1, statuses
    assert sum("识别引擎不可用" in m for m in statuses) == 0, "已回退，不得再说整场没字幕"
    assert sum("识别失败" in m for m in statuses) == 0, "本地成功，不该报批次失败"
    assert [s.text for s in r.store.segments()] == ["本地兜底"] * 3
    assert r._asr_provider == "local"


# ---------------------------------------------------------------------------
# 云端识别失败自动回退本地（不丢句）
# ---------------------------------------------------------------------------

def test_cloud_failure_falls_back_to_local(monkeypatch):
    """Groq 挂掉（限速/断网/key 失效）时不丢句：切本地引擎重跑同一批音频。

    旧实现：云端失败只告警 + 丢弃该批 → 用户整场没有字幕。
    新实现：回退本地并继续，代价是字幕变慢（本地攒批 30s/35s），但内容不丢。

    变异自查点：把 ``_fallback_to_local`` 的调用去掉（回到「丢批」）→ store 恒为空
    → 本用例的 segments 断言失败。
    """
    from meeting_bot.asr import CloudASRError

    class CeilingCloudASR:
        """云端引擎：始终不可用（模拟 Groq 限速/断网/key 失效）。"""

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("云端识别不可用（模拟限速）")

    class LocalASR:
        """回退目标：本地引擎正常，返回 1 条语句。"""

        def __init__(self, settings):
            self.calls = 0
            self.batch_lens: list[int] = []

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            self.batch_lens.append(len(audio))
            return [Segment(0.5, 2.5, "本地识别结果", lang="zh")]

    built: list[LocalASR] = []

    def _make_local(settings):
        eng = LocalASR(settings)
        built.append(eng)
        return eng

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _make_local)
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    cloud = CeilingCloudASR()
    # FakeCaptureHeld：喂完一块后继续「录音中」，回退必然发生在录制中途（未停止）
    r = Recorder(_groq_settings(), capture=FakeCaptureHeld([np.zeros(16000, np.float32)]),
                 asr=cloud, translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)
    thread = threading.Thread(target=r.start, args=("云端挂掉",), daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: r.store.segments()),             f"回退后未出字幕：cloud={cloud.calls} built={len(built)} statuses={statuses}"
        assert r._stop.is_set() is False, "回退应发生在录制中途（停止后不再回退）"
        assert r._asr_thread.is_alive()
    finally:
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()

    assert cloud.calls == 1, "云端先试一次"
    assert len(built) == 1, "必须切到本地引擎"
    assert built[0].calls == 1, "同一批音频要立即用本地重试（不丢句）"
    assert built[0].batch_lens == [16000], "重试的必须是同一批音频（不是新批/空音频）"
    assert [s.text for s in r.store.segments()] == ["本地识别结果"]
    assert any("已自动切换本地识别" in m for m in statuses), statuses
    assert not any("识别失败" in m for m in statuses), "回退成功不得报「本批已丢弃」"
    assert r._asr_provider == "local", "回退后 provider 必须转 local（后续批次走本地攒批）"
    assert r._asr_fallback_used is True
    # 需求 3c：回退后 _asr_batch_ready 必须走本地分支 —— 2 秒积压时云端会立即出批，
    # 本地则要等攒够 30s/35s（阈值在这里是模块默认值）。
    r._enqueue_asr(0, np.zeros(2 * 16000, dtype=np.float32))
    assert r._asr_batch_ready(False) is False, "回退后不得继续按云端「有积压就出批」"


def test_missing_groq_key_falls_back_to_local(monkeypatch):
    """缺 GROQ_API_KEY（云端引擎构造失败）也回退本地：不再「本场没有字幕」。

    这是最常见的翻车路径：用户配了 ``ASR_PROVIDER=groq`` 却没填 key，或 key 被删。
    旧实现每批都在构造时失败 → 只告警一次「识别引擎不可用，本场不再有字幕」，
    整场废掉。现在直接降级本地，会议照常出字幕。
    """
    from meeting_bot.config import ConfigError

    class _MissingKeyCloud:
        """云端引擎构造即失败（模拟 .env 没填 GROQ_API_KEY）。"""

        def __init__(self, settings):
            raise ConfigError("缺少 GROQ_API_KEY")

    class _LocalOK:
        def __init__(self, settings):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            return [Segment(0.5, 0.9, "本地兜底", lang="zh")]

    built: list[_LocalOK] = []

    def _make_local(settings):
        eng = _LocalOK(settings)
        built.append(eng)
        return eng

    monkeypatch.setattr("meeting_bot.recorder.CloudASREngine", _MissingKeyCloud)
    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _make_local)
    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=0),
                 translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)

    for _ in range(2):    # 首批：构造失败 → 回退 + 重试；次批：直接走本地
        r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert len(built) == 1 and built[0].calls == 2
    assert [s.text for s in r.store.segments()] == ["本地兜底"] * 2
    assert sum("已自动切换本地识别" in m for m in statuses) == 1, statuses
    assert not any("识别引擎不可用" in m for m in statuses), "已回退，不得再说整场没字幕"
    assert r._asr_provider == "local"


def test_fallback_skipped_after_stop(monkeypatch):
    """已收到停止信号时云端失败不回退：不得在关闭路径里加载数百 MB 本地模型。

    回归场景：用户点「结束」时正好赶上云端 429/断网，或退避睡眠被停止信号打断
    （asr.py 抛 CloudASRError("识别被停止")）。此时若照常回退，会在 stop() 等待
    识别线程收尾的窗口里去加载/下载本地模型 → 「点结束」变界面卡死，且状态栏
    给出「云端识别不可用」这种误诊。这类失败按普通丢批处理即可。
    """
    from meeting_bot.asr import CloudASRError

    class _Cloud:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("识别被停止")

    def _must_not_build(settings):
        pytest.fail("停止后不得构造本地引擎（关闭路径里加载模型会卡死界面）")

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _must_not_build)
    cloud = _Cloud()
    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=0), asr=cloud,
                 translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)

    r._stop.set()                                  # 模拟 stop() 已置位
    r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert cloud.calls == 1
    assert r._asr_provider == "groq", "停止后不回退"
    assert r._asr_fallback_used is False
    assert sum("已自动切换本地识别" in m for m in statuses) == 0, statuses
    assert any("识别失败" in m for m in statuses), "仍要告警「本批已丢弃」"
    assert r.store.segments() == []


def test_fallback_local_engine_failure_reported_once(monkeypatch):
    """回退后本地引擎也起不来：只报一次「本场不再有字幕」，不把原因归到云端 key。

    回归点：本地构造失败分支若不留一次性告警标记，调用方会拿云端那条异常再报一遍
    「识别引擎不可用：缺少 GROQ_API_KEY」—— 原因归错（真正拦住用户的是本地模型），
    且同一结论在状态栏刷两条。
    """
    from meeting_bot.asr import CloudASRError
    from meeting_bot.config import ConfigError

    class _Cloud:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("云端识别不可用（模拟限速）")

    built: list[str] = []

    def _broken_local(settings):
        built.append("try")
        raise ConfigError("本地模型文件损坏（模拟）")

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _broken_local)
    cloud = _Cloud()
    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=0), asr=cloud,
                 translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)

    for _ in range(2):    # 两批：第二批会再试一次构造（不放弃恢复机会）
        r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert built == ["try", "try"], "每批仍应重试本地构造（模型可能后补/磁盘腾出）"
    assert sum("已自动切换本地识别" in m for m in statuses) == 1, statuses
    assert sum("本地识别引擎无法启动" in m for m in statuses) == 1, statuses
    assert sum("识别引擎不可用" in m for m in statuses) == 0,         f"不得再拿云端异常报一次（归因错）：{statuses}"
    assert r._asr is None, "本地引擎构造失败时必须清掉（否则继续拿云端引擎打 Groq）"
    assert r._asr_provider == "local"
    assert r.store.segments() == []


def test_fallback_only_once(monkeypatch):
    """回退只发生一次：本地也失败时每批只告警丢弃，不重建引擎、不再回退。

    为什么必须限一次：云端挂掉时每批都会失败，若每次都重建本地引擎，就要反复付
    WhisperModel 的加载开销；若「回退后再失败」仍重试同一批音频，识别线程会卡死
    在这一批上无限循环。

    计数说明：断言的是本地引擎的**构造**次数 == 1（= 回退次数）。transcribe 调用
    次数等于批次数是设计使然 —— 回退后所有批次本就都该走本地。
    """
    from meeting_bot.asr import CloudASRError

    class FailingCloudASR:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("云端识别不可用（模拟限速）")

    class FailingLocalASR:
        def __init__(self, settings):
            self.calls = 0

        def transcribe(self, audio, sample_rate=16000):
            self.calls += 1
            raise CloudASRError("本地识别也不可用（模拟）")

    built: list[FailingLocalASR] = []

    def _make_local(settings):
        eng = FailingLocalASR(settings)
        built.append(eng)
        return eng

    monkeypatch.setattr("meeting_bot.recorder.ASREngine", _make_local)
    cloud = FailingCloudASR()
    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=0), asr=cloud,
                 translator=None, diarizer=None)
    statuses: list[str] = []
    r.on_status(statuses.append)

    for _ in range(3):    # 三批：云端首批失败 → 回退；之后每批只告警
        r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert cloud.calls == 1, "回退后不再调用云端引擎"
    assert len(built) == 1, "回退只发生一次（不得每批重建本地引擎）"
    assert built[0].calls == 3, "回退后每批仍交给本地（含回退当场的那次重试）"
    assert sum("已自动切换本地识别" in m for m in statuses) == 1, statuses
    assert sum("识别失败" in m for m in statuses) == 3, f"每批各告警一次：{statuses}"
    assert r.store.segments() == []
    assert r._asr_provider == "local"


def test_groq_transcribe_receives_should_stop(monkeypatch):
    """M2. 云端引擎拿到 should_stop（= _stop.is_set 的实时读数），退避才可被打断。

    直接驱动 _transcribe_batch（不经线程），避免「首批在 stop 之后才出批」的时序
    竞态把断言变成偶然；两次调用分别对应录制中（False）与停止后（True）。
    """
    seen: list[bool | None] = []

    class _Engine:
        def __init__(self, settings):
            pass

        def transcribe(self, audio, sample_rate=16000, should_stop=None):
            seen.append(should_stop() if should_stop else None)
            return [Segment(0.0, 0.5, "ok", lang="en")]

    monkeypatch.setattr("meeting_bot.recorder.CloudASREngine", _Engine)
    r = Recorder(_groq_settings(), capture=FakeCapture(chunks=0),
                 translator=None, diarizer=None)
    r.on_status(lambda m: None)

    r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])
    r._stop.set()                                   # 模拟 stop() 置位
    r._transcribe_batch([(0, np.zeros(16000, dtype=np.float32))])

    assert seen == [False, True],         f"should_stop 必须是 _stop.is_set 的实时读数，实际={seen}"


def test_local_asr_transcribe_not_given_should_stop():
    """本地 ASREngine 的 transcribe 没有 should_stop 参数：绝不能多传（会 TypeError）。

    引擎签名故意写成严格两参：多传一个参数就抛 TypeError，从而证明 Recorder
    是按引擎类型分支传参，而不是无脑透传。
    """
    class _StrictLocal:
        def transcribe(self, audio, sample_rate=16000):
            return [Segment(0.0, 0.5, "local ok", lang="en")]

    r = Recorder(_settings(), capture=FakeCapture(chunks=0),
                 asr=_StrictLocal(), translator=None, diarizer=None)
    r.on_status(lambda m: None)

    segs = r._transcribe(np.zeros(16000, dtype=np.float32))
    assert [s.text for s in segs] == ["local ok"]


# ---------------------------------------------------------------------------
# 两阶段字幕显示：原文即时上屏（emit）+ 译文原地更新（on_update）
# ---------------------------------------------------------------------------

class FakeTranslatorGated:
    """translate 阻塞到测试放行（把「原文已上屏 / 译文还没到」的时刻固定住）。"""

    def __init__(self):
        self.calls = 0
        self.started = threading.Event()      # 已进入第一次 translate
        self.release = threading.Event()      # 测试放行：translate 此刻才返回

    translate_stream = _stream_via_translate

    def translate(self, text, lang="", context=""):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=10), "测试未放行翻译"
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


def test_original_emitted_before_translation(monkeypatch):
    """两阶段显示：原文事件先到（translation 为空），放行后才收到带译文的 update。

    用户反馈的「字幕延迟明显」= 旧实现把 emit 压在翻译之后（同一条语句要等接口
    3~6 秒才出现在面板上）。修复后 store.add 之后立即 emit 原文，译文完成再走
    on_update 原地替换。

    变异自查点：把 ``_transcribe_batch`` 里的立即 emit 去掉（回到「翻译后再 emit」）
    → 翻译被卡住时 events 恒为空 → 本用例超时失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)   # 有语句就出批
    tr = FakeTranslatorGated()
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    events, updates = [], []
    r.on_segment(events.append)
    r.on_update(updates.append)
    r.on_status(lambda m: None)

    thread = threading.Thread(target=r.start, args=("两阶段会议",), daemon=True)
    thread.start()
    try:
        assert tr.started.wait(timeout=5), "翻译 worker 未开始翻译（用例前提不成立）"
        # 翻译还卡在接口里：原文必须已经上屏
        assert _wait_until(lambda: len(events) == 1), "翻译返回前原文事件就应已投递"
        assert events[0].text == "hello world"
        assert events[0].translation == ""        # 第一阶段：无译文 → 面板显示原文
        assert updates == [], "译文尚未产生，不得提前通知 update"
        assert r._stop.is_set() is False          # 确实发生在录制中途（不是 stop 兜底）

        tr.release.set()                          # 放行翻译
        assert _wait_until(lambda: len(updates) == 1), "译文填充后未通知 update"
        assert updates[0].seq == events[0].seq
        assert updates[0].translation == "你好世界"
    finally:
        tr.release.set()
        r.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()

    # 原文事件只发一次（译文到达不得再 emit 一遍原文）
    assert [s.seq for s in events] == [updates[0].seq]
    assert len({s.seq for s in events}) == 1
    assert r.store.segments()[0].translation == "你好世界"


def test_stop_backfill_notifies_update(monkeypatch):
    """翻译失败段在 stop() 补译成功后必须补一次 update（面板才拿到最终译文）。

    失败时原文已在屏上（第一阶段）→ 不通知；补译成功后必须通知，否则字幕永远停在
    原文、而逐字稿里已是译文（用户看到的两份产物不一致）。

    变异自查点：删掉 stop() 补译分支里的 ``_notify_update`` → updates 恒为空 → 失败。
    """
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    # 门槛压到「有语句就出批」：失败必须发生在 stop() 之前（否则失败与补译挤在同一次
    # stop() 里，「失败阶段不通知」就没被真正验证）
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    tr = FakeTranslatorFailOnce()        # 第 1 次（worker）抛 TranslationError，第 2 次成功
    r = Recorder(_settings(), capture=FakeCapture(chunks=1), asr=FakeASR(),
                 translator=tr, diarizer=None)
    events, updates, statuses = [], [], []
    r.on_segment(events.append)
    r.on_update(updates.append)
    r.on_status(statuses.append)

    r.start("补译通知会议")
    # worker 失败阶段：原文已上屏，且不得有 update（等「翻译失败」告警落地再断言，避免竞态）
    assert _wait_until(lambda: len(events) == 1), "原文事件未投递"
    assert events[0].translation == ""
    assert _wait_until(lambda: any("翻译失败" in m for m in statuses)), "worker 未走到失败分支"
    assert updates == [], "翻译失败不得通知 update（原文已在屏上）"

    r.stop()                             # 补译成功 → 通知 update
    segs = r.store.segments()
    assert segs[0].translation == "你好世界"
    assert tr.calls == 2                 # 1 次失败 + 1 次补译成功
    assert [s.seq for s in updates] == [segs[0].seq], f"补译成功未通知面板：{updates}"
    assert updates[0].translation == "你好世界"
    assert len(events) == 1              # 补译不重复发原文事件


# ---------------------------------------------------------------------------
# SAVE_AUDIO：录制落盘整场 wav（会后离线说话人标注的输入）
# ---------------------------------------------------------------------------

def _audio_settings(tmp_path, **kwargs):
    """带真实 output_dir 的 Settings（_settings() 的 output_dir 是 None，不落盘）。"""
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "",
                    tmp_path / "会议记录", tmp_path, **kwargs)


def test_save_audio_writes_whole_wav(tmp_path, monkeypatch):
    """SAVE_AUDIO=true：每块音频追加写入 <output_dir>/.recording/*.wav。

    2 块 × 8000 样本 = 1 秒；断言 wav 真实存在、16kHz / 单声道、时长约 1 秒 ——
    这三项正是会后 pyannote 读文件的前提（采样率错 → 时间轴全错，声道错 → 只标到半边）。
    """
    import soundfile as sf

    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    r = Recorder(_audio_settings(tmp_path, save_audio=True),
                 capture=FakeCapture(chunks=2), asr=FakeASR(), translator=None)

    r.start("录音落盘会议")
    r.stop()

    path = r.audio_path
    assert path is not None, "SAVE_AUDIO=true 却没有录音文件路径"
    assert path.is_file(), f"录音文件未生成：{path}"
    assert path.parent.name == ".recording"          # git 忽略目录，与 JSONL 日志同处
    assert path.suffix == ".wav"
    info = sf.info(str(path))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert abs(info.frames / 16000 - 1.0) < 0.05, f"时长异常：{info.frames} 样本"
    # stop() 之后文件必须已封口（会后标注线程才能读全）：能再完整读回 1 秒
    audio, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000 and len(audio) == info.frames


def test_save_audio_disabled_writes_nothing(tmp_path, monkeypatch):
    """SAVE_AUDIO=false：完全不落盘（省磁盘），audio_path 为 None。"""
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    r = Recorder(_audio_settings(tmp_path, save_audio=False),
                 capture=FakeCapture(chunks=2), asr=FakeASR(), translator=None)

    r.start("不落盘会议")
    r.stop()

    assert r.audio_path is None
    rec_dir = tmp_path / "会议记录" / ".recording"
    assert not rec_dir.exists() or not list(rec_dir.glob("*.wav"))


def test_save_audio_failure_warns_once_and_keeps_recording(tmp_path, monkeypatch):
    """落盘失败（目录建不出来）只告警一次，录制/识别照常完成（不能反噬主流程）。"""
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.ASR_BATCH_SECONDS", 0.01)
    # output_dir 是个「文件」的路径 → .recording 子目录必然建不出来
    blocker = tmp_path / "不是目录"
    blocker.write_text("x", encoding="utf-8")
    settings = Settings("https://x", "sk-test", "m1", "m2", "medium", "",
                        blocker, tmp_path, save_audio=True)

    r = Recorder(settings, capture=FakeCapture(chunks=2), asr=FakeASR(),
                 translator=None)
    statuses = []
    r.on_status(statuses.append)

    r.start("落盘失败会议")
    r.stop()

    assert r.audio_path is None
    warns = [m for m in statuses if "无法保存录音文件" in m]
    assert len(warns) == 1, f"落盘失败应只告警一次：{statuses}"
    # 录制本身不受影响：语句照常入库
    assert len(r.store.segments()) >= 1
