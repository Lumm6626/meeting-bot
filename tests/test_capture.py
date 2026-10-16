# -*- coding: utf-8 -*-
"""AudioCapture 单测：用假设备桩覆盖背压丢弃、哨兵投递、正常路径。

不依赖真实声卡：monkeypatch ``meeting_bot.capture.sc`` 的 ``default_speaker`` /
``get_microphone``，让 ``AudioCapture`` 拿到假麦克风（fake recorder 是上下文
管理器，``record(frames)`` 返回 ``np.zeros(frames)`` 或按需抛异常）。
"""
from __future__ import annotations

import queue
import time

import numpy as np
import pytest

from meeting_bot import capture as capture_mod
from meeting_bot.capture import AudioCapture

FRAMES = int(16000 * capture_mod.CHUNK_SECONDS)  # 8000


class _FakeSpeaker:
    name = "fake-speaker"


class _FastRecorder:
    """高速产出（约 1000 块/秒）的假录音器，用于制造背压。"""

    def __init__(self, delay: float = 0.001):
        self.calls = 0
        self.delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def record(self, frames):
        self.calls += 1
        time.sleep(self.delay)
        return np.zeros(frames)


class _FailingRecorder:
    """上下文管理器进入即失败（模拟设备被占用/设备热拔）。"""

    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True
        raise RuntimeError("fake device open failure")

    def __exit__(self, *exc_info):
        return False

    def record(self, frames):  # pragma: no cover - 不应被调用
        raise AssertionError("record() 不应在 __enter__ 失败后被调用")


def _install_fake_device(monkeypatch, recorder):
    """把 sc.default_speaker / sc.get_microphone 换成返回假设备的桩。"""

    class _FakeMic:
        def recorder(self, samplerate=16000, channels=1):
            return recorder

    monkeypatch.setattr(capture_mod.sc, "default_speaker", lambda: _FakeSpeaker())
    monkeypatch.setattr(
        capture_mod.sc,
        "get_microphone",
        lambda id=None, include_loopback=False: _FakeMic(),
    )


def test_backpressure_drops_instead_of_blocking(monkeypatch):
    """消费端不读时，生产者不得阻塞：调用数持续增长且队列稳定在上限。"""
    rec = _FastRecorder()
    _install_fake_device(monkeypatch, rec)
    cap = AudioCapture()
    cap.start()
    try:
        time.sleep(3.0)
        calls_3s = rec.calls
        qsize = cap._queue.qsize()
        # 再等 1 秒，确认生产者仍在推进（没有卡在 put 上）
        time.sleep(1.0)
        calls_4s = rec.calls
    finally:
        cap.stop()

    assert qsize <= 100, f"队列超出上限：qsize={qsize}"
    assert calls_3s > 500, f"3 秒内 record 仅被调用 {calls_3s} 次，疑为阻塞"
    assert calls_4s > calls_3s, f"生产者冻结：3s={calls_3s} 4s={calls_4s}"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_sentinel_delivered_when_recorder_open_fails(monkeypatch):
    """录音器打开失败时，read() 也必须在超时前收到 None 哨兵。

    打开失败的异常仍会抛出到线程外（保留诊断信息，不静默吞掉），
    因此这里仅过滤这一条预期内的未处理线程异常告警；哨兵投递由 finally 保证。
    """
    rec = _FailingRecorder()
    _install_fake_device(monkeypatch, rec)
    cap = AudioCapture()
    cap.start()
    try:
        t0 = time.monotonic()
        result = cap.read(timeout=2.0)
        elapsed = time.monotonic() - t0
    finally:
        cap.stop()

    assert rec.entered, "fake recorder 未被进入"
    assert result is None, "录音器打开失败后未投递结束哨兵"
    assert elapsed < 2.0, f"哨兵等待超时（{elapsed:.2f}s）"


def test_normal_path_reads_three_chunks_then_sentinel(monkeypatch):
    """正常路径：读到 3 块 float32 音频，stop() 后哨兵可达。"""
    rec = _FastRecorder()
    _install_fake_device(monkeypatch, rec)
    cap = AudioCapture()
    cap.start()
    chunks = []
    try:
        for _ in range(3):
            chunk = cap.read(timeout=2.0)
            assert chunk is not None
            chunks.append(chunk)
    finally:
        cap.stop()

    assert len(chunks) == 3
    for chunk in chunks:
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.float32
        assert chunk.shape == (FRAMES,)

    # stop() 之后哨兵最终可达（队列里可能还有积压块）
    saw_none = False
    for _ in range(200):
        try:
            item = cap._queue.get(timeout=0.5)
        except queue.Empty:
            break
        if item is None:
            saw_none = True
            break
    assert saw_none, "stop() 后未收到结束哨兵"


def test_read_timeout_raises_queue_empty(monkeypatch):
    """契约：超时抛 queue.Empty（与「采集结束返回 None」区分）。"""
    _install_fake_device(monkeypatch, _FastRecorder())
    cap = AudioCapture()  # 未 start()，队列始终为空
    try:
        cap.read(timeout=0.2)
    except queue.Empty:
        pass
    else:  # pragma: no cover - 契约被破坏时触发
        raise AssertionError("read() 超时应抛 queue.Empty")


class _MidStreamFailingRecorder:
    """录到第 N 块后抛错（模拟 WASAPI 流中断/设备热拔）。"""

    def __init__(self, fail_after: int = 2):
        self.fail_after = fail_after
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def record(self, frames):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("fake stream broken")
        return np.zeros(frames)


def test_mid_stream_error_is_recorded_not_swallowed(monkeypatch):
    """流中断必须记进 cap.error：否则上层只看到「正常结束哨兵」，音频静默截断。"""
    rec = _MidStreamFailingRecorder(fail_after=2)
    _install_fake_device(monkeypatch, rec)
    cap = AudioCapture()
    cap.start()
    chunks = []
    try:
        for _ in range(6):
            item = cap.read(timeout=2.0)
            if item is None:
                break
            chunks.append(item)
    finally:
        cap.stop()

    assert len(chunks) == 2, f"应读到中断前的 2 块，实际 {len(chunks)}"
    assert isinstance(cap.error, RuntimeError)
    assert "stream broken" in str(cap.error)


def test_dropped_chunks_counted_when_queue_full(monkeypatch):
    """队列满时丢块要计数（背压可观测），不再静默 pass。"""
    rec = _FastRecorder(delay=0.0)
    _install_fake_device(monkeypatch, rec)
    cap = AudioCapture()
    cap.start()
    try:
        time.sleep(2.0)
        dropped = cap.dropped_chunks
    finally:
        cap.stop()

    assert cap._queue.qsize() <= 100
    assert dropped > 0, "队列满丢块未被计数"
