# -*- coding: utf-8 -*-
"""AudioCapture 单测：假设备桩覆盖背压丢弃、哨兵投递、正常路径、双路混合。

不依赖真实声卡：monkeypatch ``meeting_bot.capture.sc`` 的 ``default_speaker`` /
``get_microphone``，让 ``AudioCapture`` 拿到假麦克风（fake recorder 是上下文
管理器，``record(frames)`` 返回 ``np.zeros(frames)`` 或按需抛异常）。
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from meeting_bot import capture as capture_mod
from meeting_bot.capture import MIX_GAIN, AudioCapture

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
    cap = AudioCapture(source="speaker")
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
    cap = AudioCapture(source="speaker")
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
    cap = AudioCapture(source="speaker")
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
    cap = AudioCapture(source="speaker")  # 未 start()，队列始终为空
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
    cap = AudioCapture(source="speaker")
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
    cap = AudioCapture(source="speaker")
    cap.start()
    try:
        time.sleep(2.0)
        dropped = cap.dropped_chunks
    finally:
        cap.stop()

    assert cap._queue.qsize() <= 100
    assert dropped > 0, "队列满丢块未被计数"


# ---------------------------------------------------------------------------
# 音频源选择：speaker（loopback）/ mic（默认麦克风）
# ---------------------------------------------------------------------------

def _install_source_spies(monkeypatch, device):
    """把 sc 的三个设备入口换成记录调用的桩，返回 calls 列表。"""
    calls: list[tuple[str, dict]] = []

    def _speaker():
        calls.append(("default_speaker", {}))
        return _FakeSpeaker()

    def _mic():
        calls.append(("default_microphone", {}))
        return device

    def _get_mic(id=None, include_loopback=False):
        calls.append(("get_microphone", {"id": id, "include_loopback": include_loopback}))
        return device

    monkeypatch.setattr(capture_mod.sc, "default_speaker", _speaker)
    monkeypatch.setattr(capture_mod.sc, "default_microphone", _mic)
    monkeypatch.setattr(capture_mod.sc, "get_microphone", _get_mic)
    return calls


class _FakeDevice:
    """带 recorder() 的假设备（麦克风分支走 default_microphone 直接返回）。"""

    def __init__(self, recorder_factory):
        self._factory = recorder_factory

    def recorder(self, samplerate=16000, channels=1):
        return self._factory()


def test_mic_source_uses_default_microphone(monkeypatch):
    """source="mic" 只走 default_microphone：不带 loopback，不碰 get_microphone。

    变异自查点：把 mic 分支写回 default_speaker + include_loopback=True →
    default_microphone 不被调用、get_microphone 被调用 → 本用例失败。
    """
    calls = _install_source_spies(monkeypatch, _FakeDevice(lambda: _FastRecorder()))
    cap = AudioCapture(source="mic")

    names = [c[0] for c in calls]
    assert names == ["default_microphone"], f"麦克风模式设备调用序列异常：{names}"
    assert cap.source == "mic"
    assert cap.sample_rate == 16000


def test_mic_source_actually_records_from_microphone(monkeypatch):
    """麦克风模式同样跑通「启动 → 读块 → 停止」链路（复用同一队列/哨兵逻辑）。"""
    rec = _FastRecorder()
    _install_source_spies(monkeypatch, _FakeDevice(lambda: rec))
    cap = AudioCapture(source="mic")
    cap.start()
    chunks = []
    try:
        for _ in range(2):
            chunk = cap.read(timeout=2.0)
            assert chunk is not None
            chunks.append(chunk)
    finally:
        cap.stop()

    assert rec.calls >= 2, "麦克风假设备未被 record()，采集线程没走 _device"
    assert all(c.shape == (FRAMES,) and c.dtype == np.float32 for c in chunks)


def test_speaker_source_keeps_loopback_path(monkeypatch):
    """source="speaker" 仍然：default_speaker → get_microphone(loopback=True)。"""
    calls = _install_source_spies(monkeypatch, _FakeDevice(lambda: _FastRecorder()))
    cap = AudioCapture(source="speaker")   # 默认已改为 both，此处显式指定
    assert cap.source == "speaker"

    names = [c[0] for c in calls]
    assert names == ["default_speaker", "get_microphone"], f"扬声器模式调用序列异常：{names}"
    kwargs = calls[1][1]
    assert kwargs["include_loopback"] is True
    assert kwargs["id"] == "fake-speaker"


@pytest.mark.parametrize("bad", ["microphone", "MIC", "", "loopback"])
def test_invalid_source_raises_capture_error(monkeypatch, bad):
    """非法 source 在碰设备之前就抛 CaptureError（早失败、无副作用）。"""
    calls = _install_source_spies(monkeypatch, _FakeDevice(lambda: _FastRecorder()))
    with pytest.raises(capture_mod.CaptureError, match="未知音频源"):
        AudioCapture(source=bad)
    assert calls == [], f"非法 source 不应探测设备，却发生了调用：{calls}"


def test_device_resolution_failure_wrapped_as_capture_error(monkeypatch):
    """设备解析抛异常（无麦克风 / 无 WASAPI）统一包成 CaptureError，供 UI 弹框。"""
    _install_source_spies(monkeypatch, _FakeDevice(lambda: _FastRecorder()))

    def _boom():
        raise RuntimeError("no such device")

    monkeypatch.setattr(capture_mod.sc, "default_microphone", _boom)
    with pytest.raises(capture_mod.CaptureError, match="麦克风"):
        AudioCapture(source="mic")


# ---------------------------------------------------------------------------
# 自动双路（source="both"）：默认模式，扬声器 loopback + 麦克风混合成一路
# ---------------------------------------------------------------------------

class _ValueRecorder:
    """按固定值产出音频的假录音器：用不同值区分两路（0.1 / 0.2 → 混合 0.3）。

    ``fail_after``：产出 N 块后抛错（模拟该路设备断开）。
    """

    def __init__(self, value=0.0, delay=0.002, fail_after=None):
        self.value = value
        self.delay = delay
        self.fail_after = fail_after
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def record(self, frames):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("fake side failure")
        if self.delay:
            time.sleep(self.delay)
        return np.full(frames, self.value, dtype=np.float32)


def _install_both_devices(monkeypatch, loopback, mic):
    """两路假设备：loopback 走 default_speaker→get_microphone，麦克风走 default_microphone。"""
    calls: list[str] = []

    def _speaker():
        calls.append("default_speaker")
        return _FakeSpeaker()

    def _get_mic(id=None, include_loopback=False):
        calls.append("get_microphone")
        return loopback

    def _default_mic():
        calls.append("default_microphone")
        return mic

    monkeypatch.setattr(capture_mod.sc, "default_speaker", _speaker)
    monkeypatch.setattr(capture_mod.sc, "get_microphone", _get_mic)
    monkeypatch.setattr(capture_mod.sc, "default_microphone", _default_mic)
    return calls


def test_both_is_default_and_opens_two_devices(monkeypatch):
    """不传 source 即自动模式：同时解析 loopback 与默认麦克风，不碰单路分支。"""
    calls = _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    cap = AudioCapture()
    assert cap.source == "both"
    assert calls == ["default_speaker", "get_microphone", "default_microphone"]
    assert len(cap._side_queues) == 2, "双路模式必须有两路中间队列"


def test_both_mixes_two_sources_into_single_stream(monkeypatch):
    """两路各出 0.1 / 0.2 → 输出块逐样本 0.3，长度 = CHUNK（8000 样本 @16k）。"""
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    cap = AudioCapture()
    cap.start()
    chunks = []
    try:
        for _ in range(3):
            chunk = cap.read(timeout=3.0)
            assert chunk is not None, "双路混合未产出音频"
            chunks.append(chunk)
    finally:
        cap.stop()

    for chunk in chunks:
        assert chunk.dtype == np.float32
        assert chunk.shape == (FRAMES,), f"块长应为 {FRAMES}，实际 {chunk.shape}"
        assert np.allclose(chunk, 0.3 * MIX_GAIN), f"混合结果应为 (0.1+0.2)*{MIX_GAIN}，实际 {chunk[:5]}"


def test_both_clips_when_sum_exceeds_full_scale(monkeypatch):
    """两路都很响（0.8 + 0.8）时必须 clip 到 [-1, 1]，不能溢出削波爆音。"""
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.8)),
        _FakeDevice(lambda: _ValueRecorder(0.8)),
    )
    cap = AudioCapture()
    cap.start()
    try:
        chunk = cap.read(timeout=3.0)
    finally:
        cap.stop()

    assert chunk is not None
    assert np.allclose(chunk, 1.0), f"clip 未生效：max={chunk.max()} min={chunk.min()}"
    assert chunk.max() <= 1.0 and chunk.min() >= -1.0


def test_both_sentinel_propagates_after_stop(monkeypatch):
    """双路模式 stop() 后消费者仍必须收到结束哨兵（否则 read 永久阻塞）。"""
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    cap = AudioCapture()
    cap.start()
    try:
        assert cap.read(timeout=3.0) is not None
    finally:
        cap.stop()

    saw_none = False
    for _ in range(200):
        try:
            item = cap.read(timeout=0.5)
        except queue.Empty:
            break
        if item is None:
            saw_none = True
            break
    assert saw_none, "双路模式 stop() 后未收到结束哨兵"
    assert not any(t.is_alive() for t in cap._threads), "stop() 后仍有采集/混合线程存活"


def test_both_one_side_dies_passthrough(monkeypatch):
    """一路结束（哨兵）后另一路直通：不再相加，也不会把整场录制拖死。

    直接驱动混合线程（白盒但确定性）：先各喂一块 → 0.1+0.2=0.3；
    再给 loopback 投哨兵 → 后续麦克风块原样 0.2 输出；两路都结束 → 输出哨兵。
    """
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    cap = AudioCapture()
    cap._running.set()
    cap._side_queues[0].put(np.full(FRAMES, 0.1, dtype=np.float32))
    cap._side_queues[1].put(np.full(FRAMES, 0.2, dtype=np.float32))
    mixer = threading.Thread(target=cap._mix_run, daemon=True)
    mixer.start()
    try:
        first = cap.read(timeout=3.0)
        assert first is not None and np.allclose(first, 0.3 * MIX_GAIN), "两路都在时应混合输出（含防削波缩放）"

        cap._side_queues[0].put(None)          # loopback 结束哨兵
        cap._side_queues[1].put(np.full(FRAMES, 0.2, dtype=np.float32))
        second = cap.read(timeout=3.0)
        assert second is not None and np.allclose(second, 0.2), "一路结束后应直通另一路"

        cap._side_queues[1].put(None)          # 第二路也结束
        assert cap.read(timeout=3.0) is None, "两路都结束应投递输出哨兵"
    finally:
        mixer.join(timeout=3.0)
    assert not mixer.is_alive()


def test_both_side_error_is_recorded(monkeypatch):
    """某一路采集抛错 → 聚合到 self.error（保留第一个错误），另一路继续出音频。"""
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2, fail_after=0)),   # 麦克风开局即断
    )
    cap = AudioCapture()
    cap.start()
    chunks = []
    try:
        for _ in range(3):
            chunk = cap.read(timeout=3.0)
            if chunk is None:
                break
            chunks.append(chunk)
    finally:
        cap.stop()

    assert isinstance(cap.error, RuntimeError), f"该路错误未聚合：{cap.error!r}"
    assert "fake side failure" in str(cap.error)
    assert chunks, "一路断掉后另一路必须仍在产出"
    for chunk in chunks:
        assert chunk.shape == (FRAMES,)


def test_both_side_buffer_capped_on_drift(monkeypatch):
    """单边缓冲超过 3 块即丢最旧数据（时钟漂移防御），并计入 dropped_chunks。"""
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    cap = AudioCapture()
    for _ in range(10):        # 单边猛塞 10 块，另一边一块没有
        cap._append(0, np.full(FRAMES, 0.1, dtype=np.float32))

    assert cap._side_buffers[0].size == FRAMES * capture_mod.MIX_MAX_BUFFERED_CHUNKS
    assert cap._side_buffers[0].size > 0
    assert cap.dropped_chunks > 0, "漂移丢弃未被计数"


def test_both_requires_both_devices(monkeypatch):
    """自动模式要求两路都可用：麦克风打不开 → 早失败 + 提示可用 AUDIO_SOURCE 单路。"""
    calls = _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    calls.clear()

    def _boom():
        calls.append("default_microphone")
        raise RuntimeError("no such mic")

    monkeypatch.setattr(capture_mod.sc, "default_microphone", _boom)
    with pytest.raises(capture_mod.CaptureError) as ei:
        AudioCapture()
    msg = str(ei.value)
    assert "麦克风" in msg and "AUDIO_SOURCE" in msg, msg

    # 单路仍可用：显式 speaker 不受麦克风缺失影响
    cap = AudioCapture(source="speaker")
    assert cap.source == "speaker"


def test_both_idle_side_bypasses_without_killing(monkeypatch):
    """一路饿死不出数据（如无播放时 loopback）→ 本轮先直通另一路，但不判死。

    关键：**不能**把闲置路永久标记失效 —— 线上会议刚开录时对方还没说话，
    loopback 饿死几秒后对方开口，那一路必须还能接回来，否则整场丢掉对方声音。
    """
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    monkeypatch.setattr("meeting_bot.capture.MIX_IDLE_GRACE", 0.3)
    monkeypatch.setattr("meeting_bot.capture.MIX_IDLE_REPORT", 0.3)
    cap = AudioCapture()
    cap._running.set()
    cap._side_last_data[0] = time.time() - 10.0   # loopback 已闲置 10s
    cap._side_last_data[1] = time.time()
    cap._side_queues[1].put(np.full(FRAMES, 0.2, dtype=np.float32))
    mixer = threading.Thread(target=cap._mix_run, daemon=True)
    mixer.start()
    try:
        out = cap.read(timeout=3.0)
        assert out is not None and np.allclose(out, 0.2), "闲置路时期另一路应直通"
        assert cap._side_alive[0] is True, "闲置路不得判死（对方开口后要能接回）"
        assert cap.error is not None, "长时间无数据应留一条诊断记录"
        assert "系统声音" in str(cap.error), str(cap.error)
    finally:
        cap._side_queues[1].put(None)
        cap._side_queues[0].put(None)
        mixer.join(timeout=3.0)


def test_both_idle_side_rejoins_mixing(monkeypatch):
    """饿死的那路恢复出数据后，双路混音自动恢复（不永久丢声音）。"""
    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _ValueRecorder(0.1)),
        _FakeDevice(lambda: _ValueRecorder(0.2)),
    )
    monkeypatch.setattr("meeting_bot.capture.MIX_IDLE_GRACE", 0.3)
    monkeypatch.setattr("meeting_bot.capture.MIX_IDLE_REPORT", 30.0)
    cap = AudioCapture()
    cap._running.set()
    cap._side_last_data[0] = time.time() - 10.0
    cap._side_last_data[1] = time.time()
    cap._side_queues[1].put(np.full(FRAMES, 0.2, dtype=np.float32))
    mixer = threading.Thread(target=cap._mix_run, daemon=True)
    mixer.start()
    try:
        first = cap.read(timeout=3.0)
        assert first is not None and np.allclose(first, 0.2), "闲置期应直通"

        # loopback 恢复出数据（先喂闲置路，避免竞态）→ 下一块应重新变成 0.1+0.2
        cap._side_queues[0].put(np.full(FRAMES, 0.1, dtype=np.float32))
        cap._side_queues[1].put(np.full(FRAMES, 0.2, dtype=np.float32))
        second = cap.read(timeout=3.0)
        assert second is not None and np.allclose(second, 0.3 * MIX_GAIN), "恢复后应重新双路混音"
    finally:
        cap._side_queues[0].put(None)
        cap._side_queues[1].put(None)
        mixer.join(timeout=3.0)


def test_mic_auto_gain_boosts_quiet_speech(monkeypatch):
    """远场小声说话：自动增益把电平拉到可识别范围（约 +x 倍），且不削波。"""
    from meeting_bot.capture import MIC_MAX_GAIN, MIC_TARGET_RMS, _AutoGain
    agc = _AutoGain()
    quiet = np.full(FRAMES, 0.005, dtype=np.float32)      # rms 0.005（远场）
    out = None
    for _ in range(5):
        out = agc.process(quiet)
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert rms > 0.02, f"增益后 rms 仍过低：{rms:.4f}"
    assert agc.gain <= MIC_MAX_GAIN + 1e-6
    assert np.abs(out).max() <= 1.0
    assert abs(rms - MIC_TARGET_RMS) < 0.03


def test_mic_auto_gain_keeps_loud_and_silence(monkeypatch):
    """近场大声不放大（增益≈1）；纯静音不放大（避免把底噪推成语音）。"""
    from meeting_bot.capture import _AutoGain
    agc = _AutoGain()
    loud = np.full(FRAMES, 0.5, dtype=np.float32)
    for _ in range(5):
        agc.process(loud)
    assert agc.gain <= 1.05
    out_loud = agc.process(loud)
    assert np.allclose(out_loud, loud)

    agc2 = _AutoGain()
    silence = np.full(FRAMES, 0.0005, dtype=np.float32)   # 低于 gate
    for _ in range(5):
        agc2.process(silence)
    assert agc2.gain <= 1.05, f"静音被放大到 {agc2.gain:.2f} 倍"


def test_both_mode_applies_gain_on_mic_side_only(monkeypatch):
    """双路模式：麦克风那一路走自动增益，系统声音那一路不动。"""
    from meeting_bot.capture import MIC_MAX_GAIN

    class _QuietRecorder:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def record(self, frames):
            return np.full(frames, self.value, dtype=np.float32)

    _install_both_devices(
        monkeypatch,
        _FakeDevice(lambda: _QuietRecorder(0.005)),    # loopback（系统声音）
        _FakeDevice(lambda: _QuietRecorder(0.005)),    # 麦克风（远场小声）
    )
    cap = AudioCapture()
    assert cap.source == "both"
    cap._running.set()
    cap._side_queues[0].put(np.full(FRAMES, 0.005, dtype=np.float32))   # 系统声音原样
    cap._side_queues[1].put(cap._mic_gain.process(np.full(FRAMES, 0.005, dtype=np.float32)))
    mixer = threading.Thread(target=cap._mix_run, daemon=True)
    mixer.start()
    try:
        out = cap.read(timeout=3.0)
        assert out is not None
        # 麦克风侧被放大（>0.005），与 0.005 相加后应明显高于 0.01
        assert float(np.abs(out).max()) > 0.02, "麦克风侧未应用自动增益"
        assert cap._mic_gain.gain <= MIC_MAX_GAIN + 1e-6
    finally:
        cap._side_queues[0].put(None)
        cap._side_queues[1].put(None)
        mixer.join(timeout=3.0)
