# -*- coding: utf-8 -*-
"""音频捕获：扬声器 loopback + 默认麦克风，双路混合（自动模式，无需选模式）。

``source`` 三种取值：

- ``"both"``（默认）：同时采集系统声音（loopback）与默认麦克风，两路逐样本相加后
  输出一路 16kHz 单声道音频。线上会议对方语音走 loopback、自己发言走麦克风，
  线下对话走麦克风——不区分线上线下，电脑能收到的声音都进识别；
- ``"speaker"``：只录系统播放的声音（loopback）；
- ``"mic"``：只录默认麦克风。
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import soundcard as sc


class CaptureError(Exception):
    """找不到可用的捕获设备（扬声器 loopback / 麦克风）。"""


CHUNK_SECONDS = 0.5
MIX_TIMEOUT = 0.2            # 混合线程等待新音频的上限（秒）：超时即跳过本轮，不阻塞
MIX_MAX_BUFFERED_CHUNKS = 3  # 单边缓冲上限：超过即丢最旧的数据（时钟漂移防御）
MIX_IDLE_GRACE = 3.0
MIX_GAIN = 0.8                # 双路相加后的缩放系数（防削波）         # 单路无数据超过此时长 → 本轮先直通另一路（防 loopback 饿死）
MIX_IDLE_REPORT = 30.0       # 单路连续无数据超过此时长 → 记一条诊断（真异常才提示）

SOURCE_SPEAKER = "speaker"   # 只录系统扬声器 loopback
SOURCE_MIC = "mic"           # 只录默认麦克风
SOURCE_BOTH = "both"         # 双路同时采集 + 混合（默认；线上/线下都不用手动切换）
SOURCES = (SOURCE_SPEAKER, SOURCE_MIC, SOURCE_BOTH)


MIC_TARGET_RMS = 0.07     # 麦克风目标电平（对语音约 -23 dBFS，够 VAD/ASR 识别）
MIC_MAX_GAIN = 20.0       # 最大放大倍数（≈ +26 dB）：远场说话也能拉起来
MIC_GATE_RMS = 0.0015     # 低于此电平视为静音，不放大（避免把底噪放大成"人声"）


class _AutoGain:
    """麦克风自动增益：把远场/小声的语音拉到可识别电平。

    只对麦克风生效（系统声音已经是满量程）。起音快、释放慢，避免音量泵动；
    静音段（低于 gate）不放大，防止底噪被推成语音导致 VAD 误触发。
    """

    def __init__(self, target: float = MIC_TARGET_RMS, max_gain: float = MIC_MAX_GAIN,
                 gate: float = MIC_GATE_RMS):
        self._target = target
        self._max = max_gain
        self._gate = gate
        self.gain = 1.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) + 1e-9
        desired = 1.0 if rms < self._gate else min(self._max, self._target / rms)
        if desired > self.gain:
            self.gain = desired                      # 起音：立即跟上
        else:
            self.gain = 0.7 * self.gain + 0.3 * desired   # 释放：缓慢回落
        if self.gain <= 1.001:
            return chunk
        return np.clip(chunk * self.gain, -1.0, 1.0).astype(np.float32, copy=False)


class AudioCapture:
    """音频捕获：统一产出 16kHz 单声道块，走队列交给消费者。

    ``source`` 决定打开哪些设备：

    - ``"both"``（默认）：loopback + 默认麦克风，两路各起一个采集线程写入各自的
      中间队列，再由混合线程配对相加（``np.clip`` 防削波）后输出。某一路先结束
      （设备断开 / 报错）时自动降级为另一路直通，不会因此中断整场录制；
    - ``"speaker"``：默认扬声器的 loopback，截取系统正在播放的声音；
    - ``"mic"``：系统默认麦克风，采集现场对话（会录到自己的声音）。

    对外接口在三种模式下完全一致（``read()`` / ``start()`` / ``stop()``、队列上限、
    结束哨兵、``error`` / ``dropped_chunks`` 诊断字段）。
    """

    def __init__(self, sample_rate: int = 16000, source: str = SOURCE_BOTH):
        if source not in SOURCES:
            raise CaptureError(f"未知音频源：{source!r}（可选：{' / '.join(SOURCES)}）")
        self.sample_rate = sample_rate
        self.source = source
        self._frames = int(sample_rate * CHUNK_SECONDS)
        # 输出队列：消费端（Recorder）只读这一个；both 模式下由混合线程写入
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=100)
        self._thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        # 采集中断的异常：record() 一旦抛错，采集线程只会退出，
        # 主循环看到的是「正常结束哨兵」——不记下来就变成「音频莫名截断」，
        # 事后无从定位（上层 Recorder 会读它并上报）。多路时保留第一个错误。
        self.error: Exception | None = None
        # 背压丢弃计数：队列满时丢块保实时，丢多了意味着处理跟不上，需要可见。
        self.dropped_chunks = 0
        self._lock = threading.Lock()
        # --- both 模式专用 ---
        self._device = None          # loopback 设备（single speaker 模式也用它）
        self._mic_device = None      # 默认麦克风（both 模式的第二路）
        self._side_queues: list[queue.Queue[np.ndarray | None]] = []
        self._side_alive: list[bool] = []
        self._side_buffers: list[np.ndarray] = []
        self._side_last_data: list[float] = [0.0, 0.0]
        # 麦克风路的自动增益（系统声音路不需要：已是满量程）
        self._mic_gain = _AutoGain()
        self._idle_reported: list[bool] = [False, False]

        if source == SOURCE_MIC:
            try:
                self._device = self._require(sc.default_microphone(), "麦克风")
            except Exception as e:
                raise CaptureError(f"无法打开麦克风捕获（请确认设备可用）：{e}") from e
        elif source == SOURCE_SPEAKER:
            try:
                self._device = self._require(self._loopback_device(), "系统声音")
            except Exception as e:
                raise CaptureError(f"无法打开系统声音捕获（请确认设备可用）：{e}") from e
        else:
            loop_err = mic_err = None
            try:
                self._device = self._require(self._loopback_device(), "系统声音")
            except Exception as e:
                loop_err = e
            try:
                self._mic_device = self._require(sc.default_microphone(), "麦克风")
            except Exception as e:
                mic_err = e
            if loop_err is not None or mic_err is not None:
                detail = "；".join(
                    part for part in (
                        f"系统声音：{loop_err}" if loop_err is not None else "",
                        f"麦克风：{mic_err}" if mic_err is not None else "",
                    ) if part)
                raise CaptureError(
                    f"自动模式需要同时打开系统声音与麦克风，其中一路不可用（{detail}）。"
                    "如只想录一路：在 .env 里设 AUDIO_SOURCE=speaker（线上会议）"
                    "或 AUDIO_SOURCE=mic（现场对话）") from (loop_err or mic_err)
            for _ in range(2):
                self._side_queues.append(queue.Queue(maxsize=100))
                self._side_alive.append(True)
                self._side_buffers.append(np.zeros(0, dtype=np.float32))
            self._side_last_data = [time.time(), time.time()]

    # --- 设备解析 -----------------------------------------------------------
    @staticmethod
    def _require(device, label: str):
        if device is None:
            raise RuntimeError(f"系统未返回{label}设备")
        return device

    @staticmethod
    def _loopback_device():
        """默认扬声器对应的 loopback 采集设备（WASAPI loopback，无需虚拟声卡）。"""
        default_speaker = sc.default_speaker()
        if default_speaker is None:
            raise RuntimeError("系统未返回默认扬声器")
        return sc.get_microphone(id=str(default_speaker.name), include_loopback=True)

    # --- 生命周期 -----------------------------------------------------------
    def start(self) -> None:
        self._running.set()
        if self.source != SOURCE_BOTH:
            self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
            self._threads = [self._thread]
            self._thread.start()
            return
        # both：两路采集线程（各自写中间队列）+ 一个混合线程（配对相加写输出队列）
        self._threads = []
        for i, (name, device) in enumerate(
                (("loopback", self._device), ("mic", self._mic_device))):
            self._side_alive[i] = True
            self._side_buffers[i] = np.zeros(0, dtype=np.float32)
            self._side_last_data[i] = time.time()
            self._idle_reported[i] = False
            self._threads.append(threading.Thread(
                target=self._run_side, args=(device, self._side_queues[i], name),
                name=f"audio-capture-{name}", daemon=True))
        self._thread = threading.Thread(target=self._mix_run, name="audio-mix", daemon=True)
        self._threads.append(self._thread)
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._running.clear()
        # single 模式 self._threads == [采集线程]；both 模式 == [两路采集, 混合]
        for thread in self._threads:
            thread.join(timeout=3.0)

    # --- 采集 ---------------------------------------------------------------
    def _run(self) -> None:
        """单路采集（speaker / mic）：设备 → 输出队列。"""
        try:
            with self._device.recorder(samplerate=self.sample_rate, channels=1) as rec:
                while self._running.is_set():
                    try:
                        data = rec.record(self._frames)
                    except Exception as e:
                        self._set_error(e)   # 不能静默 break：上层要能上报「采集中断」
                        break
                    chunk = self._as_chunk(data)
                    if self.source == SOURCE_MIC:
                        chunk = self._mic_gain.process(chunk)
                    self._put(self._queue, chunk)
        finally:
            # 无论录音器打开/读取/运行期出现任何异常，都必须投递结束哨兵，
            # 否则消费者会永久阻塞在 read() 上。
            self._put_sentinel(self._queue)

    def _run_side(self, device, out: queue.Queue, label: str) -> None:
        """both 模式的一路采集：设备 → 中间队列（由混合线程配对）。

        与 ``_run`` 的差别：打开设备失败也要记进 ``self.error`` —— 混合线程只看得
        到「结束哨兵」，不记错误就会退化成「少了一路声音」而不告警。
        """
        try:
            with device.recorder(samplerate=self.sample_rate, channels=1) as rec:
                while self._running.is_set():
                    try:
                        data = rec.record(self._frames)
                    except Exception as e:
                        self._set_error(e)
                        break
                    chunk = self._as_chunk(data)
                    if label == "mic":
                        chunk = self._mic_gain.process(chunk)
                    self._put(out, chunk)
        except Exception as e:
            self._set_error(e)
        finally:
            self._put_sentinel(out)

    def _mix_run(self) -> None:
        """混合线程：两边各攒够一块就相加（clip），只剩一路就直通，都结束则投哨兵。"""
        try:
            while True:
                for i in range(len(self._side_queues)):
                    self._absorb(i)
                self._emit_ready()
                if not any(self._side_alive):
                    break
                self._wait_one()
        finally:
            self._put_sentinel(self._queue)

    def _absorb(self, i: int) -> None:
        """把第 i 路中间队列里已有的块全部并入缓冲（非阻塞，不拖慢另一路）。"""
        q = self._side_queues[i]
        while self._side_alive[i]:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._side_alive[i] = False
                return
            self._append(i, item)

    def _wait_one(self) -> None:
        """阻塞等一路新数据（最多 MIX_TIMEOUT）：超时即返回，让下一轮重新对齐两路。"""
        for i in range(len(self._side_queues)):
            if not self._side_alive[i]:
                continue
            try:
                item = self._side_queues[i].get(timeout=MIX_TIMEOUT)
            except queue.Empty:
                return
            if item is None:
                self._side_alive[i] = False
            else:
                self._append(i, item)
            return

    def _append(self, i: int, item: np.ndarray) -> None:
        chunk = np.asarray(item, dtype=np.float32).reshape(-1)
        self._side_last_data[i] = time.time()
        if self._side_buffers[i].size:
            chunk = np.concatenate((self._side_buffers[i], chunk))
        limit = self._frames * MIX_MAX_BUFFERED_CHUNKS
        if chunk.size > limit:
            # 时钟漂移防御：单边积压超过 3 块 → 丢最旧的一段，保实时（漂移极小）
            chunk = chunk[chunk.size - limit:]
            self._count_drop()
        self._side_buffers[i] = chunk

    def _emit_ready(self) -> None:
        """攒够一块就输出：两边都活时配对混音，只剩一路时直通。

        闲置旁通（不判死）：某些驱动的 loopback 在没有播放时会「饿死」（record
        迟迟不返回），死等配对会让麦克风音频永远发不出去（字幕归零）。某路超过
        ``MIX_IDLE_GRACE`` 秒无新数据时，本轮先把另一路直通输出；该路**不**标记
        失效——对方后来开口时数据一到就自动恢复双路混音，不会永久丢掉线上声音。
        连续闲置超过 ``MIX_IDLE_REPORT`` 才记一条诊断，避免每次静场都刷告警。
        """
        frames = self._frames
        while True:
            if all(self._side_alive):
                idle = [i for i in (0, 1)
                        if self._side_buffers[i].size < frames
                        and time.time() - self._side_last_data[i] > MIX_IDLE_GRACE]
                if idle and self._side_buffers[1 - idle[0]].size >= frames:
                    self._note_idle(idle[0])
                    self._put(self._queue, self._take(1 - idle[0]))
                    continue
                if (self._side_buffers[0].size < frames
                        or self._side_buffers[1].size < frames):
                    return
                a = self._take(0)
                b = self._take(1)
                # 直接相加在「两路都响」时会削波（实测峰值顶到 1.0，削波会毁掉
                # 识别质量）：先按 0.8 缩放再限幅，保留动态余量。
                mixed = np.clip((a + b) * MIX_GAIN, -1.0, 1.0)
                self._put(self._queue, mixed.astype(np.float32, copy=False))
                continue
            i = 0 if self._side_alive[0] else 1
            if not self._side_alive[i]:        # 两路都结束
                return
            if self._side_buffers[i].size < frames:
                return
            self._put(self._queue, self._take(i))

    def _note_idle(self, i: int) -> None:
        """闲置路诊断：只在「长时间」无数据时记一次（短时静场属正常，不告警）。"""
        if self._idle_reported[i]:
            return
        if time.time() - self._side_last_data[i] <= MIX_IDLE_REPORT:
            return
        self._idle_reported[i] = True
        self._set_error(RuntimeError(
            f"{('系统声音' if i == 0 else '麦克风')}路长时间没有数据"
            f"（超过 {MIX_IDLE_REPORT:.0f} 秒，可能设备被占用或没有声音输入），"
            "期间先用另一路继续；该路恢复后会自动重新混合"))

    def _take(self, i: int) -> np.ndarray:
        chunk = self._side_buffers[i][:self._frames]
        self._side_buffers[i] = self._side_buffers[i][self._frames:]
        return chunk

    # --- 队列写入 / 诊断 -----------------------------------------------------
    def _as_chunk(self, data) -> np.ndarray:
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def _set_error(self, e: Exception) -> None:
        with self._lock:
            if self.error is None:
                self.error = e        # 保留第一个错误，后来的不覆盖

    def _count_drop(self) -> None:
        with self._lock:
            self.dropped_chunks += 1

    def _put(self, q: queue.Queue, chunk: np.ndarray) -> None:
        try:
            q.put_nowait(chunk)
        except queue.Full:
            self._count_drop()        # 丢弃当前块保实时

    def _put_sentinel(self, q: queue.Queue) -> None:
        # 哨兵同样不能阻塞：队列满时先腾出一个位置，再非阻塞写入。
        try:
            q.put_nowait(None)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    # --- 消费端接口 ---------------------------------------------------------
    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """读取一块音频。

        契约：**超时抛 ``queue.Empty``；采集结束返回 ``None``**。
        二者语义不同，调用方需区分对待：

        - 返回 ``None``：捕获线程已结束（结束哨兵），可以退出读取循环；
        - 抛 ``queue.Empty``：本次 ``timeout`` 内没有新数据，采集仍在进行，
          调用方 ``try/except queue.Empty`` 后应继续读取。

        典型用法::

            while True:
                try:
                    chunk = cap.read(timeout=1.0)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                handle(chunk)
        """
        return self._queue.get(timeout=timeout)
