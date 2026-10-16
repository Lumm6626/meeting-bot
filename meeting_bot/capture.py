# -*- coding: utf-8 -*-
"""系统声音捕获：WASAPI loopback（截取扬声器输出，无需虚拟声卡）。"""
from __future__ import annotations

import queue
import threading

import numpy as np
import soundcard as sc


class CaptureError(Exception):
    """找不到系统扬声器/loopback 设备。"""


CHUNK_SECONDS = 0.5


class AudioCapture:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        try:
            default_speaker = sc.default_speaker()
            self._mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
        except Exception as e:
            raise CaptureError(f"无法打开系统声音捕获（请确认声卡支持 WASAPI）：{e}") from e
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=100)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frames = int(self.sample_rate * CHUNK_SECONDS)
        try:
            with self._mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
                while self._running.is_set():
                    try:
                        data = rec.record(frames)
                    except Exception:
                        break
                    chunk = np.asarray(data, dtype=np.float32).reshape(-1)
                    try:
                        self._queue.put_nowait(chunk)
                    except queue.Full:
                        pass  # 丢弃当前块保实时
        finally:
            # 无论录音器打开/读取/运行期出现任何异常，都必须投递结束哨兵，
            # 否则消费者会永久阻塞在 read() 上。哨兵同样不能阻塞：
            # 队列满时先腾出一个位置，再非阻塞写入。
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass

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

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
