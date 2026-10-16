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
        with self._mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
            while self._running.is_set():
                try:
                    data = rec.record(frames)
                    self._queue.put(np.asarray(data, dtype=np.float32).reshape(-1))
                except queue.Full:
                    pass  # 消费不过来时丢弃，保证实时性
                except Exception:
                    break
        self._queue.put(None)  # 结束哨兵

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """读取一块音频；捕获结束返回 None。"""
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
