# -*- coding: utf-8 -*-
"""录制编排：串联捕获→切句→识别→标注→翻译→存储。"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Callable

import numpy as np

from .asr import ASREngine
from .capture import AudioCapture
from .config import Settings
from .diarization import Diarizer, assign_speakers
from .transcript import Segment, TranscriptStore
from .translator import TranslationError, Translator
from .vad import UtteranceBuffer

SegmentEvent = Callable[[Segment], None]
StatusEvent = Callable[[str], None]

DIARIZE_INTERVAL = 15.0       # 每 15 秒跑一次说话人聚类
PRODUCER_DONE_TIMEOUT = 10.0  # stop() 等主循环完成尾部 flush 的上限
TRANSLATE_DRAIN_TIMEOUT = 20.0  # stop() 等翻译 worker 排空退出的上限
TRANSLATE_POLL_TIMEOUT = 0.5  # 翻译 worker 队列轮询间隔


class Recorder:
    def __init__(self, settings: Settings,
                 capture: AudioCapture | None = None,
                 asr: ASREngine | None = None,
                 translator: Translator | None = None,
                 diarizer: Diarizer | None = None):
        self._settings = settings
        self._capture = capture or AudioCapture()
        self._asr = asr
        self._translator = translator
        self._diarizer = diarizer
        self._store = TranscriptStore()
        self._events: list[SegmentEvent] = []
        self._status: list[StatusEvent] = []
        self._stop = threading.Event()
        # producer-done：主循环（生产者）已完成尾部 flush，翻译 worker 方可排空退出。
        # 只靠 _stop 会有窗口期：队列一度为空时 worker 退出，之后 flush 的尾部语句无人翻译。
        self._producer_done = threading.Event()
        self._worker_done = threading.Event()   # 翻译 worker 已退出（排空完成）
        self._running = threading.Event()       # 主循环是否在跑（start 时 set、退出时 clear）
        self._translate_thread: threading.Thread | None = None
        self._translate_queue: Queue[Segment] = Queue()
        self._diarize_audio: list[np.ndarray] = []
        self._started_at: datetime | None = None
        self.title = ""

    # --- 事件 ---
    def on_segment(self, fn: SegmentEvent) -> None:
        self._events.append(fn)

    def on_status(self, fn: StatusEvent) -> None:
        self._status.append(fn)

    def _emit(self, seg: Segment) -> None:
        for fn in self._events:
            fn(seg)

    def _say(self, msg: str) -> None:
        for fn in self._status:
            fn(msg)

    @property
    def store(self) -> TranscriptStore:
        return self._store

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def translator(self) -> Translator | None:
        return self._translator

    def start(self, title: str = "") -> None:
        self.title = title
        self._started_at = datetime.now()
        self._say("开始录制（等待语音…）")
        self._producer_done.clear()
        self._worker_done.clear()
        self._running.set()
        self._capture.start()
        self._translate_thread = threading.Thread(target=self._translate_worker, daemon=True)
        self._translate_thread.start()
        threading.Thread(target=self._diarize_worker, daemon=True).start()
        self._main_loop()

    def _main_loop(self) -> None:
        buf = UtteranceBuffer()
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._capture.read(timeout=0.5)
                except queue.Empty:
                    # read() 契约：超时抛 queue.Empty，采集仍在进行 → 继续读
                    continue
                if chunk is None:
                    break        # 结束哨兵：采集已停止
                self._diarize_audio.append(chunk)
                for offset, utterance in buf.push(chunk):
                    self._process_utterance(utterance, offset / 16000)
            for offset, utterance in buf.flush():
                self._process_utterance(utterance, offset / 16000)
        finally:
            # 尾部 flush 结束（或主循环异常）才算生产者完结；顺序不能反：
            # 必须先 set(producer_done) 再 clear(running)，否则 stop() 可能跳过等待。
            self._producer_done.set()
            self._running.clear()

    def _process_utterance(self, audio: np.ndarray, start_offset: float) -> None:
        try:
            if self._asr is None:
                # 惰性创建：重型模型只在首句到来时加载（测试注入 fake 时不走此行）
                self._asr = ASREngine(self._settings)
            segments = self._asr.transcribe(audio)
        except Exception as e:
            self._say(f"识别失败：{e}")
            return
        for s in segments:
            seg = self._store.add(start_offset + s.start, start_offset + s.end,
                                  s.text, lang=s.lang)
            if self._translator:
                self._translate_queue.put(seg)
            else:
                self._emit(seg)

    def _translate_worker(self) -> None:
        try:
            # producer-done 模式：只认「生产者完结 + 队列排空」才退出。
            # 若用 _stop 判定，尾段仍在内核 ASR（0.5~3s）时 worker 就会因队列暂空而退出，
            # flush 产出的 Segment 将永远无人翻译、无人 emit。
            while not self._producer_done.is_set() or not self._translate_queue.empty():
                try:
                    seg = self._translate_queue.get(timeout=TRANSLATE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                try:
                    if seg.lang == "zh":
                        seg.translation = seg.text
                    else:
                        seg.translation = self._translator.translate(seg.text, seg.lang)
                except TranslationError:
                    seg.translation = ""   # 会议结束后统一补译
                    self._say(f"[警告] 翻译失败（{seg.text[:20]}…），已保留原文")
                self._emit(seg)
        finally:
            self._worker_done.set()

    def _diarize_worker(self) -> None:
        if not self._diarizer:
            return
        last_n = 0
        while not self._stop.is_set():
            time.sleep(DIARIZE_INTERVAL)
            if len(self._diarize_audio) == last_n:
                continue
            last_n = len(self._diarize_audio)
            audio = np.concatenate(self._diarize_audio)
            try:
                spans = self._diarizer.label(audio)
            except Exception as e:
                self._say(f"[警告] 说话人标注失败：{e}")
                continue
            assign_speakers(self._store.segments(), spans)
            for seg in self._store.segments():
                if seg.speaker:
                    self._store.register_label(seg.speaker)
            self._say(f"已识别说话人：{len(self._store.speakers())} 人")

    def stop(self) -> None:
        self._stop.set()
        self._capture.stop()
        if self._running.is_set():
            # start() 在调用线程里同步跑主循环，stop() 无法 join 它；
            # 等 producer_done 即可确保尾部 flush 已经把 Segment 投进翻译队列。
            # stop() 早于/独立于 start()（无人在跑主循环）时 _running 为 False，直接跳过。
            self._producer_done.wait(timeout=PRODUCER_DONE_TIMEOUT)
        worker = self._translate_thread
        if worker is not None and worker.is_alive():
            # 只等 queue.empty() 不够：worker 可能已取走最后一条却还没 emit。
            self._worker_done.wait(timeout=TRANSLATE_DRAIN_TIMEOUT)
        if self._translator:
            for seg in self._store.segments():
                if not seg.translation and seg.lang != "zh":
                    try:
                        seg.translation = self._translator.translate(seg.text, seg.lang)
                    except TranslationError:
                        pass
