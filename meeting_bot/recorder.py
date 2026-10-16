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

DIARIZE_INTERVAL = 15.0      # 每 15 秒跑一次说话人聚类


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
        self._capture.start()
        threading.Thread(target=self._translate_worker, daemon=True).start()
        threading.Thread(target=self._diarize_worker, daemon=True).start()
        self._main_loop()

    def _main_loop(self) -> None:
        buf = UtteranceBuffer()
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
        while not self._stop.is_set() or not self._translate_queue.empty():
            try:
                seg = self._translate_queue.get(timeout=0.5)
            except Exception:
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
        deadline = time.time() + 20
        while not self._translate_queue.empty() and time.time() < deadline:
            time.sleep(0.2)
        if self._translator:
            for seg in self._store.segments():
                if not seg.translation and seg.lang != "zh":
                    try:
                        seg.translation = self._translator.translate(seg.text, seg.lang)
                    except TranslationError:
                        pass
