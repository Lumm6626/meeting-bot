# -*- coding: utf-8 -*-
"""本地语音识别：faster-whisper，CPU int8，自动语种检测。"""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from .config import Settings
from .transcript import Segment


class ASREngine:
    def __init__(self, settings: Settings):
        self._model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[Segment]:
        """识别语句，返回不含 speaker/translation 的 Segment 列表。"""
        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            language=None,           # 自动检测语种
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        result = []
        for s in segments:
            text = s.text.strip()
            if text:
                result.append(Segment(start=float(s.start), end=float(s.end),
                                      text=text, lang=info.language or ""))
        return result
