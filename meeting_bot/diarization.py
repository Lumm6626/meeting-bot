# -*- coding: utf-8 -*-
"""说话人区分：pyannote 声纹聚类 + 语句归属。"""
from __future__ import annotations

import numpy as np
import torch
from pyannote.audio import Pipeline

from .config import Settings
from .transcript import Segment


class DiarizationError(Exception):
    """说话人聚类不可用（如未配置 HF token）。"""


class Diarizer:
    def __init__(self, settings: Settings):
        if not settings.hf_token:
            raise DiarizationError("缺少 HF_TOKEN，无法使用说话人区分")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )

    def label(self, audio: np.ndarray, sample_rate: int = 16000) -> list[tuple[float, float, str]]:
        """返回 [(start, end, label), ...]，时间以秒为单位。"""
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
        spans = []
        for turn, _, label in result.itertracks(yield_label=True):
            spans.append((float(turn.start), float(turn.end), str(label)))
        return spans


def assign_speakers(segments: list[Segment], spans: list[tuple[float, float, str]],
                    min_overlap: float = 0.3) -> list[Segment]:
    """按时间重叠给语句分配说话人；重叠不足则保持未标注。"""
    for seg in segments:
        best_label, best_overlap = "", 0.0
        for start, end, label in spans:
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        if best_label and best_overlap >= min_overlap * (seg.end - seg.start):
            seg.speaker = best_label
    return segments
