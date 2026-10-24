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

    def label(self, audio: np.ndarray, sample_rate: int = 16000,
              num_speakers: int | None = None) -> list[tuple[float, float, str]]:
        """返回 [(start, end, label), ...]，时间以秒为单位。

        ``num_speakers``：已知本场有几个人时传进来，pyannote 不再猜人数，聚类更准
        （尤其短会/人少时）；None = 自动检测。
        """
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        kwargs = {}
        if num_speakers and int(num_speakers) > 0:
            kwargs["num_speakers"] = int(num_speakers)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        spans = []
        for turn, _, label in result.itertracks(yield_label=True):
            spans.append((float(turn.start), float(turn.end), str(label)))
        return spans


from .speaker_assign import assign_speakers  # noqa: F401（纯算法拆到轻模块，避免拖入 torch）


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

    def label(self, audio: np.ndarray, sample_rate: int = 16000,
              num_speakers: int | None = None) -> list[tuple[float, float, str]]:
        """返回 [(start, end, label), ...]，时间以秒为单位。

        ``num_speakers``：已知本场有几个人时传进来，pyannote 不再猜人数，聚类更准
        （尤其短会/人少时）；None = 自动检测。
        """
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        kwargs = {}
        if num_speakers and int(num_speakers) > 0:
            kwargs["num_speakers"] = int(num_speakers)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
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
        # +epsilon：恰好等于阈值时浮点误差会静默漏标（如 7.0-4.9=2.0999999999999996 < 0.3*7.0=2.1）
        if best_label and best_overlap + 1e-9 >= min_overlap * (seg.end - seg.start):
            seg.speaker = best_label
    return segments
