# -*- coding: utf-8 -*-
"""说话人归属（纯算法，无 torch/pyannote 依赖）。

单独成模块的原因：`diarization.py` 顶层 import 了 torch 与 pyannote，而 Qt 进程内
加载 torch 有段错误风险（实测）。标注结果的「归属」只是时间重叠计算，主进程用它
时不应该被拖去加载模型 → 抽到这里。
"""
from __future__ import annotations

from .transcript import Segment


def assign_speakers(segments: list[Segment], spans: list[tuple[float, float, str]],
                    min_overlap: float = 0.3) -> list[Segment]:
    """按时间重叠给语句分配说话人；重叠不足则保持未标注。"""
    for seg in segments:
        best_label, best_overlap = "", 0.0
        for start, end, label in spans:
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        # +epsilon：恰好等于阈值时浮点误差会静默漏标
        # （如 7.0-4.9=2.0999999999999996 < 0.3*7.0=2.1）
        if best_label and best_overlap + 1e-9 >= min_overlap * (seg.end - seg.start):
            seg.speaker = best_label
    return segments
