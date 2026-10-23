# -*- coding: utf-8 -*-
"""语句切分：silero-vad 检测语音边界，输出完整语句音频。"""
from __future__ import annotations

import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME = 512                      # silero 单帧样本数 @16k
MIN_SPEECH_FRAMES = 10           # ~0.32s 起算一句
MIN_SILENCE_FRAMES = 15          # ~0.48s 静音断句（帧=512样本@16k≈32ms）
                                 # 从 15 帧(0.48s)降到 10 帧(0.32s)：更快出字幕，
                                 # 代价是句中停顿较久时可能被切成两句（可用 VAD_SILENCE_MS 调回）
MAX_UTTERANCE_FRAMES = 600       # 最长 19s 强制断句


def utterance_bounds(probs: np.ndarray, min_speech: int = MIN_SPEECH_FRAMES,
                     min_silence: int = MIN_SILENCE_FRAMES,
                     max_len: int = MAX_UTTERANCE_FRAMES) -> list[tuple[int, int]]:
    """VAD 概率数组 → [(start_frame, end_frame), ...] 语句边界（半开区间）。"""
    bounds: list[tuple[int, int]] = []
    start = None
    silence = 0
    for i, p in enumerate(probs):
        if p >= 0.5:
            if start is None:
                start = i
                silence = 0
            else:
                silence = 0
                if i - start + 1 >= max_len:   # 超长强制断句
                    bounds.append((start, i + 1))
                    start = None
        elif start is not None:
            silence += 1
            if silence >= min_silence:
                end = i - min_silence + 1   # 语句终点 = 静音起点
                if end - start >= min_speech:
                    bounds.append((start, end))
                start = None
    if start is not None and len(probs) - start >= min_speech:
        bounds.append((start, len(probs)))
    return bounds


class UtteranceBuffer:
    """流式喂入音频块，产出 (起始样本偏移, 语句音频) 列表。

    起始样本偏移是相对录制起点的绝对偏移，由内部维护的丢弃计数保证。
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 min_silence_frames: int = MIN_SILENCE_FRAMES):
        self.sample_rate = sample_rate
        self._min_silence = max(3, int(min_silence_frames))
        self._model = load_silero_vad()
        self._frames: list[np.ndarray] = []   # 未决帧，每帧 FRAME 样本
        self._probs: list[float] = []
        self._dropped = 0                     # 已丢弃的样本数（绝对偏移基准）
        self._pending = np.empty(0, dtype=np.float32)   # 不足一帧的余样，跨 push 续接

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self._pending.size:                # 续接余样，帧边界与绝对偏移才不错位
            arr = np.concatenate((self._pending, arr))
            self._pending = np.empty(0, dtype=np.float32)
        n = len(arr) // FRAME * FRAME
        for i in range(0, n, FRAME):
            frame = arr[i:i + FRAME]
            prob = float(self._model(torch.from_numpy(frame), self.sample_rate))
            self._frames.append(frame)
            self._probs.append(prob)
        self._pending = arr[n:].copy()        # 余样留到下一次 push，不丢不偏移
        return self._emit()

    def _emit(self) -> list[tuple[int, np.ndarray]]:
        probs = np.asarray(self._probs, dtype=np.float32)
        bounds = utterance_bounds(probs, min_silence=self._min_silence)
        if not bounds:
            if probs.size and probs[-1] >= 0.5:
                return []   # 正在说话，缓冲不能丢弃
            # 静音中：只保留最近一段静音上下文
            keep = max(0, len(probs) - self._min_silence * 2)
            self._drop(keep)
            return []
        last_start, last_end = bounds[-1]
        closed = last_end < len(probs) or (last_end - last_start >= MAX_UTTERANCE_FRAMES)
        if closed:
            emit_bounds, keep = bounds, len(probs)
        else:
            # 最后一段还没闭合（仍在语音），留到下一轮
            emit_bounds, keep = bounds[:-1], last_start
        out = [(self._dropped + a * FRAME, np.concatenate(self._frames[a:b]))
               for a, b in emit_bounds]
        self._drop(keep)
        return out

    def flush(self) -> list[tuple[int, np.ndarray]]:
        self._pending = np.empty(0, dtype=np.float32)   # 流结束，尾部不足一帧残留丢弃
        probs = np.asarray(self._probs, dtype=np.float32)
        out = [(self._dropped + a * FRAME, np.concatenate(self._frames[a:b]))
               for a, b in utterance_bounds(probs)]
        self._drop(len(probs))
        return out

    def _drop(self, n: int) -> None:
        del self._frames[:n]
        del self._probs[:n]
        self._dropped += n * FRAME
