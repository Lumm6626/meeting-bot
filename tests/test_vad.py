# -*- coding: utf-8 -*-
import numpy as np

from meeting_bot.vad import FRAME, UtteranceBuffer, utterance_bounds


def _prob_array(speech_ranges, total):
    probs = np.zeros(total, dtype=np.float32)
    for a, b in speech_ranges:
        probs[a:b] = 1.0
    return probs


def test_single_utterance():
    probs = _prob_array([(0, 50), (70, 120)], 200)
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10)
    assert bounds == [(0, 50), (70, 120)]


def test_tail_unclosed_utterance_included():
    probs = _prob_array([(30, 100)], 100)
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10)
    assert bounds == [(30, 100)]  # 结尾未出现静音，也算完整语句


def test_too_short_speech_ignored():
    probs = _prob_array([(0, 3)], 50)  # 3 帧 < min_speech=5
    assert utterance_bounds(probs, min_speech=5, min_silence=10) == []


def test_max_length_forces_split():
    probs = _prob_array([(0, 600)], 600)  # 600 帧 >= max_len=300 → 强制断句
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10, max_len=300)
    assert bounds == [(0, 300), (300, 600)]


def test_silence_exactly_min_silence_closes():
    probs = _prob_array([(30, 90)], 100)  # 帧 90..99 恰好 10 帧静音
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10)
    assert bounds == [(30, 90)]


def test_push_keeps_tail_samples_between_chunks(monkeypatch):
    """连续 push 非整帧对齐的块：余样必须缓存，块间一帧不丢、不错位。"""
    recorded: list[np.ndarray] = []

    class _ZeroModel:
        def __call__(self, chunk, sr):
            recorded.append(chunk.numpy().copy())
            return 0.0

    monkeypatch.setattr("meeting_bot.vad.load_silero_vad", lambda: _ZeroModel())
    stream = np.arange(16000, dtype=np.float32)

    buf = UtteranceBuffer()
    buf.push(stream[:8000])                      # 8000 = 15*512 + 320

    assert len(buf._frames) == 8000 // FRAME == 15
    assert np.array_equal(np.concatenate(recorded), stream[:15 * FRAME])

    buf.push(stream[8000:])                      # 第二块 8000 样本

    # 累计 8000+8000=16000 样本 → 31 整帧 + 128 余样，一帧都不能丢
    assert len(recorded) == stream.size // FRAME == 31
    assert np.array_equal(np.concatenate(recorded), stream[:31 * FRAME])
    assert len(buf._pending) == stream.size % FRAME == 128


def test_utterance_buffer_respects_custom_silence():
    """静音阈值可配：自定义 5 帧静音即断句（默认 10 帧），验证 VAD_SILENCE_MS 生效路径。"""
    from meeting_bot.vad import UtteranceBuffer
    probs = _prob_array([(0, 30), (35, 60)], 80)   # 中间 5 帧静音
    bounds = utterance_bounds(probs, min_speech=5, min_silence=5)
    assert bounds == [(0, 30), (35, 60)]


def test_utterance_buffer_accepts_min_silence_param():
    """UtteranceBuffer(min_silence_frames=...) 构造不报错且内部生效。"""
    from meeting_bot.vad import UtteranceBuffer
    buf = UtteranceBuffer(min_silence_frames=5)
    assert buf._min_silence == 5
