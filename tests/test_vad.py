# -*- coding: utf-8 -*-
import numpy as np

from meeting_bot.vad import utterance_bounds


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
