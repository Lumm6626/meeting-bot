# -*- coding: utf-8 -*-
"""手动验证说话人聚类：python scripts/check_diarize.py <多说话人音频.wav>"""
import sys

import numpy as np
import soundfile as sf

from meeting_bot.config import load_settings
from meeting_bot.diarization import Diarizer

audio, sr = sf.read(sys.argv[1])
if audio.ndim > 1:
    audio = audio.mean(axis=1)
spans = Diarizer(load_settings()).label(audio.astype(np.float32), sr)
for start, end, label in spans:
    print(f"[{start:7.1f}s - {end:7.1f}s] {label}")
