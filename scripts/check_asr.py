# -*- coding: utf-8 -*-
"""手动验证识别（首次运行会下载 whisper medium 模型 ~1.5GB）。

用法：python scripts/check_asr.py <音频文件.wav>
"""
import sys

import numpy as np
import soundfile as sf

from meeting_bot.asr import ASREngine
from meeting_bot.config import load_settings

audio, sr = sf.read(sys.argv[1])
if audio.ndim > 1:
    audio = audio.mean(axis=1)
segments = ASREngine(load_settings()).transcribe(audio.astype(np.float32), sr)
for s in segments:
    print(f"[{s.start:6.1f}-{s.end:6.1f}] ({s.lang}) {s.text}")
