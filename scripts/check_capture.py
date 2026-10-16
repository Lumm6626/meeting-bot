# -*- coding: utf-8 -*-
"""手动验证系统声音捕获：运行后请播放任意音频/视频 10 秒。

用法：python scripts/check_capture.py
输出：会议记录/_check/loopback_test.wav
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from meeting_bot.capture import AudioCapture

OUT = Path("会议记录/_check/loopback_test.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

cap = AudioCapture()
cap.start()
chunks = []
for _ in range(20):  # 10 秒
    c = cap.read(timeout=1.0)
    if c is None:
        break
    chunks.append(c)
cap.stop()
if not chunks:
    print("未捕获到任何音频！")
else:
    audio = np.concatenate(chunks)
    sf.write(str(OUT), audio, 16000)
    peak = float(np.abs(audio).max())
    print(f"已保存 {len(audio)/16000:.1f} 秒音频到 {OUT}（峰值 {peak:.3f}）")
    print("峰值 > 0.01 说明捕获正常；否则检查是否在播放声音")
