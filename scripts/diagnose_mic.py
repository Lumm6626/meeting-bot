# -*- coding: utf-8 -*-
"""麦克风链路诊断：60 秒录音 + 逐层打点，定位"无字幕"在哪一层断。

用法（双击 diagnose_mic.bat 或命令行）：
    .venv\Scripts\python -X utf8 scripts\diagnose_mic.py
运行后请用正常会议音量连续说话（可两个人对话）直到自动结束，
产物：会议记录/_check/diag_<时间>.json + .log
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_bot.capture import AudioCapture  # noqa: E402
from meeting_bot.vad import UtteranceBuffer  # noqa: E402

DURATION = int(os.getenv("DIAG_SECONDS", "60"))
OUT = Path("会议记录/_check")
OUT.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT / f"diag_{stamp}.log"
json_path = OUT / f"diag_{stamp}.json"

log_lines: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


log("== 诊断开始 ==")

# Windows 输入音量（必须先于 soundcard 调用，否则 COM 线程模式冲突）
try:
    from pycaw.pycaw import AudioUtilities
    for d in AudioUtilities.GetAllDevices():
        if d.state == 1 and "麦克风" in d.FriendlyName:
            try:
                vol = d.volume_percent
                log(f"  Windows 音量: {d.FriendlyName!r} = {vol:.0f}%")
            except Exception:
                pass
except Exception as e:
    log(f"Windows 音量读取失败: {e}")

log("设备列表：")
for m in sc.all_microphones(include_loopback=False):
    log(f"  - {m.name!r}")
try:
    default_mic = sc.default_microphone()
    log(f"默认输入设备: {default_mic.name!r}")
except Exception as e:
    log(f"默认输入设备不可用: {e}")

log(f"开始录制 {DURATION} 秒，请正常音量说话…")

cap = AudioCapture(source="mic")
cap.start()

buf = UtteranceBuffer()
levels: list[dict] = []
utts: list[dict] = []
utts_audio: list[np.ndarray] = []
peak_every = np.empty(0, dtype=np.float32)
window_start = time.time()
t0 = time.time()

try:
    while time.time() - t0 < DURATION:
        try:
            chunk = cap.read(timeout=0.5)
        except Exception:
            continue
        if chunk is None:
            log("采集线程已结束（异常退出）")
            break
        peak_every = np.concatenate([peak_every, chunk])
        if time.time() - window_start >= 2.0:
            if len(peak_every):
                levels.append({
                    "t": round(time.time() - t0, 1),
                    "peak": round(float(np.abs(peak_every).max()), 4),
                    "rms": round(float(np.sqrt(np.mean(peak_every ** 2))), 5),
                })
                log(f"电平 t={levels[-1]['t']:5.1f}s 峰值={levels[-1]['peak']:.4f} RMS={levels[-1]['rms']:.5f}")
            peak_every = np.empty(0, dtype=np.float32)
            window_start = time.time()
        for off, u in buf.push(chunk):
            utts.append({"offset_s": round(off / 16000, 2),
                         "len_s": round(len(u) / 16000, 2),
                         "peak": round(float(np.abs(u).max()), 4)})
            utts_audio.append(u)
            log(f"VAD 切出语句 @{utts[-1]['offset_s']}s 时长{utts[-1]['len_s']}s 峰值{utts[-1]['peak']}")
finally:
    for off, u in buf.flush():
        utts.append({"offset_s": round(off / 16000, 2),
                     "len_s": round(len(u) / 16000, 2),
                     "peak": round(float(np.abs(u).max()), 4)})
        utts_audio.append(u)
        log(f"VAD 切出语句(flush) @{utts[-1]['offset_s']}s 时长{utts[-1]['len_s']}s")
    cap.stop()

log(f"电平窗口数={len(levels)}，VAD 语句数={len(utts)}")

# 若 VAD 有产出，验证识别层
asr_result: list[dict] = []
if utts_audio:
    log("加载识别模型（small，约 2s）…")
    try:
        from meeting_bot.asr import ASREngine
        from meeting_bot.config import load_settings
        eng = ASREngine(load_settings())
        for i in range(min(2, len(utts_audio))):
            u = utts[i]
            log(f"识别第 {i + 1} 句（{u['len_s']}s）…")
            t_a = time.time()
            segs = eng.transcribe(utts_audio[i])
            dt = time.time() - t_a
            texts = [s.text for s in segs]
            asr_result.append({"idx": i, "seconds": round(dt, 1),
                               "lang": segs[0].lang if segs else "",
                               "text": texts})
            log(f"  耗时 {dt:.1f}s 语种={asr_result[-1]['lang']} 文本={texts}")
    except Exception as e:
        log(f"识别模块加载失败: {e}")

summary = {
    "time": stamp,
    "default_mic": repr(default_mic.name) if "default_mic" in dir() else "?",
    "levels": levels,
    "vad_utterances": utts,
    "asr": asr_result,
}
json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
log_path.write_text("\n".join(log_lines), encoding="utf-8")
log(f"诊断完成，报告已存：{json_path}")
