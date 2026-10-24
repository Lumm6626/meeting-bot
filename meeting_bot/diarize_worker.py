# -*- coding: utf-8 -*-
"""说话人标注子进程：在独立进程里跑 pyannote（不加载 Qt）。

为什么要独立进程：实测在创建过 QApplication 的进程里加载 pyannote 模型会
**段错误崩溃**（Qt 与 torch 的 OpenMP/DLL 冲突），而同样代码在纯脚本里正常。
把重活放到子进程后：
- 主程序不会因为标注崩溃（子进程崩了只表现为一条失败告警）；
- 也顺带绕开 torch 抢 GIL 拖慢界面。

用法（由 app.py 调用）：
    python -m meeting_bot.diarize_worker --wav <录音.wav> [--num-speakers N] --out <结果.json>
输出 JSON：
    {"ok": true, "spans": [[start, end, "SPEAKER_00"], ...],
     "embeddings": {"SPEAKER_00": [0.1, ...]}, "error": null}
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="会议说话人标注（独立进程）")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-speakers", type=int, default=0)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    result: dict = {"ok": False, "spans": [], "embeddings": {}, "error": None}
    try:
        import numpy as np
        import soundfile as sf

        from meeting_bot.config import load_settings
        from meeting_bot.diarization import Diarizer
        from meeting_bot.voiceprints import compute_speaker_embeddings

        settings = load_settings()
        audio, sample_rate = sf.read(str(args.wav), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.asarray(audio)[:, 0]
        audio = np.asarray(audio, dtype=np.float32)

        diarizer = Diarizer(settings)
        spans = diarizer.label(audio, int(sample_rate),
                               num_speakers=(args.num_speakers or None))
        result["spans"] = [[float(s), float(e), str(lbl)] for s, e, lbl in spans]
        try:
            embs = compute_speaker_embeddings(
                audio, int(sample_rate), spans, token=settings.hf_token)
            result["embeddings"] = {
                str(k): [float(x) for x in np.asarray(v).ravel()]
                for k, v in (embs or {}).items()}
        except Exception as e:       # 声纹是增强项：算不出来不影响标注结果
            result["embedding_error"] = f"{type(e).__name__}: {e}"
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-2000:]
    try:
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
