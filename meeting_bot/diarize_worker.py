# -*- coding: utf-8 -*-
"""说话人标注子进程：在独立进程里跑 pyannote（不加载 Qt）。

为什么要独立进程：实测在创建过 QApplication 的进程里加载 pyannote 模型会
**段错误崩溃**（Qt 与 torch 的 OpenMP/DLL 冲突），而同样代码在纯脚本里正常。
把重活放到子进程后：
- 主程序不会因为标注崩溃（子进程崩了只表现为一条失败告警）；
- 也顺带绕开 torch 抢 GIL 拖慢界面。

两种模式：
- 全量（默认）：pyannote 聚类全段音频，精确但慢（0.5~1.5 倍会议时长）；
- 快速（``--fast``）：只对「发言回合」算声纹向量再做在线聚类（1~2 分钟），
  用于会议结束后**立刻**出带人名的纪要（见 ``fast_identify``）。

用法（由 app.py 调用）：
    python -m meeting_bot.diarize_worker --wav <录音.wav> [--num-speakers N] --out <结果.json>
    python -m meeting_bot.diarize_worker --wav <录音.wav> --fast --spans <回合.json> --out <结果.json>
输出 JSON：
    {"ok": true, "spans": [[start, end, "SPEAKER_00"], ...],
     "embeddings": {"SPEAKER_00": [0.1, ...]}, "error": null}
    快速模式额外带 "names": {"SPEAKER_00": "Peter Ren"} 与 "scores"。
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
    parser.add_argument("--fast", action="store_true",
                        help="快速模式：按发言回合算声纹向量再聚类（1~2 分钟）")
    parser.add_argument("--spans", default="",
                        help="快速模式的发言回合 JSON：[[start, end], ...]")
    parser.add_argument("--first-index", type=int, default=0,
                        help="新建声纹标签的起始编号（避免与已有 SPEAKER_xx 撞车）")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="快速模式：单段认人的余弦阈值（0=用默认）")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    result: dict = {"ok": False, "spans": [], "embeddings": {}, "error": None}
    try:
        if args.fast:
            result = _run_fast(args, result)
        else:
            result = _run_full(args, result)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-2000:]
    try:
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return 0 if result.get("ok") else 1


def _read_wav(path: str):
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.asarray(audio)[:, 0]
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _run_full(args, result: dict) -> dict:
    from meeting_bot.config import load_settings
    from meeting_bot.diarization import Diarizer
    from meeting_bot.voiceprints import compute_speaker_embeddings

    import numpy as np

    settings = load_settings()
    audio, sample_rate = _read_wav(args.wav)
    diarizer = Diarizer(settings)
    spans = diarizer.label(audio, sample_rate,
                           num_speakers=(args.num_speakers or None))
    result["spans"] = [[float(s), float(e), str(lbl)] for s, e, lbl in spans]
    try:
        embs = compute_speaker_embeddings(
            audio, sample_rate, spans, token=settings.hf_token)
        result["embeddings"] = {
            str(k): [float(x) for x in np.asarray(v).ravel()]
            for k, v in (embs or {}).items()}
    except Exception as e:       # 声纹是增强项：算不出来不影响标注结果
        result["embedding_error"] = f"{type(e).__name__}: {e}"
    result["ok"] = True
    return result


def _run_fast(args, result: dict) -> dict:
    """快速模式：回合切片 → 声纹向量 → 在线聚类 / 认熟人。"""
    from meeting_bot.config import load_settings
    from meeting_bot.fast_identify import classify, plan_windows
    from meeting_bot.voiceprints import VoiceprintLibrary, _build_embedder

    settings = load_settings()
    audio, sample_rate = _read_wav(args.wav)
    raw_turns = json.loads(Path(args.spans).read_text(encoding="utf-8")) \
        if args.spans else []
    turns = [(float(a), float(b)) for a, b in raw_turns if float(b) > float(a)]
    if not turns:                 # 没给回合（老调用 / 空会议）：整段当一个回合
        duration = len(audio) / max(1, sample_rate)
        turns = [(0.0, duration)] if duration > 0 else []
    windows = plan_windows(turns)
    embed = _build_embedder(settings.hf_token)
    vectors = []
    for _idx, start, end in windows:
        first = max(0, int(round(start * sample_rate)))
        last = min(audio.size, int(round(end * sample_rate)))
        if last <= first:
            vectors.append(None)
            continue
        try:
            vectors.append(embed(audio[first:last], sample_rate))
        except Exception:          # 单段太短 / 太吵：跳过，别拖垮整场
            vectors.append(None)
    threshold = args.threshold or 0.0
    kwargs = {"first_index": max(0, args.first_index)}
    if threshold > 0:
        kwargs["known_threshold"] = threshold
    library = None
    if getattr(settings, "voiceprint_enabled", True):
        try:
            library = VoiceprintLibrary()
        except Exception:
            library = None
    found = classify(windows, vectors, turns, library=library, **kwargs)
    result["spans"] = [[float(s), float(e), str(lbl)] for s, e, lbl in found.spans]
    result["names"] = dict(found.names)
    result["scores"] = dict(found.scores)
    result["counts"] = dict(found.counts)
    result["embeddings"] = dict(found.embeddings)
    result["fast"] = True
    result["ok"] = True
    return result


if __name__ == "__main__":
    sys.exit(main())
