# -*- coding: utf-8 -*-
"""会议记录缓冲：语句存储、说话人命名、Markdown 渲染。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def fmt_ts(seconds: float) -> str:
    """秒 → [HH:MM:SS]"""
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    lang: str = ""
    speaker: str = ""      # 声纹标签，如 SPEAKER_00
    translation: str = ""
    seq: int = 0           # 全局递增序号


@dataclass
class Speaker:
    label: str
    name: str = ""


class TranscriptStore:
    def __init__(self) -> None:
        self._segments: list[Segment] = []
        self._speakers: dict[str, Speaker] = {}
        self._next_seq = 1

    def add(self, start: float, end: float, text: str, lang: str = "") -> Segment:
        seg = Segment(start=start, end=end, text=text, lang=lang, seq=self._next_seq)
        self._next_seq += 1
        self._segments.append(seg)
        return seg

    def update(self, seq: int, **fields) -> None:
        for seg in self._segments:
            if seg.seq == seq:
                for k, v in fields.items():
                    if hasattr(seg, k):
                        setattr(seg, k, v)
                return

    def register_label(self, label: str) -> None:
        if label and label not in self._speakers:
            self._speakers[label] = Speaker(label=label)

    def set_speaker_name(self, label: str, name: str) -> None:
        self.register_label(label)
        self._speakers[label].name = name.strip()

    def display_name(self, label: str) -> str:
        if not label:
            return "发言人?"
        sp = self._speakers.get(label)
        return sp.name if sp and sp.name else label

    def segments(self) -> list[Segment]:
        return list(self._segments)

    def speakers(self) -> list[Speaker]:
        return list(self._speakers.values())

    def to_markdown(self, title: str, started_at: datetime) -> str:
        lines = [f"# 会议逐字稿：{title}", "",
                 f"> 会议时间：{started_at:%Y-%m-%d %H:%M}", "",
                 "## 发言人", ""]
        for sp in self._speakers.values():
            lines.append(f"- {sp.label}：{sp.name or '未命名'}")
        lines += ["", "## 正文", ""]
        for seg in self._segments:
            speaker = self.display_name(seg.speaker)
            lang = seg.lang or "?"
            lines.append(f"**{fmt_ts(seg.start)} · {speaker}（{lang}）**")
            lines.append(f"原文：{seg.text}")
            lines.append(f"译文：{seg.translation or '（未翻译）'}")
            lines.append("")
        return "\n".join(lines)
