# -*- coding: utf-8 -*-
"""导出：逐字稿 Markdown + 纪要 Word。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from docx import Document

from .config import Settings
from .transcript import TranscriptStore
from .translator import MinutesResult, Translator


def meeting_dir(settings: Settings, title: str, started_at: datetime) -> Path:
    d = settings.output_dir / f"{started_at:%Y-%m-%d_%H%M}_{title}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_transcript(store: TranscriptStore, directory: Path,
                     title: str, started_at: datetime) -> Path:
    path = directory / "逐字稿.md"
    path.write_text(store.to_markdown(title, started_at), encoding="utf-8")
    return path


def minutes_to_docx(minutes: MinutesResult, path: Path) -> Path:
    doc = Document()
    doc.add_heading(minutes.title, level=1)
    doc.add_paragraph(minutes.summary)
    doc.add_heading("讨论要点", level=2)
    for p in minutes.key_points:
        doc.add_paragraph(p, style="List Bullet")
    doc.add_heading("会议结论", level=2)
    for p in minutes.decisions:
        doc.add_paragraph(p, style="List Bullet")
    doc.add_heading("行动项", level=2)
    for a in minutes.action_items:
        due = f"（截止：{a.due}）" if a.due else ""
        doc.add_paragraph(f"{a.task} — 负责人：{a.owner or '待定'}{due}", style="List Bullet")
    doc.add_heading("待跟进问题", level=2)
    for p in minutes.open_questions:
        doc.add_paragraph(p, style="List Bullet")
    doc.save(str(path))
    return path


def export_all(store: TranscriptStore, settings: Settings, title: str,
               started_at: datetime,
               translator: Translator | None = None) -> tuple[Path, Path]:
    directory = meeting_dir(settings, title, started_at)
    transcript_path = write_transcript(store, directory, title, started_at)
    if translator is not None:
        minutes = translator.minutes(store.to_markdown(title, started_at))
    else:
        minutes = MinutesResult(title=title, summary="（翻译接口不可用，未生成纪要）")
    minutes_path = minutes_to_docx(minutes, directory / "纪要.docx")
    return transcript_path, minutes_path
