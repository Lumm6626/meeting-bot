# -*- coding: utf-8 -*-
"""导出：逐字稿 Markdown + 纪要 Word。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from docx import Document

from .config import Settings
from .transcript import TranscriptStore
from .translator import MinutesResult, TranslationError, Translator

MINUTES_FAILED_SUMMARY = "（纪要生成失败，请查看逐字稿）"
NO_TRANSLATOR_SUMMARY = "（翻译接口不可用，未生成纪要）"


def _safe_title(title: str) -> str:
    r"""净化标题：替换路径分隔符与 Windows 非法字符，防止路径穿越。

    - `..` / `../` 中的斜杠与点被处理，标题无法逃出 output_dir；
    - `\ / : * ? " < > |` 与换行、制表符替换为 `_`，避免 OSError；
    - 净化后为空（空串或纯点）回退「未命名会议」，最长 80 字符。
    """
    t = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", title).strip().strip(".") or "未命名会议"
    return t[:80]


def meeting_dir(settings: Settings, title: str, started_at: datetime) -> Path:
    d = settings.output_dir / f"{started_at:%Y-%m-%d_%H%M}_{_safe_title(title)}"
    # 防线：净化后仍校验落点包含关系，且先校验后建目录（越界不留半成品）
    assert d.resolve().is_relative_to(settings.output_dir.resolve())
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_transcript(store: TranscriptStore, directory: Path,
                     title: str, started_at: datetime,
                     md_text: str | None = None) -> Path:
    """写逐字稿；md_text 非空时复用调用方缓存的 Markdown，避免重复渲染。"""
    path = directory / "逐字稿.md"
    path.write_text(md_text if md_text is not None else store.to_markdown(title, started_at),
                    encoding="utf-8")
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
        task = (a.task or "").strip() or "（未命名任务）"
        doc.add_paragraph(f"{task} — 负责人：{a.owner or '待定'}{due}", style="List Bullet")
    doc.add_heading("待跟进问题", level=2)
    for p in minutes.open_questions:
        doc.add_paragraph(p, style="List Bullet")
    doc.save(str(path))
    return path


def export_all(store: TranscriptStore, settings: Settings, title: str,
               started_at: datetime,
               translator: Translator | None = None) -> tuple[Path, Path]:
    directory = meeting_dir(settings, title, started_at)
    md = store.to_markdown(title, started_at)
    transcript_path = write_transcript(store, directory, title, started_at, md_text=md)
    if translator is not None:
        try:
            minutes = translator.minutes(md)
        except TranslationError:
            # 纪要失败不阻断导出：逐字稿已落盘，纪要降级为占位内容
            minutes = MinutesResult(title=title, summary=MINUTES_FAILED_SUMMARY)
    else:
        minutes = MinutesResult(title=title, summary=NO_TRANSLATOR_SUMMARY)
    minutes_path = minutes_to_docx(minutes, directory / "纪要.docx")
    return transcript_path, minutes_path
