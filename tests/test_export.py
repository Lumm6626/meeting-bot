# -*- coding: utf-8 -*-
from datetime import datetime

from docx import Document

from meeting_bot.export import export_all, meeting_dir, minutes_to_docx, write_transcript
from meeting_bot.transcript import TranscriptStore
from meeting_bot.translator import ActionItem, MinutesResult


def _settings(tmp_path):
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", tmp_path / "会议记录", tmp_path)


def test_meeting_dir_created(tmp_path):
    s = _settings(tmp_path)
    d = meeting_dir(s, "周会", datetime(2026, 10, 16, 9, 30))
    assert d.exists()
    assert d.name == "2026-10-16_0930_周会"


def test_write_transcript(tmp_path):
    s = _settings(tmp_path)
    d = meeting_dir(s, "周会", datetime(2026, 10, 16, 9, 30))
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")
    p = write_transcript(store, d, "周会", datetime(2026, 10, 16, 9, 30))
    content = p.read_text(encoding="utf-8")
    assert "hello" in content
    assert "译文" in content


def test_minutes_to_docx(tmp_path):
    m = MinutesResult(title="周会纪要", summary="概要",
                      key_points=["要点1"], decisions=["结论1"],
                      action_items=[ActionItem("写方案", "张三", "10/20")],
                      open_questions=["问题1"])
    p = minutes_to_docx(m, tmp_path / "纪要.docx")
    doc = Document(str(p))
    texts = "\n".join(par.text for par in doc.paragraphs)
    assert "周会纪要" in texts
    assert "张三" in texts
    assert "10/20" in texts


def test_export_all(tmp_path):
    s = _settings(tmp_path)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")

    class FakeTranslator:
        def minutes(self, md):
            return MinutesResult(title="周会纪要", summary="概要")

    t_path, m_path = export_all(store, s, "周会", datetime(2026, 10, 16, 9, 30),
                                translator=FakeTranslator())
    assert t_path.exists() and m_path.exists()
    assert t_path.suffix == ".md" and m_path.suffix == ".docx"
