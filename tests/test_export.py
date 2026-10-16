# -*- coding: utf-8 -*-
from datetime import datetime

from docx import Document

from meeting_bot.export import export_all, meeting_dir, minutes_to_docx, write_transcript
from meeting_bot.transcript import TranscriptStore
from meeting_bot.translator import ActionItem, MinutesResult, TranslationError

# Windows 非法文件名字符（chr(92) 即反斜杠）
_ILLEGAL_CHARS = set('/:*?"<>|') | {chr(92)}


def _settings(tmp_path):
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", tmp_path / "会议记录", tmp_path)


def _docx_text(path):
    return "\n".join(par.text for par in Document(str(path)).paragraphs)


class _FakeTranslator:
    def minutes(self, md):
        return MinutesResult(title="周会纪要", summary="概要")


class _FailTranslator:
    def minutes(self, md):
        raise TranslationError("接口不可用")


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
    texts = _docx_text(p)
    assert "周会纪要" in texts
    assert "张三" in texts
    assert "10/20" in texts


def test_export_all(tmp_path):
    s = _settings(tmp_path)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")

    t_path, m_path = export_all(store, s, "周会", datetime(2026, 10, 16, 9, 30),
                                translator=_FakeTranslator())
    assert t_path.exists() and m_path.exists()
    assert t_path.suffix == ".md" and m_path.suffix == ".docx"


def test_export_all_no_translator(tmp_path):
    """translator=None：纪要降级占位，逐字稿仍正常产出。"""
    s = _settings(tmp_path)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")

    t_path, m_path = export_all(store, s, "周会", datetime(2026, 10, 16, 9, 30))
    assert t_path.exists() and m_path.exists()
    assert "hello" in t_path.read_text(encoding="utf-8")
    assert "未生成纪要" in _docx_text(m_path)


def test_transcript_translation_included(tmp_path):
    """译文真实写入逐字稿（替代恒真的「译文」断言的实证用例）。"""
    s = _settings(tmp_path)
    d = meeting_dir(s, "周会", datetime(2026, 10, 16, 9, 30))
    store = TranscriptStore()
    seg = store.add(0.0, 2.0, "hello", lang="en")
    store.update(seg.seq, translation="你好")

    content = write_transcript(store, d, "周会", datetime(2026, 10, 16, 9, 30)).read_text(
        encoding="utf-8")
    assert "你好" in content
    assert "（未翻译）" not in content


def test_docx_all_sections(tmp_path):
    """docx 五个区块齐全，且各区块内容落到正文。"""
    m = MinutesResult(title="周会纪要", summary="概要",
                      key_points=["要点甲"], decisions=["结论乙"],
                      action_items=[ActionItem("写方案", "张三", "10/20")],
                      open_questions=["问题丙"])
    texts = _docx_text(minutes_to_docx(m, tmp_path / "纪要.docx"))
    for heading in ("讨论要点", "会议结论", "行动项", "待跟进问题"):
        assert heading in texts
    for content in ("要点甲", "结论乙", "问题丙", "写方案"):
        assert content in texts


def test_action_item_empty_task_placeholder(tmp_path):
    """行动项 task 为空时用占位文案，不留悬挂破折号。"""
    m = MinutesResult(title="周会纪要", summary="概要",
                      action_items=[ActionItem("", "张三"), ActionItem("  ", "")])
    texts = _docx_text(minutes_to_docx(m, tmp_path / "纪要.docx"))
    action_lines = [ln for ln in texts.splitlines() if "负责人：" in ln]
    assert len(action_lines) == 2
    assert all(ln.startswith("（未命名任务） — ") for ln in action_lines)  # 无悬挂破折号
    assert "负责人：待定" in texts


def test_export_all_minutes_failure_degrades(tmp_path):
    """纪要接口抛 TranslationError：不抛穿，逐字稿完整、纪要降级占位。"""
    s = _settings(tmp_path)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")

    t_path, m_path = export_all(store, s, "周会", datetime(2026, 10, 16, 9, 30),
                                translator=_FailTranslator())
    assert t_path.exists() and m_path.exists()
    assert "hello" in t_path.read_text(encoding="utf-8")
    assert "纪要生成失败" in _docx_text(m_path)


def test_export_all_renders_markdown_once(tmp_path, monkeypatch):
    """逐字稿与纪要复用同一份 Markdown，to_markdown 只渲染一次。"""
    s = _settings(tmp_path)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")
    calls = {"n": 0}
    original = store.to_markdown

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "to_markdown", counting)
    export_all(store, s, "周会", datetime(2026, 10, 16, 9, 30), translator=_FakeTranslator())
    assert calls["n"] == 1


def test_unsafe_title_sanitized(tmp_path):
    """title 含非法字符 / 路径分隔符：目录名净化且产物不逃出 output_dir。"""
    s = _settings(tmp_path)
    out = s.output_dir.resolve()

    d = meeting_dir(s, "A/B:C*方案", datetime(2026, 10, 16, 9, 30))
    assert d.exists()
    assert not set(d.name) & _ILLEGAL_CHARS
    assert "A_B_C_方案" in d.name
    assert d.resolve().is_relative_to(out)

    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")
    t_path, m_path = export_all(store, s, "../../evil", datetime(2026, 10, 16, 9, 30))
    assert t_path.exists() and m_path.exists()
    assert t_path.resolve().is_relative_to(out)
    assert m_path.resolve().is_relative_to(out)
    assert t_path.parent.resolve().is_relative_to(out)
    assert not set(t_path.parent.name) & _ILLEGAL_CHARS  # 净化后标题无法再建出子目录


def test_unsafe_title_backslash_traversal(tmp_path):
    """Windows 反斜杠同样是路径分隔符：必须一并净化，杜绝越界落盘。"""
    s = _settings(tmp_path)
    out = s.output_dir.resolve()
    bs = chr(92)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")
    leaked_name = "evil_r1_task11"
    for bad in (f"x{bs}..{bs}..{bs}..{bs}{leaked_name}", f"{bs}absolute", f"A{bs}B"):
        t_path, m_path = export_all(store, s, bad, datetime(2026, 10, 16, 9, 30))
        assert t_path.resolve().is_relative_to(out)
        assert m_path.resolve().is_relative_to(out)
        assert not (_ILLEGAL_CHARS & set(t_path.parent.name))
    assert not (tmp_path.parent / leaked_name).exists()


def test_unsafe_title_illegal_chars_no_crash(tmp_path):
    """换行 / 制表符 / 纯点 / 空标题不再崩溃，回退到安全目录名。"""
    s = _settings(tmp_path)
    out = s.output_dir.resolve()
    for bad, expect in (("line\nbreak", "line_break"),
                        ("tab\there", "tab_here"),
                        ("", "未命名会议"),
                        ("...", "未命名会议")):
        d = meeting_dir(s, bad, datetime(2026, 10, 16, 9, 30))
        assert d.exists()
        assert d.resolve().is_relative_to(out)
        assert d.name.endswith(expect)
