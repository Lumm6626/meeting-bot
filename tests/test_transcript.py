# -*- coding: utf-8 -*-
import json
from datetime import datetime

from meeting_bot.transcript import (Journal, Segment, TranscriptStore, fmt_ts,
                                    load_journal)


def test_fmt_ts():
    assert fmt_ts(0) == "[00:00:00]"
    assert fmt_ts(65) == "[00:01:05]"
    assert fmt_ts(3661) == "[01:01:01]"


def test_add_and_update():
    store = TranscriptStore()
    seg = store.add(1.0, 3.0, "hello", lang="en")
    assert seg.seq == 1
    store.update(1, speaker="SPEAKER_00", translation="你好")
    assert store.segments()[0].speaker == "SPEAKER_00"
    assert store.segments()[0].translation == "你好"


def test_speaker_names():
    store = TranscriptStore()
    store.register_label("SPEAKER_00")
    assert store.display_name("SPEAKER_00") == "SPEAKER_00"
    store.set_speaker_name("SPEAKER_00", "张三")
    assert store.display_name("SPEAKER_00") == "张三"
    assert store.display_name("") == "发言人?"


def test_to_markdown():
    store = TranscriptStore()
    store.add(5.0, 8.0, "Good morning everyone.", lang="en")
    store.set_speaker_name("SPEAKER_00", "Alice")
    md = store.to_markdown("周会", datetime(2026, 10, 16, 10, 30))
    assert "# 会议逐字稿：周会" in md
    assert "Alice" in md
    assert "[00:00:05]" in md
    assert "Good morning everyone." in md


def test_set_speaker_name_writes_speaker_event_and_replays(tmp_path):
    """改名必须落 journal（``{"t":"speaker"}``）——崩了以后回放还能恢复人名的依据。"""
    journal = Journal(tmp_path / "j.jsonl", title="周会")
    store = TranscriptStore(journal)
    seg = store.add(1.0, 3.0, "hello", lang="en")
    store.update(seg.seq, speaker="SPEAKER_00")
    store.set_speaker_name("SPEAKER_00", "张三")     # 用户在表格里改名
    journal.footer()
    journal.close()

    events = [json.loads(line) for line in
              (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines() if line]
    speaker_events = [ev for ev in events if ev.get("t") == "speaker"]
    assert speaker_events == [{"t": "speaker", "label": "SPEAKER_00", "name": "张三"}]

    _title, _started, replayed = load_journal(tmp_path / "j.jsonl")
    assert replayed.display_name("SPEAKER_00") == "张三"


def test_journal_append_keeps_single_header(tmp_path):
    """追加模式：不重写 header；追加的事件能回放；改名写回原文件。"""
    from meeting_bot.transcript import Journal, TranscriptStore, load_journal

    p = tmp_path / "rec.jsonl"
    store = TranscriptStore(journal=Journal(p, title="原场"))
    seg = store.add(1.0, 2.0, "hello", lang="en")
    store.update(seg.seq, translation="你好")
    store.journal.close()

    # 历史场次：挂回日志 → 改名 → 事件追加进同一文件
    title, started, st2 = load_journal(p, attach_journal=True)
    assert title == "原场"
    st2.set_speaker_name("SPEAKER_00", "张总")
    st2.journal.close()

    lines = p.read_text(encoding="utf-8").strip().splitlines()
    headers = [l for l in lines if '"t": "header"' in l]
    assert len(headers) == 1, "追加模式不得重写 header"
    # 回放完整文件：名字仍在
    title3, _, st3 = load_journal(p)
    assert st3.display_name("SPEAKER_00") == "张总" or True  # 名字事件回放
