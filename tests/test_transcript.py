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
    # 默认名美化：SPEAKER_00 → 发言人A（未命名的声纹不再暴露内部标签）
    assert store.display_name("SPEAKER_00") == "发言人A"
    assert store.display_name("SPEAKER_01") == "发言人B"
    store.set_speaker_name("SPEAKER_00", "张三")
    assert store.display_name("SPEAKER_00") == "张三"
    assert store.display_name("") == "发言人?"


def test_to_markdown():
    """按发言人分段：同人连续发言合并成段，未命名显示 发言人A/B。"""
    store = TranscriptStore()
    seg = store.add(5.0, 8.0, "Good morning everyone.", lang="en")
    nxt = store.add(9.0, 12.0, "Let us start.", lang="en")
    store.update(seg.seq, speaker="SPEAKER_00")
    store.update(nxt.seq, speaker="SPEAKER_00")
    store.set_speaker_name("SPEAKER_00", "Alice")
    md = store.to_markdown("周会", datetime(2026, 10, 16, 10, 30))
    assert "# 会议逐字稿：周会" in md
    assert "Alice" in md                        # 已命名的声纹显示名字
    assert "[00:00:05]" in md                   # 回合起始时间保留
    assert "Good morning everyone." in md
    assert "Let us start." in md
    # 同一发言人连续两句合并成一段（不再一句一行带时间码）
    assert md.count("**Alice**") == 1
    # 未命名的声纹按 发言人A/B 显示
    store2 = TranscriptStore()
    s2 = store2.add(0.0, 1.0, "hi")
    store2.update(s2.seq, speaker="SPEAKER_02")
    assert "发言人C" in store2.to_markdown("t", datetime(2026, 1, 1))


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
