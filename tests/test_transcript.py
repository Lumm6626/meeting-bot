# -*- coding: utf-8 -*-
from datetime import datetime

from meeting_bot.transcript import Segment, TranscriptStore, fmt_ts


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
