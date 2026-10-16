# -*- coding: utf-8 -*-
from meeting_bot.diarization import assign_speakers
from meeting_bot.transcript import Segment


def test_assign_by_overlap():
    segs = [Segment(0.0, 4.0, "hello", lang="en"),
            Segment(5.0, 9.0, "world", lang="en")]
    spans = [(0.0, 4.5, "SPEAKER_00"), (5.5, 10.0, "SPEAKER_01")]
    assign_speakers(segs, spans)
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[1].speaker == "SPEAKER_01"


def test_no_overlap_stays_unlabeled():
    segs = [Segment(0.0, 4.0, "hello")]
    spans = [(10.0, 20.0, "SPEAKER_00")]
    assign_speakers(segs, spans)
    assert segs[0].speaker == ""


def test_threshold_blocks_weak_overlap():
    segs = [Segment(0.0, 4.0, "hello")]
    spans = [(3.6, 4.0, "SPEAKER_00")]  # 只覆盖 10% < 30%
    assign_speakers(segs, spans)
    assert segs[0].speaker == ""


def test_exact_threshold_is_assigned():
    segs = [Segment(0.0, 7.0, "hello")]
    assign_speakers(segs, [(4.9, 7.0, "SPEAKER_00")])
    assert segs[0].speaker == "SPEAKER_00"


def test_span_containing_segment_is_assigned():
    segs = [Segment(5.0, 9.0, "hello")]
    assign_speakers(segs, [(0.0, 26.0, "SPEAKER_00")])
    assert segs[0].speaker == "SPEAKER_00"
