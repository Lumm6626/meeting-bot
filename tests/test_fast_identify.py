# -*- coding: utf-8 -*-
"""快速说话人识别：回合切分、采样、聚类、认熟人（全部用合成向量，不碰模型）。"""
import numpy as np

from meeting_bot.fast_identify import (IdentifyResult, agglomerate, build_turns,
                                       classify, plan_windows)
from meeting_bot.transcript import TranscriptStore


class _Seg:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def _unit(vec):
    return np.asarray(vec, dtype=np.float32) / np.linalg.norm(vec)


class FakeLibrary:
    """只实现 match 的声纹库替身（classify 只依赖这一个方法）。"""

    def __init__(self, people):
        self.people = {name: _unit(vec) for name, vec in people.items()}

    def match(self, vector, threshold=0.75):
        query = _unit(vector)
        hits = [(name, float(np.dot(query, vec)))
                for name, vec in self.people.items()
                if float(np.dot(query, vec)) >= threshold]
        hits.sort(key=lambda item: -item[1])
        return hits


def test_build_turns_merges_close_sentences():
    """停顿 ≤ 0.8s 的相邻句子并成一个发言回合（与逐字稿口径一致）。"""
    segs = [_Seg(0.0, 2.0), _Seg(2.5, 4.0), _Seg(9.0, 11.0)]
    assert build_turns(segs) == [(0.0, 4.0), (9.0, 11.0)]


def test_plan_windows_skips_very_short_turns():
    """短于 1.2 秒的片段声纹不可靠：不采样，让 classify 去补标签。"""
    turns = [(0.0, 4.0), (5.0, 5.4), (8.0, 12.0)]
    windows = plan_windows(turns)
    assert [idx for idx, _s, _e in windows] == [0, 2]
    for _idx, start, end in windows:
        assert end - start >= 1.2


def test_agglomerate_separates_three_voices():
    """三个人 → 三簇（余弦可分），且返回稳定编号。"""
    base = [_unit([1.0, 0.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0, 0.0]),
            _unit([0.0, 0.0, 1.0, 0.0])]
    vectors = []
    for b in base:
        for _ in range(3):
            vectors.append(_unit(b + 0.05 * np.random.randn(4)))
    labels = agglomerate(vectors)
    assert len(set(labels)) == 3
    assert labels[:3] == [labels[0]] * 3          # 同一声音的 3 段同簇
    assert labels[0] != labels[3] != labels[6]


def test_classify_names_known_speaker_and_leaves_stranger():
    """库里有 Peter → 他的簇被命名；陌生人只给 发言人B，不硬认。"""
    peter = _unit([1.0, 0.0, 0.0, 0.0])
    stranger = _unit([0.0, 1.0, 0.0, 0.0])
    library = FakeLibrary({"Peter Ren": peter})
    vectors = [peter, peter, peter, stranger, stranger, stranger]
    turns = [(i * 4.2, i * 4.2 + 4.0) for i in range(6)]  # 连续发言（间隔 0.2s）
    windows = [(i, turns[i][0], turns[i][1]) for i in range(6)]
    found = classify(windows, vectors, turns, library=library)

    assert isinstance(found, IdentifyResult)
    assert found.speakers == 2
    named = [label for label, name in found.names.items() if name == "Peter Ren"]
    assert len(named) == 1
    assert found.counts[named[0]] == 3
    unnamed = [label for label in found.counts if label not in found.names]
    assert len(unnamed) == 1                     # 陌生人留空，等人工命名
    assert len(found.spans) == 2                 # 同标签相邻回合合并成一条


def test_classify_merges_split_clusters_of_same_person():
    """同一人被切成两簇（音量/麦变了）→ 合并成同一个标签，避免两个 Peter。"""
    peter = _unit([1.0, 0.0, 0.0, 0.0])
    peter2 = _unit([0.95, 0.15, 0.0, 0.0])
    library = FakeLibrary({"Peter Ren": peter})
    vectors = [peter, peter, peter2, peter2]
    turns = [(i * 4.2, i * 4.2 + 4.0) for i in range(4)]
    windows = [(i, turns[i][0], turns[i][1]) for i in range(4)]
    found = classify(windows, vectors, turns, library=library)
    assert list(found.names.values()) == ["Peter Ren"]
    assert found.counts == {list(found.names)[0]: 4}


def test_classify_without_library_still_splits_speakers():
    """没有声纹库 / 库为空：仍要分出 发言人A/B（只是没有名字）。"""
    a = _unit([1.0, 0.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0, 0.0])
    vectors = [a, a, b, b]
    turns = [(i * 4.2, i * 4.2 + 4.0) for i in range(4)]
    windows = [(i, turns[i][0], turns[i][1]) for i in range(4)]
    found = classify(windows, vectors, turns, library=None)
    assert found.speakers == 2
    assert found.names == {}
    assert [span[2] for span in found.spans] == ["SPEAKER_00", "SPEAKER_01"]


def test_classify_ignores_missing_vectors_and_empty_input():
    assert classify([], [], [], library=None).spans == []
    turns = [(0.0, 4.0)]
    found = classify([(0, 0.0, 4.0)], [None], turns, library=None)
    assert found.spans == []                      # 算不出向量：不瞎标


def test_classify_speaker_labels_start_at_first_index():
    """新标签从 first_index 起编，避免与录制中已有的 SPEAKER_xx 撞车。"""
    a = _unit([1.0, 0.0, 0.0, 0.0])
    turns = [(0.0, 4.0), (4.2, 8.2)]
    windows = [(0, 0.0, 4.0), (1, 10.0, 14.0)]
    found = classify(windows, [a, _unit([0.0, 1.0, 0.0, 0.0])], turns,
                     library=None, first_index=3)
    assert [span[2] for span in found.spans] == ["SPEAKER_03", "SPEAKER_04"]


def test_classify_from_real_transcript_store():
    """端到端最小场景：store 里的句子 → 回合 → 认出两个人。"""
    store = TranscriptStore()
    for i in range(6):
        store.add(float(i * 5), float(i * 5 + 4), f"line {i}")
    turns = build_turns(store.segments())
    assert len(turns) == 6
    a = _unit([1.0, 0.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0, 0.0])
    vectors = [a, a, a, b, b, b]
    windows = plan_windows(turns)
    found = classify(windows, vectors, turns,
                     library=FakeLibrary({"张总": a}), first_index=0)
    assert sorted(found.names.values()) == ["张总"]
    assert found.speakers == 2
