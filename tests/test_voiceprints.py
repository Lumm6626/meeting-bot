# -*- coding: utf-8 -*-
"""本地声纹库：cosine / 增删改查 / 阈值匹配 / 纠错剔除 / JSON 往返 / IO 容错。

全程纯内存 + tmp_path：不加载 pyannote / torch，不碰网络、不碰仓库里的真库。
"""
import json
import math

import numpy as np
import pytest

from meeting_bot import voiceprints as vp
from meeting_bot.voiceprints import (DEFAULT_THRESHOLD, MAX_VECTORS,
                                     VoiceprintLibrary, compute_speaker_embeddings,
                                     cosine)

LIB_NAME = "声纹库.json"


def _lib(tmp_path):
    return VoiceprintLibrary(tmp_path / LIB_NAME)


# ---------------------------------------------------------------------------
# cosine：纯函数
# ---------------------------------------------------------------------------

def test_cosine_basic_values():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """声纹比对只关心方向：同向不同长度必须算 1.0（不同录音音量不同）。"""
    assert cosine([1, 1], [2, 2]) == pytest.approx(1.0)
    assert cosine(np.array([3.0, 4.0]), np.array([6.0, 8.0])) == pytest.approx(1.0)


def test_cosine_guards_return_zero():
    """维度不同 / 零向量 / 空 / None 一律 0.0（绝不抛、绝不 NaN）。"""
    assert cosine([1, 0], [1, 0, 0]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine([], []) == 0.0
    assert cosine([1, 0], None) == 0.0


# ---------------------------------------------------------------------------
# add / 上限 / people
# ---------------------------------------------------------------------------

def test_add_and_meetings_counter(tmp_path):
    lib = _lib(tmp_path)
    assert lib.people() == []                        # 文件不存在 = 空库（不是错误）
    assert lib.last_error is None
    assert lib.add("张总", [1.0, 0.0]) is True
    assert lib.add("张总", [0.0, 1.0]) is True
    assert lib.people() == ["张总"]
    assert lib.meetings_of("张总") == 2              # 每次 add 记一场（库内简单计数）
    assert len(lib.vectors_of("张总")) == 2


def test_add_rejects_useless_input(tmp_path):
    lib = _lib(tmp_path)
    assert lib.add("", [1.0, 0.0]) is False          # 没名字 = 没人可记
    assert lib.add("张总", [0.0, 0.0]) is False      # 零向量算不出夹角
    assert lib.add("张总", [float("nan"), 1.0]) is False
    assert lib.add("张总", []) is False
    assert lib.people() == []


def test_add_keeps_at_most_max_vectors_dropping_oldest(tmp_path):
    lib = _lib(tmp_path)
    for i in range(MAX_VECTORS + 5):                 # 25 条
        lib.add("张总", [float(i + 1), 0.0])
    vectors = lib.vectors_of("张总")
    assert len(vectors) == MAX_VECTORS
    # 最旧的 5 条被丢掉：首条应是第 6 次写入的方向，末尾是最新一条
    assert cosine(vectors[0], [6.0, 0.0]) == pytest.approx(1.0)
    assert cosine(vectors[-1], [float(MAX_VECTORS + 5), 0.0]) == pytest.approx(1.0)
    assert lib.meetings_of("张总") == MAX_VECTORS + 5   # 场次照常累计（只截样本）


# ---------------------------------------------------------------------------
# match：每人取最像的一条、阈值过滤、分数降序
# ---------------------------------------------------------------------------

def test_match_uses_best_vector_per_person(tmp_path):
    """同一人存了多段样本时取「最像的那条」：max-pooling 比平均更抗设备差异。"""
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    lib.add("张总", [0.0, 1.0])                      # 完全不像的一条
    hits = lib.match([0.9, 0.1])
    assert [name for name, _ in hits] == ["张总"]
    assert hits[0][1] == pytest.approx(0.9938, abs=1e-3)


def test_match_threshold_filters_and_sorts_desc(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])                      # 1.00
    lib.add("王工", [1.0, 0.5])                      # 0.89
    lib.add("李四", [1.0, 1.0])                      # 0.71（低于默认阈值）
    hits = lib.match([1.0, 0.0])
    assert [name for name, _ in hits] == ["张总", "王工"]
    assert hits[0][1] > hits[1][1]
    assert all(score >= DEFAULT_THRESHOLD for _, score in hits)


def test_match_threshold_is_configurable(tmp_path):
    lib = _lib(tmp_path)
    lib.add("李四", [1.0, 1.0])                      # 0.707
    assert lib.match([1.0, 0.0], threshold=0.9) == []
    assert [n for n, _ in lib.match([1.0, 0.0], threshold=0.7)] == ["李四"]


def test_match_empty_library_and_bad_query(tmp_path):
    lib = _lib(tmp_path)
    assert lib.match([1.0, 0.0]) == []
    lib.add("张总", [1.0, 0.0])
    assert lib.match([0.0, 0.0]) == []                # 零向量 query：无从比对
    assert lib.match([1.0, 0.0, 0.0]) == []           # 维度不一致（模型换了）


# ---------------------------------------------------------------------------
# penalize：自动认错人 + 用户纠正 → 剔除误归属样本
# ---------------------------------------------------------------------------

def test_penalize_removes_the_most_similar_vector_only(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])          # 错归到张总的那条
    lib.add("张总", [0.0, 1.0])          # 真的是张总的另一场
    assert lib.penalize("张总", [0.99, 0.01]) is True
    vectors = lib.vectors_of("张总")
    assert len(vectors) == 1
    assert cosine(vectors[0], [0.0, 1.0]) == pytest.approx(1.0), "删错了样本"
    assert lib.meetings_of("张总") == 2, "剔除样本不该重置场次"


def test_penalize_drops_person_when_last_vector_removed(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    assert lib.penalize("张总", [1.0, 0.0]) is True
    assert lib.people() == []            # 删空 = 这个人从库里消失（不留空壳）
    assert lib.penalize("张总", [1.0, 0.0]) is False


def test_penalize_unknown_person_is_noop(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    assert lib.penalize("查无此人", [1.0, 0.0]) is False
    assert lib.penalize("张总", [0.0, 0.0]) is False      # 坏 query：不动库
    assert lib.people() == ["张总"]


def test_penalize_respects_tol_floor(tmp_path):
    """最相似度连 tol 都不到 = 毫不相干：不删（盲目删只会把真样本删掉）。"""
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    assert lib.penalize("张总", [0.0, 1.0], tol=0.15) is False
    assert len(lib.vectors_of("张总")) == 1
    assert lib.penalize("张总", [0.99, 0.01], tol=0.15) is True


# ---------------------------------------------------------------------------
# remove / rename
# ---------------------------------------------------------------------------

def test_remove_forgets_person(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    assert lib.remove("张总") is True
    assert lib.people() == []
    assert lib.remove("张总") is False


def test_rename_moves_archive(tmp_path):
    lib = _lib(tmp_path)
    lib.add("Zhang", [1.0, 0.0])
    assert lib.rename("Zhang", "张总") is True
    assert lib.people() == ["张总"]
    assert len(lib.vectors_of("张总")) == 1
    assert lib.rename("查无此人", "张总") is False
    assert lib.rename("张总", "张总") is False        # 同名：无事可做


def test_rename_merges_into_existing_person(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张三", [1.0, 0.0])
    lib.add("Zhang San", [0.0, 1.0])
    assert lib.rename("Zhang San", "张三") is True
    assert lib.people() == ["张三"]
    assert len(lib.vectors_of("张三")) == 2           # 两边样本都留着（不丢数据）
    assert lib.meetings_of("张三") == 1               # 合并取较多的一边，不虚增场次


# ---------------------------------------------------------------------------
# JSON 往返 / 格式
# ---------------------------------------------------------------------------

def test_json_roundtrip(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    lib.add("张总", [0.6, 0.8])
    lib.add("王工", [0.0, 1.0])
    assert lib.save() is True

    reloaded = _lib(tmp_path)
    assert reloaded.last_error is None
    assert reloaded.people() == ["张总", "王工"]
    assert len(reloaded.vectors_of("张总")) == 2
    assert reloaded.meetings_of("张总") == 2
    # 匹配行为在往返后完全一致
    assert [n for n, _ in reloaded.match([1.0, 0.0])] == \
           [n for n, _ in lib.match([1.0, 0.0])]


def test_saved_json_shape(tmp_path):
    lib = _lib(tmp_path)
    lib.add("张总", [1.0, 0.0])
    lib.save()
    data = json.loads((tmp_path / LIB_NAME).read_text(encoding="utf-8"))
    assert data["version"] == 1
    entry = data["people"]["张总"]
    assert set(entry) == {"vectors", "updated", "meetings"}
    assert isinstance(entry["vectors"], list) and isinstance(entry["vectors"][0], list)
    assert entry["meetings"] == 1
    raw = (tmp_path / LIB_NAME).read_text(encoding="utf-8")
    assert '"张总"' in raw, "中文必须原样写入（ensure_ascii=False）"
    assert (tmp_path / (LIB_NAME + ".tmp")).exists() is False, "临时文件没清掉"


def test_save_creates_parent_dir(tmp_path):
    lib = VoiceprintLibrary(tmp_path / "深" / "一层" / LIB_NAME)
    lib.add("张总", [1.0, 0.0])
    assert lib.save() is True
    assert (tmp_path / "深" / "一层" / LIB_NAME).is_file()


# ---------------------------------------------------------------------------
# IO 容错：坏文件 / 写不进去都只能「记下来」，绝不抛
# ---------------------------------------------------------------------------

def test_corrupt_file_becomes_empty_library(tmp_path):
    path = tmp_path / LIB_NAME
    path.write_text("{这不是 JSON", encoding="utf-8")
    lib = VoiceprintLibrary(path)
    assert lib.people() == []
    assert lib.last_error is not None                 # 记下来供排障
    assert path.exists(), "坏文件不该被删除（用户可以自己找回）"
    assert lib.add("张总", [1.0, 0.0]) is True        # 仍可继续用（内存态）
    assert lib.save() is True                         # 覆盖成合法 JSON


def test_unreadable_path_does_not_raise(tmp_path):
    """库路径是目录（或权限异常）：读失败只记 last_error，不抛。"""
    bad = tmp_path / LIB_NAME
    bad.mkdir()
    lib = VoiceprintLibrary(bad)
    assert lib.people() == []
    assert lib.last_error is not None


def test_save_failure_is_swallowed(tmp_path):
    bad = tmp_path / LIB_NAME
    bad.mkdir()                                       # 目标路径被目录占住 → os.replace 失败
    lib = VoiceprintLibrary(bad)
    lib.add("张总", [1.0, 0.0])
    assert lib.save() is False
    assert lib.last_error is not None


# ---------------------------------------------------------------------------
# compute_speaker_embeddings：按 label 聚合 → 平均 → 归一化（全程注入 fake embed）
# ---------------------------------------------------------------------------

SR = 10          # 「采样率」故意取小：切片边界一眼能算出来（1 秒 = 10 个样本）


def _audio(seconds: float, sr: int = SR) -> np.ndarray:
    return np.ones(int(seconds * sr), dtype=np.float32)


def test_embeddings_average_then_normalize():
    calls = []

    def embed(chunk, sr):
        calls.append((chunk.size, sr))
        return np.array([1.0, 0.0]) if len(calls) == 1 else np.array([0.0, 1.0])

    out = compute_speaker_embeddings(_audio(60), SR,
                                     [(0, 10, "A"), (20, 30, "A")], embed_fn=embed)
    assert calls == [(100, SR), (100, SR)]                  # 按段切片、带上采样率
    expected = [math.sqrt(0.5), math.sqrt(0.5)]             # 两段不同方向的平均
    assert np.allclose(out["A"], expected, atol=1e-6)
    assert float(np.linalg.norm(out["A"])) == pytest.approx(1.0, abs=1e-6)


def test_embeddings_cap_per_segment_and_per_person():
    """每段最多 20 秒、每人累计最多 60 秒：两小时的会不会白等几分钟。"""
    calls = []

    def embed(chunk, sr):
        calls.append(chunk.size)
        return np.array([1.0, 0.0])

    out = compute_speaker_embeddings(
        _audio(700), SR,
        [(0, 100, "A"),        # 截到 20s → 200 样本
         (200, 210, "A"),      # 10s → 100
         (300, 400, "A"),      # 还有 30s 额度 → 20s → 200
         (500, 510, "A"),      # 还剩 10s → 100（累计 60s）
         (600, 610, "A"),      # 额度用尽 → 跳过
         (0, 5, "B")],         # 另一个 label 独立计算
        embed_fn=embed)
    assert calls == [200, 100, 200, 100, 50]
    assert set(out) == {"A", "B"}


def test_embeddings_labels_are_independent():
    def embed(chunk, sr):
        return np.array([1.0, 0.0]) if chunk.size == 100 else np.array([0.0, 1.0])

    out = compute_speaker_embeddings(_audio(100), SR,
                                     [(0, 10, "A"), (20, 25, "B")], embed_fn=embed)
    assert set(out) == {"A", "B"}
    assert not np.allclose(out["A"], out["B"])


def test_embeddings_skip_broken_segments():
    """单段异常（太短 / 模型炸 / NaN）：跳过该段，其余照常平均；全坏则不出该人。"""
    state = {"n": 0}

    def embed(chunk, sr):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("这一段模型炸了")
        if state["n"] == 2:
            return np.array([float("nan"), 1.0])
        return np.array([0.0, 1.0])

    out = compute_speaker_embeddings(_audio(100), SR,
                                     [(0, 10, "A"), (10, 20, "A"), (20, 30, "A")],
                                     embed_fn=embed)
    assert np.allclose(out["A"], [0.0, 1.0], atol=1e-6)      # 只有第 3 段有效

    def always_boom(chunk, sr):
        raise RuntimeError("模型不可用")

    assert compute_speaker_embeddings(_audio(100), SR, [(0, 10, "A")],
                                      embed_fn=always_boom) == {}


def test_embeddings_guards():
    assert compute_speaker_embeddings(_audio(10), SR, [], embed_fn=lambda c, s: [1, 0]) == {}
    assert compute_speaker_embeddings(_audio(10), 0, [(0, 5, "A")]) == {}
    assert compute_speaker_embeddings(np.zeros(0, dtype=np.float32), SR,
                                      [(0, 5, "A")]) == {}
    assert compute_speaker_embeddings(_audio(10), SR, [(-5, -1, "A")],
                                      embed_fn=lambda c, s: [1, 0]) == {}
    assert compute_speaker_embeddings(_audio(10), SR, [(0, 5, "")],
                                      embed_fn=lambda c, s: [1, 0]) == {}


def test_embeddings_model_unavailable_returns_empty(monkeypatch):
    """真模型加载失败（没 token / 没缓存 / 断网）→ 空字典，调用方跳过认人即可。"""
    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(vp, "_build_embedder", boom)
    assert compute_speaker_embeddings(_audio(30), SR, [(0, 10, "A")]) == {}
