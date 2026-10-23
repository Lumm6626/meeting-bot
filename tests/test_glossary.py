# -*- coding: utf-8 -*-
from meeting_bot.glossary import (CorrectionTable, GlossaryLibrary,
                                  extract_diff_pairs)


def test_glossary_add_top_and_clear(tmp_path):
    lib = GlossaryLibrary(tmp_path / "术语库.json")
    lib.add_many(["过敏门诊", "复合药房", "过敏门诊", "DPSP"])
    assert lib.size() == 3
    top = lib.top(2)
    assert top[0] == "过敏门诊"          # count 2 优先
    lib.save()
    assert (tmp_path / "术语库.json").is_file()
    lib2 = GlossaryLibrary(tmp_path / "术语库.json")
    assert lib2.size() == 3              # 往返一致
    lib.clear()
    assert lib.size() == 0


def test_glossary_ignores_noise(tmp_path):
    lib = GlossaryLibrary(tmp_path / "t.json")
    lib.add("  ")
    lib.add("，")
    lib.add("a")
    assert lib.size() == 0


def test_correction_needs_two_confirms(tmp_path):
    t = CorrectionTable(tmp_path / "纠错表.json")
    t.learn("全不能跳过", "这个不能跳过")
    assert t.active() == {}              # 1 次不生效
    t.learn("全不能跳过", "这个不能跳过")
    assert t.active() == {"全不能跳过": "这个不能跳过"}
    assert t.apply("流程全不能跳过，肯定") == "流程这个不能跳过，肯定"


def test_correction_equal_pair_not_learned_and_roundtrip(tmp_path):
    t = CorrectionTable(tmp_path / "c.json")
    t.learn("过敏门诊", "过敏门诊")       # 相同不学
    t.learn("全不能跳过", "这个不能跳过")
    t.learn("全不能跳过", "这个不能跳过")
    t.save()
    t2 = CorrectionTable(tmp_path / "c.json")
    assert t2.active() == {"全不能跳过": "这个不能跳过"}
    assert t2.apply("全不能跳过") == "这个不能跳过"


def test_extract_diff_pairs_heuristic():
    pairs = extract_diff_pairs("要走流程,这个是应该不能跳过这个downstream", "要走流程，这个是应该不能跳过这个 downstream。")
    assert pairs, "至少抽出一对"
    # 仅标点差异不构成纠错对
    assert extract_diff_pairs("要流程", "要流程") == []
