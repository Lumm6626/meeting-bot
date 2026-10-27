# -*- coding: utf-8 -*-
"""标记面板：示例发言展示 + 保存返回名字映射。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from meeting_bot.speaker_dialog import SpeakerLabelDialog

_app = QApplication.instance() or QApplication([])


def test_dialog_prefills_suggest_and_returns_names():
    dlg = SpeakerLabelDialog([{
        "label": "SPEAKER_00",
        "name": "",
        "samples": ["要走流程，这个不能跳过。", "　→ This needs the proper process."],
        "suggest": "张总",
        "score": 0.86,
        "count": 12,
        "span": "00:01:02 → 00:21:45",
    }], known_names=["张总", "李四"])
    edit = dlg._inputs["SPEAKER_00"]
    assert edit.currentText() == "张总"     # 声纹匹配建议预填
    # 声纹库已知姓名都在候选里（可直接选，不用重打）
    items = [edit.itemText(i) for i in range(edit.count())]
    assert "张总" in items and "李四" in items
    edit.setCurrentText("李四")
    dlg._on_save()
    assert dlg.names == {"SPEAKER_00": "李四"}


def test_dialog_keeps_placeholder_name_unchanged():
    """没改名字（仍是 发言人A）不算命名 —— 否则会把「发言人A」当人名存进声纹库。"""
    dlg = SpeakerLabelDialog([{"label": "SPEAKER_00", "name": "", "samples": [],
                               "suggest": "", "score": 0, "count": 1, "span": ""}])
    combo = dlg._inputs["SPEAKER_00"]
    combo.setCurrentText("发言人A")
    dlg._on_save()
    assert dlg.names == {}


def test_dialog_shows_pretty_speaker_label():
    """卡片标题用「发言人A/B」（与逐字稿一致），不暴露 SPEAKER_00。"""
    from PyQt6.QtWidgets import QLabel

    dlg = SpeakerLabelDialog([{"label": "SPEAKER_01", "name": "", "samples": [],
                               "suggest": "", "score": 0, "count": 2, "span": ""}])
    texts = [w.text() for w in dlg.findChildren(QLabel)]
    assert any("发言人B" in t for t in texts)
    assert not any("SPEAKER_01" in t for t in texts)


def test_dialog_empty_names_omitted():
    dlg = SpeakerLabelDialog([{"label": "SPEAKER_01", "samples": [], "suggest": "",
                               "count": 3, "span": ""}])
    dlg._on_save()
    assert dlg.names == {}               # 没填名字不提交


def test_segment_dialog_orders_columns_original_before_translation():
    """标注窗口与导出格式一致：时间 / 发言人 / 原文 / 译文（原文在前）。"""
    from meeting_bot.segment_label_dialog import SegmentLabelDialog
    from meeting_bot.transcript import TranscriptStore

    st = TranscriptStore()
    seg = st.add(5.0, 9.0, "Good morning everyone.", lang="en")
    st.update(seg.seq, speaker="SPEAKER_00", translation="大家早上好。")
    dlg = SegmentLabelDialog(st.segments(), display_resolver=st.display_name)

    assert [dlg.table.horizontalHeaderItem(i).text() for i in range(4)] == \
        ["时间", "发言人", "原文", "译文"]
    assert dlg.table.item(0, 2).text() == "Good morning everyone."   # 原文在前
    assert dlg.table.item(0, 3).text() == "大家早上好。"              # 译文在后
    # 未命名声纹显示 发言人A（不是内部标签）
    assert dlg.table.cellWidget(0, 1).currentText() == "发言人A"
    # 没改动 → 不产生 assignment
    dlg._on_save()
    assert dlg.assignments == {}


def test_segment_dialog_rows_sorted_by_time():
    from meeting_bot.segment_label_dialog import SegmentLabelDialog
    from meeting_bot.transcript import TranscriptStore

    st = TranscriptStore()
    for t in (30.0, 5.0, 60.0):
        st.add(t, t + 2, f"line {int(t)}")
    dlg = SegmentLabelDialog(st.segments())
    times = [dlg.table.item(r, 0).text() for r in range(dlg.table.rowCount())]
    assert times == sorted(times), "行未按时间顺序排列"
