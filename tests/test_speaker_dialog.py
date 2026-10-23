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
        "samples": ["要走流程，这个不能跳过。\n  〔原文〕 要走流程,这个不能跳过"],
        "suggest": "张总",
        "count": 12,
        "span": "00:01:02 → 00:21:45",
    }])
    edit = dlg._inputs["SPEAKER_00"]
    assert edit.text() == "张总"          # 声纹建议预填
    edit.setText("李四")
    dlg._on_save()
    assert dlg.names == {"SPEAKER_00": "李四"}


def test_dialog_empty_names_omitted():
    dlg = SpeakerLabelDialog([{"label": "SPEAKER_01", "samples": [], "suggest": "",
                               "count": 3, "span": ""}])
    dlg._on_save()
    assert dlg.names == {}               # 没填名字不提交
