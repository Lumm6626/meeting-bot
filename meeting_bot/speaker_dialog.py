# -*- coding: utf-8 -*-
"""说话人标记面板：按逐字稿内容给每个声纹起名。

会后标注完成后，主窗口弹出本面板：每个声纹一行卡片，显示
「该声纹的示例发言（原文+译文）」+ 名字输入框（有声纹库建议时预填）。
用户看着内容判断谁是谁，填完保存 → 整场回填。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)


class SpeakerLabelDialog(QDialog):
    """给声纹起名（内容辅助）。

    ``speakers``：list[dict]，每项
        label   : 声纹标签（SPEAKER_00）
        samples : 该声纹的示例发言（原文 / 译文），最多 3 条
        suggest : 声纹库建议名（可空）
        count   : 该声纹的总句数
        span    : 首句→末句时间（如 "00:03:12 → 00:21:45"）
    """

    def __init__(self, speakers: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("标记说话人（对照逐字稿内容）")
        self.resize(560, 520)
        self._inputs: dict[str, QLineEdit] = {}
        self.names: dict[str, str] = {}

        root = QVBoxLayout(self)
        tip = QLabel("下面每位发言人列出了 TA 说过的几句话，请对照内容填写姓名；"
                     "留空则保持 SPEAKER_xx。保存后整场逐字稿与纪要会全部替换。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #8B93A7; font-size: 9pt;")
        root.addWidget(tip)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setSpacing(10)
        for sp in speakers:
            body_lay.addWidget(self._card(sp))
        body_lay.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        btns = QHBoxLayout()
        self.save_btn = QPushButton("保存并回填整场")
        self.save_btn.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #2A7DE1, stop:1 #1FB6B8); color: white; border-radius: 8px;"
            "padding: 10px 18px; }")
        self.skip_btn = QPushButton("先不标（保持 SPEAKER_xx）")
        btns.addWidget(self.skip_btn, 1)
        btns.addWidget(self.save_btn, 1)
        root.addLayout(btns)

        self.save_btn.clicked.connect(self._on_save)
        self.skip_btn.clicked.connect(self.reject)

    def _card(self, sp: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: #232839; border: 1px solid #343B4F;"
            "border-radius: 8px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        head = QHBoxLayout()
        label_text = str(sp.get("label", ""))
        if sp.get("suggest"):
            label_text += f"　≈ 疑似 {sp['suggest']}"
        title = QLabel(label_text)
        title.setStyleSheet("color: #4FC3F7; font-weight: bold;")
        meta = QLabel(f"共 {int(sp.get('count', 0))} 句"
                      + (f" · {sp['span']}" if sp.get("span") else ""))
        meta.setStyleSheet("color: #8B93A7; font-size: 8.5pt;")
        head.addWidget(title, 1)
        head.addWidget(meta, 0)
        lay.addLayout(head)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("输入姓名（如：张总 / Alex）")
        if sp.get("suggest"):
            name_edit.setText(str(sp["suggest"]))
        name_edit.setStyleSheet(
            "QLineEdit { background: #1B1E2B; border: 1px solid #343B4F;"
            "border-radius: 6px; padding: 6px 10px; color: #E8EAF2; }")
        lay.addWidget(name_edit)
        self._inputs[str(sp.get("label", ""))] = name_edit

        for sample in (sp.get("samples") or [])[:3]:
            lab = QLabel(sample)
            lab.setWordWrap(True)
            lab.setStyleSheet("color: #B0BEC5; font-size: 8.5pt;")
            lay.addWidget(lab)
        return card

    def _on_save(self) -> None:
        self.names = {
            label: edit.text().strip()
            for label, edit in self._inputs.items()
            if edit.text().strip()
        }
        self.accept()
