# -*- coding: utf-8 -*-
"""说话人标记面板：**按声纹认人**，一位参会者一张卡。

会后声纹标注完成（或打开历史会议）时弹出本面板：

- 每个声纹一张卡：`发言人A` + 句数 / 时间跨度 + **声纹库匹配分数**（≈ 张总 87%）；
- 名字用可编辑下拉框：已认识的姓名直接选，新人手输一次；
- 示例发言（原文在前、译文在后）帮用户判断「这是谁的声音」；
- 保存 → 整场回填姓名，同时把该声纹向量学进声纹库（下次自动认人）。
"""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)


def _pretty_label(label: str) -> str:
    """SPEAKER_00 → 发言人A（与逐字稿里的显示保持一致）。"""
    text = str(label or "").strip()
    if text.upper().startswith("SPEAKER_"):
        tail = text.split("_", 1)[1]
        if tail.isdigit():
            index = int(tail)
            letter = chr(ord("A") + index) if index < 26 else str(index + 1)
            return f"发言人{letter}"
    return text or "发言人?"


class SpeakerLabelDialog(QDialog):
    """给声纹起名（内容 + 声纹双重辅助）。

    ``speakers``：list[dict]，每项
        label   : 声纹标签（SPEAKER_00）
        name    : 当前姓名（可空）
        samples : 该声纹的示例发言（原文 / 译文交替的若干行）
        suggest : 声纹库建议名（可空）
        score   : 建议的相似度（0~1，可空）
        count   : 该声纹的总句数
        span    : 首句→末句时间（如 "00:03:12 → 00:21:45"）
    """

    def __init__(self, speakers: list[dict], parent=None, known_names=()):
        super().__init__(parent)
        self.setWindowTitle("标记发言人（按声纹认人）")
        self.resize(600, 560)
        self._inputs: dict[str, QComboBox] = {}
        self._original: dict[str, str] = {}
        self._pretty: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self._known = [str(n) for n in dict.fromkeys(known_names) if str(n).strip()]

        root = QVBoxLayout(self)
        tip = QLabel(
            "下面每一位是**一段声纹**（自动分出的一位参会者）。填一次名字，"
            "TA 在这场说的所有话都会一起改；同时这个声音会记住，下次开会自动认出。"
            "留空则保持「发言人A/B」。")
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
        self.skip_btn = QPushButton("先不标（保持 发言人A/B）")
        btns.addWidget(self.skip_btn, 1)
        btns.addWidget(self.save_btn, 1)
        root.addLayout(btns)

        self.save_btn.clicked.connect(self._on_save)
        self.skip_btn.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    def _card(self, sp: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: #232839; border: 1px solid #343B4F;"
            "border-radius: 8px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        label = str(sp.get("label", ""))
        pretty = _pretty_label(label)
        current = str(sp.get("name") or "").strip()
        self._original[label] = current
        self._pretty[label] = pretty

        # ---- 标题行：发言人A · 句数 / 时间跨度 ----
        head = QHBoxLayout()
        title = QLabel(pretty if not current else f"{pretty}　({current})")
        title.setStyleSheet("color: #4FC3F7; font-weight: bold;")
        meta = QLabel(f"共 {int(sp.get('count', 0))} 句"
                      + (f" · {sp['span']}" if sp.get("span") else ""))
        meta.setStyleSheet("color: #8B93A7; font-size: 8.5pt;")
        head.addWidget(title, 1)
        head.addWidget(meta, 0)
        lay.addLayout(head)

        # ---- 声纹匹配：≈ 张总（相似度 87%）----
        suggest = str(sp.get("suggest") or "").strip()
        if suggest:
            try:
                score = float(sp.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            pct = f"（相似度 {score * 100:.0f}%）" if score > 0 else ""
            hint = QLabel(f"声纹库比对：疑似 {suggest}{pct}")
            hint.setStyleSheet("color: #7FD1AE; font-size: 8.5pt;")
            lay.addWidget(hint)

        # ---- 名字：可编辑下拉，已认识的姓名可直接选 ----
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItem("")                      # 空 = 保持 发言人A/B
        for name in self._known:
            combo.addItem(name)
        if current:
            combo.setCurrentText(current)
        elif suggest:
            combo.setCurrentText(suggest)      # 声纹匹配置信度够高时预填
        combo.setPlaceholderText("选择或输入姓名（如：张总 / Alex）")
        combo.setStyleSheet(
            "QComboBox { background: #1B1E2B; border: 1px solid #343B4F;"
            "border-radius: 6px; padding: 6px 10px; color: #E8EAF2; }")
        lay.addWidget(combo)
        self._inputs[label] = combo

        for sample in (sp.get("samples") or [])[:6]:
            lab = QLabel(str(sample))
            lab.setWordWrap(True)
            lab.setStyleSheet("color: #B0BEC5; font-size: 8.5pt;")
            lay.addWidget(lab)
        return card

    # ------------------------------------------------------------------
    def _on_save(self) -> None:
        """收集「有改动」的命名（没动的保持原样，避免把 发言人A 当成人名存进去）。"""
        for label, combo in self._inputs.items():
            picked = combo.currentText().strip()
            if not picked:
                continue
            if picked == self._original.get(label, "") or picked == self._pretty.get(label, ""):
                continue
            self.names[label] = picked
        self.accept()
