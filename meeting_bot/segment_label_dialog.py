# -*- coding: utf-8 -*-
"""逐字稿手工标注发言人：把整场逐字稿摊开，逐句/批量指定是谁说的。

设计取舍（用户反馈）：既然逐字稿已经生成，标注就不该依赖"先跑几十分钟的
自动声纹聚类"。这里直接给一张表：时间 / 发言人（可编辑下拉） / 译文 / 原文，
支持多选行批量指定发言人，保存即回填整场并更新产物。

自动聚类（pyannote）降级为可选按钮，只用于"想先按声音自动分几组"的场景。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)


class SegmentLabelDialog(QDialog):
    """逐字稿标注：每行一句话，可在「发言人」列选择/输入名字。

    ``segments``：list[Segment]（会按现有顺序展示）
    ``known_names``：已出现过的发言人名字（下拉候选）
    返回值 ``self.assignments``：dict[seq -> 名字]（只含用户改动过的行）
    """

    def __init__(self, segments, known_names=(), parent=None, display_resolver=None):
        super().__init__(parent)
        self.setWindowTitle("标注发言人（按逐字稿）")
        self.resize(980, 720)
        self._segments = sorted(segments, key=lambda x: (x.start, x.seq))
        self._names = [n for n in dict.fromkeys(known_names) if n]
        self._display = display_resolver or (lambda label: label or "发言人?")
        self.assignments: dict[int, str] = {}

        root = QVBoxLayout(self)

        tip = QLabel(
            "每条一行：直接在「发言人」列选择或输入姓名（如：张总）；"
            "也可以先按住 Ctrl/Shift 多选多行，再在上方输入名字点「应用到选中行」。"
            "保存后整场逐字稿与纪要都会更新。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #8B93A7; font-size: 9pt;")
        root.addWidget(tip)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("批量指定："))
        self.bulk_edit = QComboBox()
        self.bulk_edit.setEditable(True)
        self.bulk_edit.addItems(self._names)
        self.bulk_edit.setMinimumWidth(180)
        bar.addWidget(self.bulk_edit)
        self.bulk_btn = QPushButton("应用到选中行")
        bar.addWidget(self.bulk_btn)
        bar.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #8B93A7; font-size: 8.5pt;")
        bar.addWidget(self.count_label)
        root.addLayout(bar)

        self.table = QTableWidget(len(self._segments), 4)
        # 顺序与导出的逐字稿一致：原文在前、译文在后
        self.table.setHorizontalHeaderLabels(["时间", "发言人", "原文", "译文"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(True)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self._combos: dict[int, QComboBox] = {}
        for row, seg in enumerate(self._segments):
            self.table.setItem(row, 0, self._cell(f"{int(seg.start)//60:02d}:{int(seg.start)%60:02d}"))
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(self._names)
            current = seg.speaker or ""
            # 未命名时显示「发言人A/B」（与逐字稿一致），而不是内部标签 SPEAKER_00
            combo.setCurrentText(self._display(current) if current else "")
            self.table.setCellWidget(row, 1, combo)
            self._combos[seg.seq] = combo
            self.table.setItem(row, 2, self._cell((seg.text or "").strip()))
            translated = (seg.translation or "").strip()
            self.table.setItem(row, 3,
                               self._cell(translated if translated != (seg.text or "").strip() else ""))
            self.table.setRowHeight(row, 46)

        root.addWidget(self.table, 1)

        self.bulk_btn.clicked.connect(self._apply_bulk)
        self.table.itemSelectionChanged.connect(self._update_count)

        btns = QHBoxLayout()
        self.cancel_btn = QPushButton("取消")
        self.save_btn = QPushButton("保存并回填整场")
        self.save_btn.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #2A7DE1, stop:1 #1FB6B8); color: white; border-radius: 8px;"
            "padding: 10px 18px; }")
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.save_btn)
        root.addLayout(btns)
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self._on_save)

    @staticmethod
    def _cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def _update_count(self) -> None:
        n = len({i.row() for i in self.table.selectedIndexes()})
        self.count_label.setText(f"已选 {n} 行" if n else "")

    def _apply_bulk(self) -> None:
        name = self.bulk_edit.currentText().strip()
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not name or not rows:
            return
        for row in rows:
            seg = self._segments[row]
            self._combos[seg.seq].setCurrentText(name)

    def _on_save(self) -> None:
        for seg in self._segments:
            combo = self._combos.get(seg.seq)
            if combo is None:
                continue
            name = combo.currentText().strip()
            if not name:
                continue
            if name == self._display(seg.speaker or ""):
                continue          # 没改动（仍是 发言人A / 原名）
            self.assignments[seg.seq] = name
        self.accept()
