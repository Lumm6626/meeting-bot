# -*- coding: utf-8 -*-
"""字幕浮窗：置顶、半透明、可拖拽、滚轮调字号。"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .transcript import Segment


class SubtitleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 160); border-radius: 10px;")
        self._font_size = 20
        self._speaker = QLabel("发言人")
        self._zh = QLabel("")
        self._orig = QLabel("")
        self._speaker.setStyleSheet("color: #4FC3F7; font-weight: bold; background: transparent;")
        self._zh.setStyleSheet("color: white; background: transparent;")
        self._orig.setStyleSheet("color: #B0BEC5; background: transparent;")
        for w in (self._speaker, self._zh, self._orig):
            w.setWordWrap(True)
        lay = QVBoxLayout(self)
        for w in (self._speaker, self._zh, self._orig):
            lay.addWidget(w)
        self.resize(720, 180)
        self._apply_font()
        self._drag = None

    def _apply_font(self):
        f = self.font()
        f.setPointSize(self._font_size)
        self._speaker.setFont(f)
        self._zh.setFont(f)
        f2 = self.font()
        f2.setPointSize(max(10, self._font_size - 4))
        self._orig.setFont(f2)

    def show_segment(self, seg: Segment, display_name: str):
        self._speaker.setText(display_name)
        self._zh.setText(seg.translation or seg.text)
        self._orig.setText(f"({seg.lang}) {seg.text}" if seg.translation else "")
        self.adjustSize()
        self.show()
        self.raise_()

    def change_font(self, delta: int):
        self._font_size = max(12, min(40, self._font_size + delta))
        self._apply_font()

    # --- 拖拽 ---
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        self.change_font(1 if e.angleDelta().y() > 0 else -1)
