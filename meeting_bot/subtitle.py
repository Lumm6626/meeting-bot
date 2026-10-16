# -*- coding: utf-8 -*-
"""字幕浮窗：置顶、半透明、可拖拽、滚轮调字号。"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .transcript import Segment

# 半透明黑底参数（样式表与 paintEvent 共用，避免两处硬编码漂移）
BG_RGBA = (0, 0, 0, 160)
BG_RADIUS = 10
BG_STYLE = (
    f"background-color: rgba({BG_RGBA[0]}, {BG_RGBA[1]}, {BG_RGBA[2]}, {BG_RGBA[3]}); "
    f"border-radius: {BG_RADIUS}px;"
)


class SubtitleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # QWidget 子类默认不绘制样式表背景，需显式开启 WA_StyledBackground
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(BG_STYLE)
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

    def paintEvent(self, e):
        """自绘半透明黑底圆角。

        WA_TranslucentBackground 的顶层窗口 Qt 不会绘制任何背景（样式表背景
        同样不画，实测 alpha 全 0），故必须在此手动绘制。用 CompositionMode_Source
        覆盖式写入，保证无论绘制几次，结果都是精确的 rgba(0,0,0,160)。
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(*BG_RGBA))
        p.drawRoundedRect(QRectF(self.rect()), BG_RADIUS, BG_RADIUS)
        p.end()

    def _apply_font(self):
        f = self.font()
        f.setPointSize(self._font_size)
        self._speaker.setFont(f)
        self._zh.setFont(f)
        f2 = self.font()
        f2.setPointSize(max(10, self._font_size - 4))
        self._orig.setFont(f2)

    def _relayout(self):
        """按固定宽度重算窗口高度。

        wordWrap 下 adjustSize() 与 heightForWidth 相互依赖，会算出偏矮的高度；
        改为定宽 + 取 heightForWidth 与 sizeHint 的较大值，长句不再被裁切。
        """
        pol = self.sizePolicy()
        pol.setHeightForWidth(True)
        self.setSizePolicy(pol)
        w = 720
        self.resize(w, max(self.heightForWidth(w), self.sizeHint().height()))

    def show_segment(self, seg: Segment, display_name: str):
        self._speaker.setText(display_name)
        self._zh.setText(seg.translation or seg.text)
        self._orig.setText(f"({seg.lang}) {seg.text}" if seg.translation else "")
        self._relayout()
        self.show()
        self.raise_()

    def change_font(self, delta: int):
        self._font_size = max(12, min(40, self._font_size + delta))
        self._apply_font()
        self._relayout()

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
        d = e.angleDelta().y() or e.pixelDelta().y()
        if not d:
            e.ignore()
            return
        self.change_font(1 if d > 0 else -1)
        e.accept()
