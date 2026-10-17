# -*- coding: utf-8 -*-
"""字幕浮窗：Teams 会议转录式滚动面板。

设计要点（对外接口只有 ``SubtitleWindow()`` 与 ``show_segment(seg, name)``，
app.py 依赖这两条，不可变更）：

* 回合分块历史（Teams 转录风）：同一说话人连续说的话合成一个「回合块」——
  头行时间+说话人、正文为回合内各句译文逐行拼接、副标签为各句原文+语种；
  换人（声纹标签变化）或停顿超过 ``TURN_GAP_SECONDS`` 才开新块；同回合追加
  语句时原地更新该块（不新增条目）；最多保留 ``MAX_TURNS`` 个块，丢弃最旧；
* 自动跟随：停在底部时新记录自动滚到底；用户向上翻看时暂停，滚回底部恢复；
* 自由拉伸：无边框窗口自绘圆角黑底，8px 边缘/四角命中即改光标拖拽缩放；
  非边缘区域按住左键移动窗口（沿用原实现）；
* 字号自适配：基础字号 = 窗口宽度 × ``BASE_RATIO``（12~40 钳制），
  滚轮 / +/- 键调整倍率（0.6~2.2），实际字号 = 基础字号 × 倍率，全部条目同步。
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, QRectF, QTimer, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QFrame, QLabel, QScrollArea, QSizePolicy, QVBoxLayout,
                             QWidget)

from .transcript import Segment, fmt_ts

# 半透明黑底参数（paintEvent 与样式表共用，避免两处硬编码漂移）
BG_RGBA = (0, 0, 0, 160)
BG_RADIUS = 10
BG_STYLE = (
    f"background-color: rgba({BG_RGBA[0]}, {BG_RGBA[1]}, {BG_RGBA[2]}, {BG_RGBA[3]}); "
    f"border-radius: {BG_RADIUS}px;"
)

COLOR_META = "#4FC3F7"      # 时间 + 说话人
COLOR_MAIN = "#FFFFFF"      # 译文（主字号）
COLOR_SUB = "#B0BEC5"       # 原文 + 语种
COLOR_ZOOM = "#90A4AE"      # 角落倍率提示

TURN_GAP_SECONDS = 2.0      # 相邻语句起始时间间隔上限，超过即视为新回合（新块）
MAX_TURNS = 60              # 历史回合块上限（一个回合 = 一个条目块）
MAX_ENTRIES = MAX_TURNS     # 兼容旧命名（历史上限，现按回合块计数）
MIN_W, MIN_H = 320, 200     # 最小窗口尺寸
EDGE = 8                    # 边缘拖拽判定宽度（px）
ZOOM_MIN, ZOOM_MAX = 0.6, 2.2
ZOOM_STEP = 0.1
BASE_RATIO = 0.022          # 基础字号 = 宽度 × 该比例
FONT_MIN, FONT_MAX = 12, 40
ZOOM_BADGE_MS = 3000        # 倍率提示自动隐藏时间

SCROLL_STYLE = """
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical {
    background: rgba(176, 190, 197, 120); border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""

_CURSORS = {
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
}


class SubtitleEntry(QWidget):
    """一个「说话回合」块。

    头行（小号蓝）：回合首句时间 + 说话人；
    主标签（主字号白）：回合内各句译文逐行拼接，译文缺失的句子用原文顶上；
    副标签（小号灰）：回合内各句原文（带 (语种) 前缀）逐行拼接——某句译文与
    原文完全一致（或该句无译文）时，原文已在主标签可见，不再重复进副标签；
    副标签整体为空时整行隐藏。

    同回合追加语句走 :meth:`add_segment`，原地重建标签文本，控件数量不变。
    """

    def __init__(self, seg: Segment, display_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        # 整块不吃鼠标：点击穿透给容器，由窗口统一做移动/缩放
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.display_name = display_name
        self.segments: list[Segment] = []

        self.meta = QLabel("")
        self.main = QLabel("")
        self.sub = QLabel("")
        for w in (self.meta, self.main, self.sub):
            w.setWordWrap(True)
            w.setTextFormat(Qt.TextFormat.PlainText)
            w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.meta.setStyleSheet(f"color: {COLOR_META}; font-weight: bold; background: transparent;")
        self.main.setStyleSheet(f"color: {COLOR_MAIN}; background: transparent;")
        self.sub.setStyleSheet(f"color: {COLOR_SUB}; background: transparent;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.meta)
        lay.addWidget(self.main)
        lay.addWidget(self.sub)
        # 副标签常驻布局、按内容显隐：回合内句子增加后可能需要重新显示
        self.sub.hide()
        self.add_segment(seg)

    # ------------------------------------------------------------------ 内容
    @property
    def first(self) -> Segment:
        """回合首句（头行时间取它）。"""
        return self.segments[0]

    @property
    def speaker(self) -> str:
        """回合内首个非空声纹标签，没有则空串。"""
        for s in self.segments:
            if s.speaker:
                return s.speaker
        return ""

    def add_segment(self, seg: Segment) -> None:
        """把一条语句并入本回合（顺序追加），原地刷新三个标签。"""
        self.segments.append(seg)
        if len(self.segments) == 1:
            self.meta.setText(f"{fmt_ts(seg.start)}  {self.display_name}")
        self._rebuild()

    def _rebuild(self) -> None:
        mains: list[str] = []
        subs: list[str] = []
        for s in self.segments:
            tr = (s.translation or "").strip()
            txt = (s.text or "").strip()
            mains.append(tr or txt)             # 译文缺失 → 原文顶上主标签
            if tr and tr != txt:                # 译文==原文时原文已在主标签，不重复
                lang_part = f"({s.lang}) " if s.lang else ""
                subs.append(f"{lang_part}{s.text}")
        self.main.setText("\n".join(mains))
        self.sub.setText("\n".join(subs))
        self.sub.setVisible(bool(subs))

    def set_sizes(self, main_pt: int) -> None:
        """主字号 ``main_pt``；小号行取 0.72 倍（下限 9pt）。"""
        sub_pt = max(9, round(main_pt * 0.72))
        for w, pt in ((self.meta, sub_pt), (self.main, main_pt), (self.sub, sub_pt)):
            f = w.font()
            f.setPointSize(pt)
            w.setFont(f)


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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(MIN_W, MIN_H)     # 与拖拽缩放的最小尺寸保持一致

        self._entries: list[SubtitleEntry] = []      # 回合块（旧 -> 新）
        self._current_turn: list[Segment] = []       # 当前未关闭回合的语句
        self._current_block: SubtitleEntry | None = None
        self._current_speaker = ""                   # 当前回合的声纹标签（可能为空）
        self._zoom = 1.0
        self._auto_scroll = True          # 停在底部 → True，用户向上翻看 → False
        self._op: str | None = None       # "move" / "resize"
        self._grab_edge: str | None = None
        self._press_global = QPoint()
        self._press_geom = self.geometry()
        self._move_offset = QPoint()

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(SCROLL_STYLE)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scroll.viewport().setAutoFillBackground(False)

        self._container = QWidget()
        self._container.setAutoFillBackground(False)
        self._container.setStyleSheet("background: transparent;")
        self._entries_layout = QVBoxLayout(self._container)
        self._entries_layout.setContentsMargins(0, 0, 0, 0)
        self._entries_layout.setSpacing(10)
        self._entries_layout.addStretch(1)      # 条目少时贴顶显示
        self.scroll.setWidget(self._container)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(0)
        lay.addWidget(self.scroll)

        self.zoom_label = QLabel("1.0x", self)
        self.zoom_label.setStyleSheet(
            f"color: {COLOR_ZOOM}; background: transparent; font-weight: bold;")
        self.zoom_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        f = self.zoom_label.font()
        f.setPointSize(9)
        self.zoom_label.setFont(f)
        self.zoom_label.hide()

        bar = self.scroll_bar()
        bar.valueChanged.connect(self._on_scroll_changed)
        bar.rangeChanged.connect(self._on_range_changed)

        # 子控件（滚动区/视口/容器）自己会吃掉鼠标与滚轮，统一过滤回窗口处理
        for w in (self.scroll, self.scroll.viewport(), self._container, self.zoom_label):
            w.installEventFilter(self)

        # 布局落定后再补一次滚到底（rangeChanged 已覆盖多数情况，这是兜底）
        self._stick_timer = QTimer(self)
        self._stick_timer.setSingleShot(True)
        self._stick_timer.setInterval(0)
        self._stick_timer.timeout.connect(self._stick_if_auto)

        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(ZOOM_BADGE_MS)
        self._zoom_timer.timeout.connect(self.zoom_label.hide)

        self.resize(720, 360)

    # ------------------------------------------------------------------ 绘制
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

    # ------------------------------------------------------------ 对外接口
    def show_segment(self, seg: Segment, display_name: str):
        """把一条语句并入当前回合（必要时新开一个回合块）并滚到底部。"""
        self._merge_into_turn(seg, display_name)
        if not self.isVisible():
            self.show()
        self.raise_()
        self._stick_if_auto()
        self._stick_timer.start()

    # ------------------------------------------------------------ 测试辅助
    def entries(self) -> list[SubtitleEntry]:
        """按时间顺序（旧 → 新）返回当前保留的回合块。"""
        return list(self._entries)

    @property
    def current_turn(self) -> list[Segment]:
        """当前未关闭回合里的语句（副本）。"""
        return list(self._current_turn)

    def scroll_bar(self):
        return self.scroll.verticalScrollBar()

    def current_font_size(self) -> int:
        """当前主字号 = 宽度比例基础字号 × 倍率。"""
        base = int(self.width() * BASE_RATIO)
        base = max(FONT_MIN, min(FONT_MAX, base))
        return max(8, int(round(base * self._zoom)))

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def auto_scroll(self) -> bool:
        return self._auto_scroll

    def set_zoom(self, value: float) -> None:
        value = max(ZOOM_MIN, min(ZOOM_MAX, round(float(value), 2)))
        changed = abs(value - self._zoom) > 1e-9
        self._zoom = value
        if changed:
            self._apply_fonts()
        self._show_zoom_badge()

    # -------------------------------------------------------- 回合归并/分块
    def _merge_into_turn(self, seg: Segment, display_name: str) -> None:
        """同回合 → 原地追加；停顿或换人 → 关闭当前回合、开新块。"""
        if self._current_block is not None and self._same_turn(seg):
            self._current_block.add_segment(seg)
            self._current_turn.append(seg)
            if not self._current_speaker and seg.speaker:
                # 说话人标签延迟出现（pyannote 滑窗打标后才有）→ 归入本回合并记住
                self._current_speaker = seg.speaker
            return
        self._start_turn(seg, display_name)

    def _same_turn(self, seg: Segment) -> bool:
        """新语句是否属于当前回合。"""
        if not self._current_turn:
            return False
        last = self._current_turn[-1]
        # 间隔按「起始时间差」判定：VAD/声纹切出的语句常首尾相接甚至重叠，
        # 用上一句 end 到本句 start 的空隙判定会永远合并，识别不出停顿。
        if abs(seg.start - last.start) > TURN_GAP_SECONDS:
            return False
        # 只有两边都有声纹标签且不同才算换人；标签为空（未配 HF token 的现状）
        # 时只按时间间隔分块。
        if seg.speaker and self._current_speaker and seg.speaker != self._current_speaker:
            return False
        return True

    def _start_turn(self, seg: Segment, display_name: str) -> None:
        self._current_turn = [seg]
        self._current_speaker = seg.speaker
        self._current_block = self._add_block(seg, display_name)

    # -------------------------------------------------------------- 块管理
    def _add_block(self, seg: Segment, display_name: str) -> SubtitleEntry:
        entry = SubtitleEntry(seg, display_name, self._container)
        entry.set_sizes(self.current_font_size())
        for w in (entry, entry.meta, entry.main, entry.sub):
            w.installEventFilter(self)
        self._entries_layout.insertWidget(self._entries_layout.count() - 1, entry)
        self._entries.append(entry)
        # 当前回合块永远是最新的一块，裁剪只会淘汰更早的块
        while len(self._entries) > MAX_TURNS:
            old = self._entries.pop(0)
            # 先 hide 再 deleteLater：removeWidget 后它在下一个事件循环前仍是可见子控件
            # （会闪一下）；不 setParent(None)，避免瞬时冒出一个顶层窗口
            old.hide()
            self._entries_layout.removeWidget(old)
            old.deleteLater()
        return entry

    # -------------------------------------------------------------- 字号/滚动
    def _apply_fonts(self) -> None:
        size = self.current_font_size()
        for entry in self._entries:
            entry.set_sizes(size)

    def _show_zoom_badge(self) -> None:
        self.zoom_label.setText(f"{self._zoom:.1f}x")
        self._position_zoom_label()
        self.zoom_label.show()
        self.zoom_label.raise_()
        self._zoom_timer.start()

    def _position_zoom_label(self) -> None:
        label = getattr(self, "zoom_label", None)
        if label is None:           # 构造期 resizeEvent 可能早于控件创建
            return
        label.adjustSize()
        label.move(
            max(6, self.width() - label.width() - 16),
            max(4, self.height() - label.height() - 12),
        )

    def _on_scroll_changed(self, value: int) -> None:
        # 停留底部 → 自动跟随；用户往上翻 → 暂停
        self._auto_scroll = value >= self.scroll_bar().maximum()

    def _on_range_changed(self, _min: int, maximum: int) -> None:
        if self._auto_scroll:
            self.scroll_bar().setValue(maximum)

    def _stick_if_auto(self) -> None:
        if self._auto_scroll:
            bar = self.scroll_bar()
            bar.setValue(bar.maximum())

    def _zoom_step(self, direction: int) -> None:
        self.set_zoom(self._zoom + direction * ZOOM_STEP)

    # ------------------------------------------------------------ 窗口事件
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "_entries", None) is not None:
            self._apply_fonts()
        self._position_zoom_label()

    def showEvent(self, e):
        super().showEvent(e)
        self._apply_fonts()
        self.setFocus()

    def wheelEvent(self, e):
        if self._handle_wheel(e):
            return
        super().wheelEvent(e)

    def _handle_wheel(self, e) -> bool:
        """滚轮调倍率；Ctrl+滚轮交给滚动区翻历史。返回 True 表示已消费。"""
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        d = e.angleDelta().y() or e.pixelDelta().y()
        if not d:
            return False
        self._zoom_step(1 if d > 0 else -1)
        e.accept()
        return True

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_step(1)
        elif e.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._zoom_step(-1)
        else:
            super().keyPressEvent(e)
            return
        e.accept()

    # ------------------------------------------------------- 边缘缩放 / 拖动
    def _hit_test(self, pos: QPoint) -> str | None:
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        if not (0 <= x < w and 0 <= y < h):
            return None
        left, right = x < EDGE, x >= w - EDGE
        top, bottom = y < EDGE, y >= h - EDGE
        if top and left:
            return "tl"
        if top and right:
            return "tr"
        if bottom and left:
            return "bl"
        if bottom and right:
            return "br"
        if left:
            return "l"
        if right:
            return "r"
        if top:
            return "t"
        if bottom:
            return "b"
        return None

    def _update_cursor(self, pos: QPoint) -> None:
        edge = self._hit_test(pos)
        if edge:
            self.setCursor(_CURSORS[edge])
        else:
            self.unsetCursor()

    def _begin_interaction(self, local: QPoint, global_pos: QPoint) -> None:
        self._press_global = global_pos
        edge = self._hit_test(local)
        if edge:
            self._op = "resize"
            self._grab_edge = edge
            self._press_geom = self.geometry()
        else:
            self._op = "move"
            self._grab_edge = None
            self._move_offset = global_pos - self.frameGeometry().topLeft()

    def _continue_interaction(self, global_pos: QPoint) -> None:
        if self._op is None:
            return
        delta = global_pos - self._press_global
        if self._op == "move":
            self.move(global_pos - self._move_offset)
            return
        g = self._grab_edge or ""
        pg = self._press_geom
        l, t, r, b = pg.x(), pg.y(), pg.x() + pg.width(), pg.y() + pg.height()
        if "l" in g:
            l = min(l + delta.x(), r - MIN_W)
        if "r" in g:
            r = max(r + delta.x(), l + MIN_W)
        if "t" in g:
            t = min(t + delta.y(), b - MIN_H)
        if "b" in g:
            b = max(b + delta.y(), t + MIN_H)
        self.setGeometry(l, t, r - l, b - t)

    def _end_interaction(self) -> None:
        self._op = None
        self._grab_edge = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            g = e.globalPosition().toPoint()
            self._begin_interaction(self.mapFromGlobal(g), g)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # 位置一律用 globalPosition 映射：子控件冒泡上来的事件里 position() 已被
        # 按冒泡链翻译过（实测落在 (24,22) 这种错值），会让光标/命中判定错乱
        g = e.globalPosition().toPoint()
        if self._op is not None:
            self._continue_interaction(g)
            e.accept()
            return
        self._update_cursor(self.mapFromGlobal(g))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._op is not None:
            self._end_interaction()
            self._update_cursor(self.mapFromGlobal(e.globalPosition().toPoint()))
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # ---------------------------------------------------------------- 事件过滤
    def eventFilter(self, obj, ev):
        """子控件上的鼠标/滚轮转发到窗口，保证整窗都能拖动、缩放、调字号。"""
        t = ev.type()
        if t == QEvent.Type.Wheel:
            return True if self._handle_wheel(ev) else False
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove,
                 QEvent.Type.MouseButtonRelease):
            g = ev.globalPosition().toPoint()
            local = self.mapFromGlobal(g)
            if t == QEvent.Type.MouseButtonPress:
                if ev.button() != Qt.MouseButton.LeftButton:
                    return False
                self._begin_interaction(local, g)
                return True
            if t == QEvent.Type.MouseMove:
                if self._op is not None:
                    self._continue_interaction(g)
                    return True
                self._update_cursor(local)
                return False
            if self._op is not None:
                self._end_interaction()
                self._update_cursor(local)
                return True
        return super().eventFilter(obj, ev)
