# -*- coding: utf-8 -*-
"""字幕浮窗（Teams 转录式滚动面板）离屏回归测试。

覆盖：
  a. 追加式历史 + MAX_ENTRIES 上限（最旧被删）；
  b. 自动跟随底部 / 用户上翻暂停 / 滚回底部恢复；
  c. 字号随窗口宽度缩放 + 倍率 0.6~2.2 钳制；
  d. 滚轮调倍率（走 eventFilter 真实路径）、倍率标签文本与字号同步；
  e. 8px 边缘（右下角）拖拽缩放、中间区域拖动移动窗口、最小尺寸约束；
  f. show_segment 的文本内容（时间+名字 / 译文 / 原文+语种 / 省略第三行）。

全程 QT_QPA_PLATFORM=offscreen：不弹真实窗口、不建事件循环、不发网络请求。
布局/滚动区的 range 更新是事件循环驱动的，故断言前用 ``_pump`` 手动跑几轮
``processEvents``（真实运行时由主事件循环完成，行为一致）。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 必须在 import PyQt6 之前

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from meeting_bot.subtitle import (MAX_ENTRIES, MIN_H, MIN_W, ZOOM_MAX, ZOOM_MIN,
                                  SubtitleWindow)
from meeting_bot.transcript import Segment


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------

def _pump(qapp, rounds: int = 3) -> None:
    """跑几轮事件循环：滚动区的 range/layout 更新是事件驱动的。"""
    for _ in range(rounds):
        qapp.processEvents()


def _seg(i: int, *, text: str = "原文", lang: str = "en", translation: str | None = None,
         start: float | None = None) -> Segment:
    start = i * 3.0 if start is None else start
    tr = f"译文{i}" if translation is None else translation
    return Segment(start=start, end=start + 3, text=text, lang=lang,
                   translation=tr, seq=i + 1)


def _mouse(mtype, local: QPoint, global_pos: QPoint) -> QMouseEvent:
    return QMouseEvent(mtype, QPointF(local), QPointF(global_pos),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _wheel(widget, delta: int, pos: QPoint | None = None) -> QWheelEvent:
    pos = QPoint(widget.width() // 2, widget.height() // 2) if pos is None else pos
    global_pos = widget.mapToGlobal(pos)
    return QWheelEvent(QPointF(pos), QPointF(global_pos), QPoint(0, 0),
                       QPoint(0, delta), Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier,
                       Qt.ScrollPhase.NoScrollPhase, False)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def win(qapp):
    w = SubtitleWindow()
    w.move(100, 100)
    w.resize(520, 320)      # 基础字号 int(520*0.022)=11 → 钳到 12
    w.show()
    _pump(qapp)
    try:
        yield w
    finally:
        w.hide()
        w.deleteLater()
        _pump(qapp)


# ---------------------------------------------------------------------------
# a. 追加 + 60 条上限
# ---------------------------------------------------------------------------

def test_appends_entries_and_keeps_cap(qapp, win):
    for i in range(65):
        win.show_segment(_seg(i), f"说话人{i}")
    _pump(qapp)

    entries = win.entries()
    assert MAX_ENTRIES == 60
    assert len(entries) == MAX_ENTRIES
    # 最旧 5 条（译文0..译文4）被丢弃，最新一条在末尾
    assert entries[0].main.text() == "译文5"
    assert entries[-1].main.text() == "译文64"
    # 被删的旧条目确实已从容器里摘掉，不会继续占高度
    alive = win._container.findChildren(type(entries[0]))
    assert len(alive) == MAX_ENTRIES


# ---------------------------------------------------------------------------
# b. 自动跟随 / 暂停 / 恢复
# ---------------------------------------------------------------------------

def test_auto_scroll_bottom(qapp, win):
    bar = win.scroll_bar()
    for i in range(20):
        win.show_segment(_seg(i, text="long line " * 6), f"说话人{i}")
    _pump(qapp)
    assert bar.maximum() > 0, "内容应已超出可视区，布局/range 未更新"
    assert bar.value() == bar.maximum()
    assert win.auto_scroll is True

    # 用户向上翻 → 暂停自动跟随
    bar.setValue(0)
    _pump(qapp)
    assert win.auto_scroll is False
    win.show_segment(_seg(99, text="while paused"), "甲")
    _pump(qapp)
    assert bar.value() == 0
    assert win.auto_scroll is False

    # 滚回底部 → 恢复自动跟随
    bar.setValue(bar.maximum())
    _pump(qapp)
    assert win.auto_scroll is True
    win.show_segment(_seg(100, text="after resume"), "乙")
    _pump(qapp)
    assert bar.value() == bar.maximum()


# ---------------------------------------------------------------------------
# c. 字号随宽度缩放 / 倍率钳制
# ---------------------------------------------------------------------------

def test_font_scales_with_width(qapp, win):
    win.show_segment(_seg(0, translation="译文"), "甲")
    win.set_zoom(1.0)
    _pump(qapp)

    win.resize(800, 400)
    _pump(qapp)
    big_pt = win.entries()[-1].main.font().pointSize()
    assert big_pt == win.current_font_size()

    win.resize(500, 400)
    _pump(qapp)
    small_pt = win.entries()[-1].main.font().pointSize()
    assert small_pt == win.current_font_size()
    assert big_pt > small_pt, f"800 宽 {big_pt}pt 应大于 500 宽 {small_pt}pt"

    # 倍率钳制在 0.6~2.2，字号随之变化
    win.resize(800, 400)
    _pump(qapp)
    base_pt = win.current_font_size()
    win.set_zoom(9.0)
    assert win.zoom == ZOOM_MAX
    assert win.current_font_size() > base_pt
    win.set_zoom(0.1)
    assert win.zoom == ZOOM_MIN
    assert win.current_font_size() < base_pt
    win.set_zoom(1.0)
    assert win.current_font_size() == base_pt


# ---------------------------------------------------------------------------
# d. 滚轮调倍率
# ---------------------------------------------------------------------------

def test_wheel_zoom_changes_multiplier(qapp, win):
    win.show_segment(_seg(0, translation="译文"), "甲")
    win.set_zoom(1.0)
    _pump(qapp)
    assert win.zoom_label.isVisible()
    before_pt = win.entries()[-1].main.font().pointSize()

    # 发给滚动区视口：验证 eventFilter 把滚轮转发成倍率调节（真实路径）
    viewport = win.scroll.viewport()
    QApplication.sendEvent(viewport, _wheel(viewport, 120))
    _pump(qapp)
    assert win.zoom > 1.0
    assert win.zoom_label.text() == f"{win.zoom:.1f}x"
    assert win.entries()[-1].main.font().pointSize() > before_pt

    after_up = win.zoom
    QApplication.sendEvent(viewport, _wheel(viewport, -120))
    _pump(qapp)
    assert win.zoom < after_up
    assert win.entries()[-1].main.font().pointSize() == before_pt

    # 倍率标签 3 秒后自动隐藏的定时器已启动
    assert win._zoom_timer.isActive()

    # 上限钳制
    for _ in range(30):
        QApplication.sendEvent(viewport, _wheel(viewport, 120))
    _pump(qapp)
    assert win.zoom == ZOOM_MAX
    assert win.zoom_label.text() == f"{ZOOM_MAX:.1f}x"


# ---------------------------------------------------------------------------
# e. 边缘拉伸 / 窗口拖动
# ---------------------------------------------------------------------------

def test_edge_resize(qapp, win):
    g0 = win.geometry()

    # 右下角 5px（落在 8px 判定区内）按下 → 拖 (+100, +80) → 尺寸变大
    local = QPoint(win.width() - 5, win.height() - 5)
    gp = win.mapToGlobal(local)
    win.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, local, gp))
    win.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, local + QPoint(100, 80),
                              gp + QPoint(100, 80)))
    win.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, local + QPoint(100, 80),
                                 gp + QPoint(100, 80)))
    _pump(qapp)
    assert win.width() == g0.width() + 100
    assert win.height() == g0.height() + 80

    # 继续往左上收 → 受最小尺寸 320x200 约束
    local = QPoint(win.width() - 5, win.height() - 5)
    gp = win.mapToGlobal(local)
    win.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, local, gp))
    win.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, local, gp - QPoint(2000, 2000)))
    win.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, local,
                                 gp - QPoint(2000, 2000)))
    _pump(qapp)
    assert (win.width(), win.height()) == (MIN_W, MIN_H)

    # 中间区域拖动 → 窗口移动（尺寸不变）
    size0 = win.size()
    pos0 = win.pos()
    center = QPoint(win.width() // 2, win.height() // 2)
    gp = win.mapToGlobal(center)
    win.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, center, gp))
    win.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, center + QPoint(60, 40),
                              gp + QPoint(60, 40)))
    win.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, center + QPoint(60, 40),
                                 gp + QPoint(60, 40)))
    _pump(qapp)
    assert win.pos() == pos0 + QPoint(60, 40)
    assert win.size() == size0


def test_edge_cursor_and_child_drag(qapp, win):
    """命中判定改光标；落在滚动区容器上的拖动也要能移动窗口（eventFilter 路径）。"""
    right = win.mapToGlobal(QPoint(win.width() - 3, win.height() // 2))
    QApplication.sendEvent(win._container,
                           _mouse(QEvent.Type.MouseMove, QPoint(10, 10), right))
    assert win.cursor().shape() == Qt.CursorShape.SizeHorCursor

    corner = win.mapToGlobal(QPoint(3, 3))
    QApplication.sendEvent(win._container,
                           _mouse(QEvent.Type.MouseMove, QPoint(10, 10), corner))
    assert win.cursor().shape() == Qt.CursorShape.SizeFDiagCursor

    middle = win.mapToGlobal(QPoint(win.width() // 2, win.height() // 2))
    QApplication.sendEvent(win._container,
                           _mouse(QEvent.Type.MouseMove, QPoint(10, 10), middle))
    assert win.cursor().shape() == Qt.CursorShape.ArrowCursor

    pos0 = win.pos()
    QApplication.sendEvent(win._container,
                           _mouse(QEvent.Type.MouseButtonPress, QPoint(10, 10), middle))
    QApplication.sendEvent(win._container,
                           _mouse(QEvent.Type.MouseMove, QPoint(10, 10),
                                  middle + QPoint(30, 20)))
    QApplication.sendEvent(win._container,
                           _mouse(QEvent.Type.MouseButtonRelease, QPoint(10, 10),
                                  middle + QPoint(30, 20)))
    _pump(qapp)
    assert win.pos() == pos0 + QPoint(30, 20)


# ---------------------------------------------------------------------------
# f. 条目内容
# ---------------------------------------------------------------------------

def test_show_segment_content(qapp, win):
    seg = Segment(start=65.0, end=68.0, text="Good morning everyone", lang="en",
                  speaker="SPEAKER_00", translation="大家早上好", seq=1)
    win.show_segment(seg, "张三")
    _pump(qapp)
    e = win.entries()[-1]
    assert e.meta.text() == "[00:01:05]  张三"
    assert e.main.text() == "大家早上好"
    assert e.sub.text() == "(en) Good morning everyone"
    assert e.sub.isVisibleTo(e)

    # 无译文 → 正文降级为原文，第三行省略
    seg2 = Segment(start=70.0, end=72.0, text="Hello there", lang="en", seq=2)
    win.show_segment(seg2, "李四")
    _pump(qapp)
    e2 = win.entries()[-1]
    assert e2.main.text() == "Hello there"
    assert e2.sub.text() == ""
    assert not e2.sub.isVisibleTo(e2)

    # 译文与原文相同 → 第三行同样省略（重复显示没意义）
    seg3 = Segment(start=75.0, end=78.0, text="same text", lang="en",
                   translation="same text", seq=3)
    win.show_segment(seg3, "王五")
    _pump(qapp)
    e3 = win.entries()[-1]
    assert e3.main.text() == "same text"
    assert not e3.sub.isVisibleTo(e3)
