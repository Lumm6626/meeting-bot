# -*- coding: utf-8 -*-
"""字幕浮窗（Teams 转录式滚动面板）离屏回归测试。

覆盖：
  a. 回合分块历史 + MAX_TURNS 上限（最旧块被删）；
  b. 自动跟随底部 / 用户上翻暂停 / 滚回底部恢复；
  c. 字号随窗口宽度缩放 + 倍率 0.6~2.2 钳制；
  d. 滚轮调倍率（走 eventFilter 真实路径）、倍率标签文本与字号同步；
  e. 8px 边缘（右下角）拖拽缩放、中间区域拖动移动窗口、最小尺寸约束；
  f. show_segment 的块内容（时间+名字 / 译文段 / 原文段+语种 / 省略重复原文）；
  g. 回合分块语义：同人连续合并、停顿分块、换人分块、原地更新、按块计数；
  h. 段落拼接：块内各句不逐句换行，译文段补分隔符规则（无标点补 "，"、
     下句以标点开头不补、已有句末标点直接相接），原文段用空格拼成小段；

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

from meeting_bot.subtitle import (MAX_ENTRIES, MAX_TURNS, MIN_H, MIN_W,
                                  TURN_GAP_SECONDS, ZOOM_MAX, ZOOM_MIN,
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
         start: float | None = None, speaker: str = "") -> Segment:
    """造一条语句。默认起点 i*5.0（相邻起点差 5s > TURN_GAP_SECONDS=4s，各自成块）。"""
    start = i * 5.0 if start is None else start
    tr = f"译文{i}" if translation is None else translation
    return Segment(start=start, end=start + 3, text=text, lang=lang,
                   speaker=speaker, translation=tr, seq=i + 1)


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
# a. 回合块追加 + 60 块上限
# ---------------------------------------------------------------------------

def test_appends_blocks_and_keeps_cap(qapp, win):
    """65 个回合（起点间隔 5s > 4s 阈值，互不合并）→ 只留最新 60 块。"""
    for i in range(65):
        win.show_segment(_seg(i), f"说话人{i}")
    _pump(qapp)

    entries = win.entries()
    assert MAX_ENTRIES == 60
    assert len(entries) == MAX_ENTRIES
    # 最旧 5 块（译文0..译文4）被丢弃，最新一块在末尾
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
    """间隔都 >4s → 每句各自成块，块内排版与单条记录一致。"""
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


# ---------------------------------------------------------------------------
# g. 回合分块（同人连续合并 / 停顿分块 / 换人分块 / 原地更新 / 按块计数）
# ---------------------------------------------------------------------------

def test_same_turn_merges_into_one_block(qapp, win):
    """3 条间隔 <4s 且说话人相同/为空的语句 → 1 个块，块内译文/原文各 3 行。"""
    for i, t0 in enumerate((0.0, 1.2, 2.4)):
        win.show_segment(_seg(i, start=t0, text=f"line{i}",
                              translation=f"译文{i}"), "发言人?")
    _pump(qapp)

    blocks = win.entries()
    assert len(blocks) == 1
    b = blocks[0]
    assert b.meta.text() == "[00:00:00]  发言人?"      # 头行取回合首句时间
    # 译文段：块内各句译文连续拼成一整段（无标点 → 补 "，"），不逐句换行
    assert b.main.text() == "译文0，译文1，译文2"
    assert len(b.main.text().splitlines()) == 1
    # 原文段：空格拼接的一小段，语种标签只在段首出现一次
    assert b.sub.text() == "(en) line0 line1 line2"
    assert len(b.sub.text().splitlines()) == 1
    assert b.sub.isVisibleTo(b)
    assert len(b.segments) == 3
    assert [s.start for s in win.current_turn] == [0.0, 1.2, 2.4]


def test_gap_creates_new_block(qapp, win):
    """间隔 5s（> TURN_GAP_SECONDS=4s）→ 两个块。"""
    win.show_segment(_seg(0, start=0.0), "甲")
    win.show_segment(_seg(1, start=5.0), "甲")
    _pump(qapp)

    blocks = win.entries()
    assert len(blocks) == 2
    assert [b.main.text() for b in blocks] == ["译文0", "译文1"]
    assert blocks[0].meta.text() == "[00:00:00]  甲"
    assert blocks[1].meta.text() == "[00:00:05]  甲"
    # 新块开张 → 当前回合只剩新的一句
    assert [s.start for s in win.current_turn] == [5.0]
    # 阈值 2.0 → 4.0：句中稍作停顿（2~4s）不再被切成新块
    assert TURN_GAP_SECONDS == 4.0


def test_speaker_change_creates_new_block(qapp, win):
    """换人（两边声纹标签都非空且不同）→ 即使间隔 <TURN_GAP 也新开块。"""
    win.show_segment(_seg(0, start=0.0, speaker="A"), "甲")
    win.show_segment(_seg(1, start=1.0, speaker="B"), "乙")
    _pump(qapp)

    blocks = win.entries()
    assert len(blocks) == 2
    assert [b.speaker for b in blocks] == ["A", "B"]
    assert blocks[0].main.text() == "译文0"
    assert blocks[1].main.text() == "译文1"

    # 标签从空变为非空（延迟出现）不切块，而是归入并记住
    win2 = SubtitleWindow()
    win2.show()
    win2.show_segment(_seg(0, start=0.0, speaker=""), "发言人?")
    win2.show_segment(_seg(1, start=1.0, speaker="A"), "发言人?")
    win2.show_segment(_seg(2, start=2.0, speaker="A"), "发言人?")
    _pump(qapp)
    assert len(win2.entries()) == 1
    assert win2.entries()[0].main.text() == "译文0，译文1，译文2"
    win2.hide()
    win2.deleteLater()
    _pump(qapp)


def test_block_updates_in_place(qapp, win):
    """同一回合追加第 2 句：块数不变（仍 1）、仍是同一控件、标签含两句。"""
    win.show_segment(_seg(0, start=0.0, text="aa", translation="甲甲"), "甲")
    _pump(qapp)
    first = win.entries()[0]
    assert len(win.entries()) == 1

    win.show_segment(_seg(1, start=1.0, text="bb", translation="乙乙"), "甲")
    _pump(qapp)

    assert len(win.entries()) == 1
    assert win.entries()[0] is first                   # 原地更新，不是新增块
    assert first.main.text() == "甲甲，乙乙"          # 追加后重新拼成一段
    assert first.sub.text() == "(en) aa bb"
    # 头行仍是回合首句时间
    assert first.meta.text() == "[00:00:00]  甲"


def test_turn_block_falls_back_to_text(qapp, win):
    """回合内某句无译文 → 该句原文顶到主标签，副标签只留真有译文的句子。"""
    win.show_segment(_seg(0, start=0.0, text="hello", translation="你好"), "甲")
    win.show_segment(_seg(1, start=1.0, text="world", translation=""), "甲")
    _pump(qapp)

    b = win.entries()[0]
    # 无译文那句的原文占位进译文段（"你好" 无标点 → 补 "，"）
    assert b.main.text() == "你好，world"
    # 原文段只收真有译文的句子（world 已在上方占位，不重复）
    assert b.sub.text() == "(en) hello"
    assert b.sub.isVisibleTo(b)


def test_original_paragraph_joins_with_space(qapp, win):
    """原文段：多句原文用空格拼成一小段（英文自然），不逐句换行。"""
    win.show_segment(_seg(0, start=0.0, text="hello", lang="", translation="你好"), "甲")
    win.show_segment(_seg(1, start=1.0, text="world", lang="", translation="世界"), "甲")
    _pump(qapp)

    b = win.entries()[0]
    assert b.main.text() == "你好，世界"
    assert b.sub.text() == "hello world"
    assert len(b.sub.text().splitlines()) == 1
    assert b.sub.isVisibleTo(b)


def test_no_translation_placeholder_not_duplicated(qapp, win):
    """无译文句：译文段用原文占位，原文段不含该句（不重复显示）。"""
    win.show_segment(_seg(0, start=0.0, text="hello", translation="你好"), "甲")
    win.show_segment(_seg(1, start=1.0, text="pending part", translation=""), "甲")
    win.show_segment(_seg(2, start=2.0, text="world", translation="世界"), "甲")
    _pump(qapp)

    b = win.entries()[0]
    assert b.main.text() == "你好，pending part，世界"
    assert b.sub.text() == "(en) hello world"          # pending part 不在原文段
    assert "pending part" not in b.sub.text()

    # 译文到位 → 原地重渲染：占位换译文，原文同时进入原文段
    assert win.update_segment(
        _seg(1, start=1.0, text="pending part", translation="待译"), "甲") is True
    _pump(qapp)
    assert b.main.text() == "你好，待译，世界"
    assert b.sub.text() == "(en) hello pending part world"


def test_paragraph_separator_rules(qapp, win):
    """补分隔符只在必要时出现：无标点补 "，"，下一句以标点开头不补，已有标点直接相接。"""
    # 3 句译文 "你好" / "世界" / "！" → "你好，世界！"（"！" 前不补 "，"）
    for i, (txt, tr) in enumerate([("a", "你好"), ("b", "世界"), ("c", "！")]):
        win.show_segment(_seg(i, start=float(i), text=txt, translation=tr), "甲")
    _pump(qapp)
    assert win.entries()[0].main.text() == "你好，世界！"

    # 上一句已带句末标点 → 直接相接（靠时间间隔拉开新块）
    win.show_segment(_seg(10, start=100.0, text="x", translation="今天下雨。"), "乙")
    win.show_segment(_seg(11, start=101.0, text="y", translation="我们改期"), "乙")
    _pump(qapp)
    assert win.entries()[-1].main.text() == "今天下雨。我们改期"


def test_join_paragraph_helper():
    """join_paragraph：空片段跳过、空输入返回空、标点处不重复补分隔符。"""
    from meeting_bot.subtitle import join_paragraph

    assert join_paragraph([], "，") == ""
    assert join_paragraph(["你好"], "，") == "你好"
    assert join_paragraph([" 你好 ", "", "   "], "，") == "你好"
    assert join_paragraph(["你好", "世界"], "，") == "你好，世界"
    assert join_paragraph(["你好。", "世界"], "，") == "你好。世界"
    assert join_paragraph(["你好", "！"], "，") == "你好！"
    # 英文原文段：punct_gap=" " → 半角标点后补空格，保持词边界
    assert join_paragraph(["hello", "world"], " ", punct_gap=" ") == "hello world"
    assert join_paragraph(["hello,", "world"], " ", punct_gap=" ") == "hello, world"
    # 全角标点后不补空格（中文原文段不会被塞进空格）
    assert join_paragraph(["今天下雨。", "我们改期"], " ", punct_gap=" ") == "今天下雨。我们改期"


def test_turn_cap_counts_blocks(qapp, win):
    """65 个回合（间隔 5s > 4s）→ 历史只保留 60 个块。"""
    for i in range(65):
        win.show_segment(_seg(i), f"说话人{i}")
    _pump(qapp)

    blocks = win.entries()
    assert MAX_TURNS == 60
    assert len(blocks) == 60
    assert blocks[0].main.text() == "译文5"
    assert blocks[-1].main.text() == "译文64"
    # 每块只有一句（间隔都 > 4s）
    assert all(len(b.segments) == 1 for b in blocks)


# ---------------------------------------------------------------------------
# h. 两阶段显示：原文即时上屏 + 译文原地更新（update_segment / seq 幂等）
# ---------------------------------------------------------------------------

def test_update_segment_replaces_in_place(qapp, win):
    """译文到达 → 原地替换同一句：块数不变、块对象不变、主标签从原文变译文。

    两阶段显示的判据：识别完先 show_segment（translation 为空 → 主标签是原文），
    翻译完成后 update_segment 只重渲染那一块，不新增块、不搬动历史顺序。
    """
    seg = _seg(0, start=0.0, text="Good morning", translation="")
    win.show_segment(seg, "张三")
    _pump(qapp)

    block = win.entries()[-1]
    assert len(win.entries()) == 1
    assert block.main.text() == "Good morning"      # 第一阶段：原文顶到主标签
    assert block.sub.text() == ""                    # 没有译文 → 副标签为空

    # 第二阶段：同一个 seq 的语句对象已带上译文（真实流程里是同一个 Segment 被填充）
    done = _seg(0, start=0.0, text="Good morning", translation="早上好")
    assert win.update_segment(done, "张三") is True
    _pump(qapp)

    assert len(win.entries()) == 1                   # 不新增块
    assert win.entries()[-1] is block                # 仍是同一个控件（原地更新）
    assert block.main.text() == "早上好"              # 主标签变译文
    assert block.sub.text() == "(en) Good morning"    # 原文降为副标签
    assert block.sub.isVisibleTo(block)
    assert [s.translation for s in block.segments] == ["早上好"]
    # 当前回合视图也拿到新对象（同一批语句，不留旧译文）
    assert [s.translation for s in win.current_turn] == ["早上好"]

    # 未知 seq（不属于任何保留块）→ 返回 False 且不新建块（不会凭空冒出一行）
    ghost = _seg(77, start=999.0, text="ghost", translation="幽灵")
    assert win.update_segment(ghost, "无人") is False
    _pump(qapp)
    assert len(win.entries()) == 1


def test_show_segment_dedupes_by_seq(qapp, win):
    """同 seq 重复 show_segment → 不追加新块、文本不重复（幂等，不是 append）。

    真实场景：原文事件与译文更新可能乱序/重发（跨线程队列投递），面板必须按 seq
    去重，否则同一句话会在块里冒出两行、或者额外开一个块。
    """
    first = _seg(0, start=0.0, text="line", translation="")
    win.show_segment(first, "甲")
    _pump(qapp)
    assert len(win.entries()) == 1
    assert win.entries()[0].main.text() == "line"

    # 同 seq 再发一次（带译文）→ 更新同一块，不追加
    win.show_segment(_seg(0, start=0.0, text="line", translation="译文"), "甲")
    _pump(qapp)
    assert len(win.entries()) == 1
    block = win.entries()[0]
    assert block.main.text() == "译文"
    assert block.main.text().count("\n") == 0        # 没有重复行
    assert len(block.segments) == 1

    # 完全相同的重复投递（原文、无变化）也不得追加
    win.show_segment(_seg(0, start=0.0, text="line", translation="译文"), "甲")
    _pump(qapp)
    assert len(win.entries()) == 1
    assert [s.seq for s in win.entries()[0].segments] == [first.seq]

    # 新 seq 照常合并进同一回合（去重只针对同一 seq）
    win.show_segment(_seg(1, start=1.0, text="next", translation="下一句"), "甲")
    _pump(qapp)
    assert len(win.entries()) == 1
    assert win.entries()[0].main.text() == "译文，下一句"


def test_update_segment_after_block_evicted_is_noop(qapp, win):
    """块被 MAX_TURNS 淘汰后，迟到译文不得复活旧块 / 抛异常（seq 映射同步摘除）。"""
    for i in range(MAX_TURNS + 3):
        win.show_segment(_seg(i), f"说话人{i}")
    _pump(qapp)
    assert len(win.entries()) == MAX_TURNS

    stale = _seg(0, translation="迟到的译文")          # 最老的块已被淘汰
    assert win.update_segment(stale, "说话人0") is False
    _pump(qapp)
    assert len(win.entries()) == MAX_TURNS             # 没有多出块
