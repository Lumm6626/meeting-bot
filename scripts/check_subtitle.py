# -*- coding: utf-8 -*-
"""手动验证字幕浮窗（Teams 转录式「回合分块」面板）：python scripts/check_subtitle.py

用法：
  python scripts/check_subtitle.py             # 真实窗口，按脚本节奏逐条追加，播完循环
  python scripts/check_subtitle.py --smoke     # 离屏冒烟：灌完样例后自动退出

手动检查清单：
  1. 同一个说话人连续说的话合成**一个块**：块内每句译文一行、每句原文一行，
     追加新句时原块原地变高（不新增条目），头行时间保持首句时间；
  2. 停顿 3 秒或换人 → 另起一个块，块之间留空隙；
  3. 无译文的句子用原文顶上主标签，副标签不再重复该句原文（整块无副标签时
     灰色行隐藏）；
  4. 拖拽窗口右下角/四边可自由拉伸（鼠标变双向箭头），最小 320x200；
     窗口中间按住左键可移动面板；
  5. 滚轮（或 +/- 键）调缩放倍率 0.6x~2.2x，右下角显示「1.0x」3 秒后自动隐藏，
     字号随窗口宽度自动变化；
  6. 向上滚动翻看历史时暂停自动跟随，滚回底部自动恢复；最多保留 60 个回合块。
"""
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from meeting_bot.subtitle import TURN_GAP_SECONDS, SubtitleWindow
from meeting_bot.transcript import Segment

# (原文, 语种, 译文, 说话人, 与上一条的起始时间差)
# 同一说话人 + 间隔 < TURN_GAP_SECONDS → 并入同一个回合块；
# 间隔 >= 2s 或换人 → 新块。
SAMPLES = [
    # ── 说话人1 连续 3 句（间隔 1.2s）→ 合成一个块
    ("Good morning everyone, let's get started.", "en", "大家早上好，我们开始吧。", "说话人1", 0.0),
    ("Let's review the Q3 numbers first.", "en", "我们先看一下第三季度的数字。", "说话人1", 1.2),
    ("Then we can move on to the roadmap.", "en", "然后我们再谈路线图。", "说话人1", 1.2),
    # ── 换人 + 停顿 3s → 新块（说话人2 两句）
    ("Buenos días, gracias por venir.", "es", "早上好，感谢各位到来。", "说话人2", 3.0),
    # 无译文：正文降级为原文，副标签省掉这一句
    ("（这段没有译文，直接显示原文）", "zh", "", "说话人2", 1.0),
    # ── 停顿 4s 后同一人继续 → 又开新块
    ("Great, any questions before we close?", "en", "很好，收尾前还有问题吗？", "说话人2", 4.0),
    ("Let's sync again next Thursday.", "en", "我们下周四再同步。", "说话人2", 1.0),
]

app = QApplication(sys.argv)
win = SubtitleWindow()
win.show()

state = {"i": 0, "t": 0.0}


def tick():
    text, lang, translation, name, gap = SAMPLES[state["i"] % len(SAMPLES)]
    state["t"] += gap
    start = state["t"]
    seg = Segment(start=start, end=start + 2.0, text=text, lang=lang,
                  translation=translation, seq=state["i"] + 1)
    win.show_segment(seg, name)
    state["i"] += 1


# 先灌完一整轮，直观看到「同人合并 / 停顿换人分块」两种效果（也验证自动跟随）
for _ in range(len(SAMPLES)):
    tick()

if "--smoke" in sys.argv:
    QTimer.singleShot(0, app.quit)
else:
    print(__doc__)
    print(f"TURN_GAP_SECONDS = {TURN_GAP_SECONDS}s")
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(2000)       # 每 2 秒追加一句，观察原地更新 / 新块
    QTimer.singleShot(0, lambda: print(f"当前块数：{len(win.entries())}"))

sys.exit(app.exec())
