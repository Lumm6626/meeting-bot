# -*- coding: utf-8 -*-
"""手动验证字幕浮窗（Teams 转录式滚动面板）：python scripts/check_subtitle.py

用法：
  python scripts/check_subtitle.py             # 真实窗口，每 2 秒追加一条，共 12 条后循环
  python scripts/check_subtitle.py --smoke     # 离屏冒烟：追加 8 条后自动退出（CI/无显示环境）

手动检查清单：
  1. 面板逐条追加，行 1 蓝色「[时间]  说话人」，行 2 白色中文译文，
     行 3 灰色「(语种) 原文」；无译文的条目只有两行；
  2. 拖拽窗口右下角/四边可自由拉伸（鼠标变双向箭头），最小 320x200；
  3. 窗口中间按住左键可移动面板；
  4. 滚轮（或 +/- 键）调缩放倍率 0.6x~2.2x，右下角显示「1.0x」3 秒后自动隐藏，
     字号随窗口宽度自动变化；
  5. 向上滚动翻看历史时暂停自动跟随，滚回底部自动恢复；最多保留 60 条。
"""
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from meeting_bot.subtitle import SubtitleWindow
from meeting_bot.transcript import Segment

SAMPLES = [
    ("Good morning everyone, let's get started.", "en", "大家早上好，我们开始吧。", "说话人1"),
    ("Buenos días, gracias por venir.", "es", "早上好，感谢各位到来。", "说话人2"),
    ("Let's review the Q3 numbers first.", "en", "我们先看一下第三季度的数字。", "说话人1"),
    # 无译文：正文降级为原文，第三行省略
    ("（这段没有译文，直接显示原文）", "zh", "", "说话人3"),
    ("Then we can move on to the roadmap.", "en", "然后我们再谈路线图。", "说话人2"),
]

app = QApplication(sys.argv)
win = SubtitleWindow()
win.show()

state = {"i": 0, "t": 0.0}


def tick():
    text, lang, translation, name = SAMPLES[state["i"] % len(SAMPLES)]
    start = state["t"]
    seg = Segment(start=start, end=start + 3.0, text=text, lang=lang,
                  translation=translation, seq=state["i"] + 1)
    win.show_segment(seg, name)
    state["i"] += 1
    state["t"] += 3.0


# 先灌几条历史，直观看到滚动面板（也顺带验证自动跟随底部）
for _ in range(6):
    tick()

if "--smoke" in sys.argv:
    QTimer.singleShot(0, app.quit)
else:
    print(__doc__)
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(2000)

sys.exit(app.exec())
