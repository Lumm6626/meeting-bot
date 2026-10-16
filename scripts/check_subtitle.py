# -*- coding: utf-8 -*-
"""手动验证字幕浮窗：python scripts/check_subtitle.py"""
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from meeting_bot.subtitle import SubtitleWindow
from meeting_bot.transcript import Segment

app = QApplication(sys.argv)
win = SubtitleWindow()
samples = [
    Segment(0, 3, "Good morning everyone, let's get started.", lang="en",
            translation="大家早上好，我们开始吧。", speaker="SPEAKER_00"),
    Segment(3, 6, "Buenos días, gracias por venir.", lang="es",
            translation="早上好，感谢各位到来。", speaker="SPEAKER_01"),
]
i = 0


def tick():
    global i
    seg = samples[i % len(samples)]
    win.show_segment(seg, f"说话人{i % len(samples) + 1}")
    i += 1


timer = QTimer()
timer.timeout.connect(tick)
timer.start(3000)
tick()
sys.exit(app.exec())
