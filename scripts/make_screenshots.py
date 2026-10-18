# -*- coding: utf-8 -*-
"""生成 README 用的界面截图（只截取本应用窗口，不含桌面其他内容）。

用法：.venv\\Scripts\\python scripts/make_screenshots.py
产物：docs/screenshots/main-window.png / subtitle-panel.png
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from PIL import ImageGrab
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


def window_rect(title: str) -> tuple[int, int, int, int] | None:
    """按标题找窗口并返回屏幕坐标矩形（只截应用窗口，绝不截整个桌面）。"""
    u = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(h, _l):
        if u.IsWindowVisible(h):
            n = u.GetWindowTextLengthW(h)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                u.GetWindowTextW(h, buf, n + 1)
                if buf.value == title:
                    found.append(h)
        return True

    u.EnumWindows(cb, 0)
    if not found:
        return None
    h = found[0]
    u.SetForegroundWindow(h)
    time.sleep(0.5)
    rect = wt.RECT()
    u.GetWindowRect(h, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def grab(title: str, out: Path) -> bool:
    r = window_rect(title)
    if not r:
        print(f"未找到窗口：{title}")
        return False
    img = ImageGrab.grab(bbox=r, all_screens=True)
    img.save(out)
    print(f"已保存 {out}  尺寸 {img.size}")
    return True


def main() -> int:
    from meeting_bot.app import MainWindow, app_icon
    from meeting_bot.config import load_settings
    from meeting_bot.subtitle import SubtitleWindow
    from meeting_bot.transcript import Segment

    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    win = MainWindow(load_settings())
    win.resize(560, 600)
    win.show()

    # 演示数据：让字幕面板有内容（回合分块 + 原文/译文段落）
    sub = SubtitleWindow()
    sub.resize(760, 360)
    base = datetime(2026, 10, 18, 10, 0, 0)
    demo = [
        Segment(0.0, 8.0, "Good morning everyone, thanks for joining the allergy clinic review.",
                lang="en", translation="大家早上好，感谢参加过敏门诊评审。"),
        Segment(9.0, 20.0, "The Saudi market is projected to grow at seven percent annually through 2030.",
                lang="en", translation="沙特市场预计到2030年将保持每年7%的增长。"),
        Segment(26.0, 34.0, "Nuestro distribuidor en Dubái confirmó que el envío llega el martes.",
                lang="es", translation="我们在迪拜的分销商确认货物将于下周二到达。"),
    ]
    for i, seg in enumerate(demo):
        seg.start = (base + timedelta(seconds=i * 12)).timestamp() % 3600
        sub.show_segment(seg, "发言人?" if i < 2 else "发言人2")

    def shoot():
        try:
            grab("会议翻译助手", OUT / "main-window.png")
            sub.raise_()
            sub.activateWindow()
            time.sleep(0.6)
            grab("会议翻译助手", OUT / "subtitle-panel.png") if False else None
            # 字幕浮窗标题为空（无边框）：按几何位置直接截
            g = sub.geometry()
            img = ImageGrab.grab(
                bbox=(g.x(), g.y(), g.x() + g.width(), g.y() + g.height()),
                all_screens=True)
            img.save(OUT / "subtitle-panel.png")
            print(f"已保存 {OUT / 'subtitle-panel.png'}  尺寸 {img.size}")
        finally:
            app.quit()

    QTimer.singleShot(2500, shoot)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
