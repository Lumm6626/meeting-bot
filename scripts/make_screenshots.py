# -*- coding: utf-8 -*-
"""生成 README 界面截图（用 Qt 自身渲染，不截桌面，避免混入其他窗口/缩放问题）。

用法：.venv\\Scripts\\python scripts/make_screenshots.py
产物：docs/screenshots/main-window.png / subtitle-panel.png
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 离屏渲染：不弹窗、不截桌面，像素完全由 Qt 决定
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
sys.path.insert(0, str(ROOT))


def save_widget(widget, path: Path, backdrop: tuple[int, int, int] | None = None) -> None:
    """widget.grab() → PNG；给半透明窗口垫一层背景色（否则透明区域看起来是黑的）。"""
    pix = widget.grab()
    img = pix.toImage()
    tmp = OUT / "_tmp.png"
    img.save(str(tmp))
    pic = Image.open(tmp).convert("RGBA")
    if backdrop is not None:
        canvas = Image.new("RGBA", pic.size, backdrop + (255,))
        canvas.alpha_composite(pic)
        pic = canvas
    pic.convert("RGB").save(path)
    tmp.unlink(missing_ok=True)
    print(f"已保存 {path.name}  {pic.size}")


def main() -> int:
    from meeting_bot.app import MainWindow, app_icon
    from meeting_bot.config import load_settings
    from meeting_bot.subtitle import SubtitleWindow
    from meeting_bot.transcript import Segment

    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())

    win = MainWindow(load_settings())
    win.resize(560, 620)
    win.show()
    app.processEvents()
    save_widget(win, OUT / "main-window.png")

    sub = SubtitleWindow()
    sub.resize(760, 380)
    base = datetime(2026, 10, 18, 10, 0, 0)
    demo = [
        Segment(0.0, 8.0, "Good morning everyone, thanks for joining the allergy clinic review.",
                lang="en", translation="大家早上好，感谢参加过敏门诊评审。"),
        Segment(9.0, 20.0, "The Saudi market is projected to grow at seven percent annually through 2030, driven by new immunotherapy programs.",
                lang="en", translation="沙特市场预计到2030年将保持每年7%的增长，主要来自新的免疫治疗项目。"),
        Segment(26.0, 34.0, "Nuestro distribuidor en Dubái confirmó que el envío llega el martes.",
                lang="es", translation="我们在迪拜的分销商确认货物将于下周二到达。"),
    ]
    for i, seg in enumerate(demo):
        seg.start = float(60 + i * 45)
        sub.show_segment(seg, "发言人?" if i < 2 else "发言人2")
    app.processEvents()
    save_widget(sub, OUT / "subtitle-panel.png", backdrop=(38, 42, 54))
    return 0


if __name__ == "__main__":
    sys.exit(main())
