# -*- coding: utf-8 -*-
"""生成应用图标 assets/app.ico（窗口 / 托盘 / 桌面快捷方式共用）。

设计：256×256 圆角方形（半径 56px）+ 蓝→青垂直渐变（#2A7DE1 → #1FB6B8），
上区白色「译」字（微软雅黑粗体，约 96px），下区三条白色圆角字幕条
（宽度递减、居左错落、胶囊圆头），整体寓意「实时翻译字幕」。

实现要点：
* 先按 4 倍超采样（1024）绘制再 LANCZOS 缩小，圆角 / 圆头 / 字形边缘才不毛刺；
* 每个尺寸都从 1024 母版单独缩放（不经 256 二次缩放），小尺寸不糊；
* 排版常量全部写在 256 尺度上（``_s()`` 换算到超采样尺度），改版不用重算倍数；
* 字体缺失（非 Windows / 精简系统）时降级：只画字幕条，不报错。

用法：
    python scripts/make_icon.py                       # 生成 assets/app.ico
    python scripts/make_icon.py --preview out.png     # 另存 256px PNG 预览
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_ICO = ROOT / "assets" / "app.ico"

SIZE = 256                  # 图标逻辑尺寸（ICO 最大帧）
SS = 4                      # 超采样倍数 → 1024×1024 母版

# ---- 排版常量（全部以 256px 尺度书写）----
RADIUS = 56                 # 圆角方形半径
GLYPH = "译"
GLYPH_PX = 92               # 字号；字形高度 ≈ 89px
GLYPH_TOP = 44              # 字形墨迹顶端 y
BAR_H = 17                  # 字幕条高度
BAR_GAP = 11                # 字幕条间距
BAR_TOP = 156               # 第一条字幕条顶端 y
BAR_X = 40                  # 字幕条左端 x
BAR_WIDTHS = (176, 132, 88) # 三条条宽（递减）
BAR_RADIUS = BAR_H // 2     # 两端圆头 → 胶囊状

COLOR_TOP = (0x2A, 0x7D, 0xE1)
COLOR_BOTTOM = (0x1F, 0xB6, 0xB8)
WHITE = (255, 255, 255)
ICO_SIZES = (256, 128, 64, 48, 32, 24, 16)
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 粗体
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑 常规
)


def _s(v: float) -> int:
    """256 尺度 → 超采样尺度。"""
    return int(round(v * SS))


def _gradient(n: int) -> Image.Image:
    """垂直线性渐变：1px 宽渐变条 → 横向拉伸（比逐像素 putdata 快且够准）。"""
    strip = Image.new("RGB", (1, n))
    for y in range(n):
        t = y / max(n - 1, 1)
        strip.putpixel((0, y), tuple(
            round(a + (b - a) * t) for a, b in zip(COLOR_TOP, COLOR_BOTTOM)))
    return strip.resize((n, n), Image.Resampling.NEAREST)


def _load_font(px: int) -> ImageFont.FreeTypeFont | None:
    """按候选列表加载中文字体；全部失败返回 None（调用方降级为只画字幕条）。"""
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return None


def _draw_master() -> Image.Image:
    """绘制 1024×1024 母版（RGBA，圆角方形以外透明）。"""
    n = SIZE * SS
    canvas = Image.new("RGBA", (n, n), (0, 0, 0, 0))

    # 1) 圆角方形蒙版 + 渐变底
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, n - 1, n - 1), radius=_s(RADIUS), fill=255)
    canvas.paste(_gradient(n), (0, 0), mask)

    draw = ImageDraw.Draw(canvas)

    # 2) 「译」字：按墨迹框定位（水平居中 + 顶端对齐 GLYPH_TOP），
    #    不依赖 font metrics（不同字体 ascent/descent 差异会让字形上下漂移）
    font = _load_font(_s(GLYPH_PX))
    if font is not None:
        left, top, right, bottom = draw.textbbox((0, 0), GLYPH, font=font)
        x = (n - (right - left)) / 2 - left
        draw.text((x, _s(GLYPH_TOP) - top), GLYPH, font=font, fill=WHITE)

    # 3) 三条字幕条：胶囊圆头、居左错落、宽度递减
    for i, w in enumerate(BAR_WIDTHS):
        top = _s(BAR_TOP + i * (BAR_H + BAR_GAP))
        draw.rounded_rectangle(
            (BAR_X * SS, top, _s(BAR_X + w), top + _s(BAR_H)),
            radius=_s(BAR_RADIUS), fill=WHITE)

    return canvas


def build_icons() -> dict[int, Image.Image]:
    """母版 → {尺寸: 该尺寸 RGBA 图}。"""
    master = _draw_master()
    return {s: master.resize((s, s), Image.Resampling.LANCZOS) for s in ICO_SIZES}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 assets/app.ico")
    ap.add_argument("--preview", help="额外输出 256px PNG 预览的路径")
    args = ap.parse_args(argv)

    icons = build_icons()
    OUT_ICO.parent.mkdir(parents=True, exist_ok=True)
    icons[256].save(OUT_ICO, format="ICO",
                    sizes=[(s, s) for s in ICO_SIZES],
                    append_images=[icons[s] for s in ICO_SIZES if s != 256])
    if args.preview:
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        icons[256].save(args.preview, format="PNG")

    print(f"已生成 {OUT_ICO}（{OUT_ICO.stat().st_size} bytes，尺寸 "
          f"{', '.join(map(str, ICO_SIZES))}）")
    if _load_font(_s(GLYPH_PX)) is None:
        print("警告：未找到微软雅黑字体，图标已降级为「仅字幕条」", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
