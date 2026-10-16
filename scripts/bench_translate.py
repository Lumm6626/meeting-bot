# -*- coding: utf-8 -*-
"""翻译延迟基准：python scripts/bench_translate.py

对 .env 里的 TRANSLATE_MODEL 连发 3 条测试翻译，输出平均耗时。
若平均耗时 > 5 秒，建议在 .env 中更换更快模型（如 deepseek-v4-flash）。
"""
import time

from meeting_bot.config import load_settings
from meeting_bot.translator import Translator

SAMPLES = [
    ("Good morning everyone, let's get started with the allergy clinic update.", "en"),
    ("El envío llegará la próxima semana, sin falta.", "es"),
    ("We need to finalize the pricing before Friday.", "en"),
]

t = Translator(load_settings())
total = 0.0
for text, lang in SAMPLES:
    t0 = time.perf_counter()
    out = t.translate(text, lang)
    dt = time.perf_counter() - t0
    total += dt
    print(f"[{dt:5.2f}s] {text}\n        → {out}")
print(f"\n平均耗时：{total / len(SAMPLES):.2f} 秒/句")
