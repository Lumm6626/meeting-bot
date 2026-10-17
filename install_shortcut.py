# -*- coding: utf-8 -*-
"""在桌面创建快捷方式。用法：python install_shortcut.py"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BAT = ROOT / "run_meeting_bot.bat"
LNK = Path.home() / "Desktop" / "会议翻译助手.lnk"

ICO = ROOT / "assets" / "app.ico"

ps = (
    "$ws = New-Object -ComObject WScript.Shell;"
    f"$s = $ws.CreateShortcut('{LNK}');"
    f"$s.TargetPath = '{BAT}';"
    f"$s.WorkingDirectory = '{ROOT}';"
    # 桌面图标用应用自己的 ico（否则显示 bat/解释器的白纸图标）
    f"$s.IconLocation = '\"{ICO}\",0';"
    "$s.Save()"
)
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
print(f"已创建桌面快捷方式：{LNK}")
