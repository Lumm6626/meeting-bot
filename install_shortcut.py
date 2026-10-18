# -*- coding: utf-8 -*-
"""在桌面创建快捷方式：pythonw 直接启动 GUI（无命令行窗口）+ 应用图标。

用法：.venv\\Scripts\\python install_shortcut.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
ICO = ROOT / "assets" / "app.ico"
LNK = Path.home() / "Desktop" / "会议翻译助手.lnk"


def main() -> int:
    if not PYTHONW.exists():
        print(f"找不到 {PYTHONW}，请先在项目里创建虚拟环境")
        return 1
    if not ICO.exists():
        print(f"找不到图标 {ICO}（可运行 scripts/make_icon.py 重新生成）")
        return 1

    # IconLocation 必须是 "路径,索引" 的裸格式：带引号会被 Windows 忽略、回退默认图标
    ps = (
        "$ws = New-Object -ComObject WScript.Shell;"
        f"$s = $ws.CreateShortcut('{LNK}');"
        f"$s.TargetPath = '{PYTHONW}';"
        "$s.Arguments = '-m meeting_bot';"
        f"$s.WorkingDirectory = '{ROOT}';"
        f"$s.IconLocation = '{ICO},0';"
        "$s.Save()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print(f"已创建桌面快捷方式：{LNK}")
    print(f"  目标：{PYTHONW} -m meeting_bot（pythonw，无命令行窗口）")
    print(f"  图标：{ICO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
