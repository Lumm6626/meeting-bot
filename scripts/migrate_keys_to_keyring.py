# -*- coding: utf-8 -*-
"""把现有 .env 里的密钥迁移到 Windows 凭据管理器（Windows Credential Locker）。

用法：
    python scripts/migrate_keys_to_keyring.py            # 迁移项目根 .env
    python scripts/migrate_keys_to_keyring.py 其他.env

行为：
1. 读取 .env 中 OPENCODE_API_KEY / GROQ_API_KEY / HF_TOKEN 的非空值；
2. 逐项写入凭据管理器（服务名 ``meeting-bot``）；
3. 全部写成功 → 把这些变量在 .env 里置空，并在其上方加注释说明；
4. 打印摘要。**幂等**：值已为空 / 已迁移时不做任何改动，可反复运行。

密钥优先级的实现在 ``meeting_bot.config.load_settings``：凭据管理器 > .env。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # 支持在没有 pip install -e 的机器上直接跑
    sys.path.insert(0, str(ROOT))

from meeting_bot.config import KEYRING_SERVICE, SECRET_ITEMS, save_secret  # noqa: E402

# 改写 .env 时写在空值上方的注释
MIGRATED_COMMENT = (
    "# 已迁移到 Windows 凭据管理器"
    "（{var} 从凭据管理器读取）"
)


def _read_values(path: Path) -> dict[str, str]:
    """扫一遍 .env，取三项密钥的最后一次非空出现值（重复定义以最后一次为准）。"""
    wanted = {var: "" for _item, var in SECRET_ITEMS}
    if not path.is_file():
        return wanted
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name in wanted:
            wanted[name] = value.strip()
    return wanted


def _rewrite_env(path: Path, migrated: set[str]) -> str:
    """把 ``migrated`` 里的变量置空并加迁移注释；返回新文本（不改磁盘）。"""
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        name = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if name in migrated and not stripped.startswith("#"):
            comment = MIGRATED_COMMENT.format(var=name)
            if comment not in out:                 # 重复定义只留一条注释
                out.append(comment)
            out.append(f"{name}=")
            continue
        out.append(raw)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env_path = Path(argv[0]).expanduser() if argv else ROOT / ".env"
    if not env_path.is_file():
        print(f"[跳过] 找不到 {env_path}（没有可迁移的密钥）")
        return 0

    values = _read_values(env_path)
    pending = {var: val for var, val in values.items() if val}
    if not pending:
        print(f"[跳过] {env_path} 里三项密钥均为空 / 已迁移过（幂等，无改动）")
        return 0

    item_of = {var: item for item, var in SECRET_ITEMS}
    migrated: set[str] = set()
    failed: list[str] = []
    for var, value in pending.items():
        if save_secret(item_of[var], value):
            migrated.add(var)
            print(f"[完成] {var} → 凭据管理器（服务名 {KEYRING_SERVICE}）")
        else:
            failed.append(var)
            print(f"[失败] {var} 未能写入凭据管理器（保持 .env 原值不变）")

    if not migrated:
        print("\n❌ 没有密钥迁移成功：请检查 keyring 后端（Windows 上应为 Credential Locker）。")
        print(f"   {env_path} 未做任何修改。")
        return 1

    env_path.write_text(_rewrite_env(env_path, migrated), encoding="utf-8")
    print(f"\n✅ 已迁移 {len(migrated)} 项，{env_path} 中对应变量已置空。")
    if failed:
        print(f"⚠️ 未迁移：{', '.join(failed)}（仍以明文留在 .env）")
    print("   程序启动时会优先从凭据管理器读取密钥；重复运行本脚本无副作用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
