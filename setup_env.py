# -*- coding: utf-8 -*-
"""首次配置向导：交互式填写 .env（翻译服务 / Groq 识别 / HuggingFace 说话人）。

用法：.venv\\Scripts\\python setup_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from meeting_bot.config import save_secret

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"

# 写入 .env 的注释提示（凭据管理器模式下敏感项留空 + 注释说明）
KEYRING_HINT = "已存入 Windows 凭据管理器"

PROVIDERS = {
    "1": {
        "name": "OpenCode Go（如果你有 pi / opencode 账号）",
        "base": "https://opencode.ai/zen/go/v1",
        "translate_model": "deepseek-v4-pro",
        "minutes_model": "deepseek-v4-pro",
    },
    "2": {
        "name": "DeepSeek 官方（platform.deepseek.com，便宜稳定）",
        "base": "https://api.deepseek.com/v1",
        "translate_model": "deepseek-chat",
        "minutes_model": "deepseek-chat",
    },
    "3": {
        "name": "自定义（任意 OpenAI 兼容接口，如通义/月之暗面/本地 Ollama）",
        "base": "",
        "translate_model": "deepseek-chat",
        "minutes_model": "deepseek-chat",
    },
}


def ask(prompt: str, default: str = "") -> str:
    hint = f"（默认 {default}）" if default else ""
    value = input(f"{prompt}{hint}: ").strip()
    return value or default


def main() -> int:
    print("=" * 60)
    print("会议翻译助手 · 首次配置")
    print("=" * 60)

    print("\n① 选择翻译服务（翻译只发送文字，不发送音频）：")
    for key, item in PROVIDERS.items():
        print(f"  {key}) {item['name']}")
    choice = ask("选择", "2")
    provider = PROVIDERS.get(choice, PROVIDERS["2"])

    base = provider["base"] or ask("接口地址 base_url（OpenAI 兼容，如 https://api.xxx.com/v1）")
    api_key = ask("翻译服务 API Key")
    if not api_key:
        print("\n❌ 没填 API Key，无法保存配置。请重新运行本向导。")
        return 1
    translate_model = ask("翻译模型名", provider["translate_model"])
    minutes_model = ask("纪要模型名", provider["minutes_model"] or translate_model)

    print("\n② 语音识别方式：")
    print("  groq = 云端识别（推荐，延迟约 3~5 秒，需免费注册 console.groq.com 拿 key）")
    print("  local = 本地识别（零成本、不出本机，但延迟 30~55 秒）")
    asr = ask("识别方式 groq/local", "groq").lower()
    groq_key = ""
    if asr == "groq":
        groq_key = ask("Groq API Key（gsk_ 开头，直接回车则改用本地识别）")
        if not groq_key:
            asr = "local"
            print("  已切换为本地识别模式")

    print("\n③ 说话人区分（可选，需要 HuggingFace 账号）：")
    print("  需要：1) 注册 huggingface.co 创建 token；2) 依次在三个模型页点 Agree：")
    print("     https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("     https://huggingface.co/pyannote/segmentation-3.0")
    print("     https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
    hf_token = ask("HuggingFace token（hf_ 开头，直接回车跳过）")

    # ---- 密钥写入 Windows 凭据管理器（三处：翻译 / Groq / HF）----
    # 空值也调用一次：清空旧凭据（删条目），避免「向导里留空、程序仍用旧 key」。
    secrets = (
        ("opencode_api_key", api_key),
        ("groq_api_key", groq_key),
        ("hf_token", hf_token),
    )
    keyring_ok = True
    for name, value in secrets:
        if value and not save_secret(name, value):
            keyring_ok = False
        elif not value:
            save_secret(name, value)      # 失败无所谓：本来就没有要写的值

    lines: list[str] = []
    if keyring_ok:
        lines.append("# 由 setup_env.py 生成；密钥已存入 Windows 凭据管理器（服务名 meeting-bot）")
        lines.append("# 本文件不再保存明文密钥，可安全备份（仍建议勿提交到 git）")
    else:
        lines.append("# 由 setup_env.py 生成；密钥只存在本机，勿提交到 git")
    lines.append(f"OPENCODE_BASE_URL={base}")
    if keyring_ok:
        lines += [f"# OPENCODE_API_KEY：{KEYRING_HINT}", "OPENCODE_API_KEY="]
    else:
        lines.append(f"OPENCODE_API_KEY={api_key}")
    lines += [
        f"TRANSLATE_MODEL={translate_model}",
        f"MINUTES_MODEL={minutes_model}",
        "WHISPER_MODEL=small",
        f"ASR_PROVIDER={asr}",
    ]
    if keyring_ok:
        lines += [f"# GROQ_API_KEY：{KEYRING_HINT}", "GROQ_API_KEY="]
    else:
        lines.append(f"GROQ_API_KEY={groq_key}")
    if keyring_ok:
        lines += [f"# HF_TOKEN：{KEYRING_HINT}", "HF_TOKEN="]
    else:
        lines.append(f"HF_TOKEN={hf_token}")
    lines += [
        "DIARIZE_ENABLED=" + ("true" if hf_token else "false"),
        "AUDIO_SOURCE=both",
        "OUTPUT_DIR=会议记录",
        "# 国内网络下载 HuggingFace 模型走镜像（官方 CDN 常被墙）",
        "HF_ENDPOINT=https://hf-mirror.com",
    ]

    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if keyring_ok:
        print(f"\n✅ 配置已保存：{ENV}")
        print("   密钥已存入 Windows 凭据管理器（凭据管理器 → Windows 凭据 → meeting-bot）。")
    else:
        print(f"\n⚠️ 未能写入凭据管理器，密钥仍保存在 .env（请勿分享该文件）：{ENV}")
    print("   下一步：双击桌面「会议翻译助手」开始使用。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
