# -*- coding: utf-8 -*-
"""配置加载：读取 .env，校验必填项。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """配置缺失或不合法。"""


@dataclass(frozen=True)
class Settings:
    opencode_base_url: str
    opencode_api_key: str
    translate_model: str
    minutes_model: str
    whisper_model: str
    hf_token: str
    output_dir: Path
    root: Path


def load_settings(env_path: Path | None = None) -> Settings:
    env_path = env_path or PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    key = (os.getenv("OPENCODE_API_KEY") or "").strip()
    if not key:
        raise ConfigError("缺少 OPENCODE_API_KEY：请复制 .env.example 为 .env 并填入 API key")
    if not key.startswith("sk-"):
        raise ConfigError("OPENCODE_API_KEY 格式异常（应以 sk- 开头）")

    output_dir = os.getenv("OUTPUT_DIR") or "会议记录"
    if not Path(output_dir).is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return Settings(
        opencode_base_url=(os.getenv("OPENCODE_BASE_URL") or "https://opencode.ai/zen/go/v1").rstrip("/"),
        opencode_api_key=key,
        translate_model=os.getenv("TRANSLATE_MODEL") or "deepseek-v4-flash",
        minutes_model=os.getenv("MINUTES_MODEL") or "deepseek-v4-pro",
        whisper_model=os.getenv("WHISPER_MODEL") or "medium",
        hf_token=(os.getenv("HF_TOKEN") or "").strip(),
        output_dir=Path(output_dir),
        root=PROJECT_ROOT,
    )
