# -*- coding: utf-8 -*-
"""配置加载：读取 .env，校验必填项。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASR_PROVIDERS = ("local", "groq")
# 纪要模板：standard 标准（现有顺序）| action_first 行动项优先 | bilingual 中英双语
MINUTES_TEMPLATES = ("standard", "action_first", "bilingual")
# 字幕显示模式：bilingual 中英对照（默认）| zh_only 只显示中文译文
SUBTITLE_MODES = ("bilingual", "zh_only")


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
    # 以下两项带默认值，必须排在最后：既有代码用位置参数构造 Settings
    groq_api_key: str = ""
    asr_provider: str = "local"      # local（本地 faster-whisper）| groq（云端，低延迟）
    # 音频源：both（默认，扬声器+麦克风混合）| speaker | mic；非法值由 capture 报错
    audio_source: str = "both"
    # 本地模式攒批阈值：None = 用 recorder.py 里的模块默认值（30.0 / 35.0）
    asr_batch_seconds: float | None = None
    asr_batch_max_wait: float | None = None
    # 说话人标注开关：默认 true；本机 CPU 标注慢于实时 1.3~1.8 倍（标签迟到数分钟），
    # 关掉可省下会议期间的 CPU 占用（.env 设 DIARIZE_ENABLED=false）
    diarize_enabled: bool = True
    # 云端识别（Groq）：模型 / 语言提示（空=自动检测）/ 领域词表提示
    groq_asr_model: str = "whisper-large-v3"
    asr_language: str = ""
    asr_prompt: str = ""
    # 纪要模板（见 MINUTES_TEMPLATES）：决定 minutes() 用哪套 prompt
    minutes_template: str = "standard"
    # 字幕显示模式（见 SUBTITLE_MODES）：bilingual 中英对照 | zh_only 只中文
    subtitle_mode: str = "bilingual"


def _clean_hf_token(raw: str) -> str:
    """清洗 HF token：占位符（含中文/非 ASCII）一律视为未配置。

    原因：faster-whisper 会把 os.getenv("HF_TOKEN") 直接塞进 HTTP
    Authorization 头，非 ASCII 值会导致 UnicodeEncodeError 崩溃。
    """
    token = raw.strip()
    if not token:
        return ""
    try:
        token.encode("ascii")
    except UnicodeEncodeError:
        return ""
    return token


def _float_env(name: str) -> float | None:
    """读正数浮点环境变量：未配置 → None（由调用方回落到默认值）。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw} 不是合法数字：{e}") from e
    if value <= 0:
        raise ConfigError(f"{name}={raw} 必须大于 0")
    return value


def load_settings(env_path: Path | None = None) -> Settings:
    env_path = env_path or PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    key = (os.getenv("OPENCODE_API_KEY") or "").strip()
    if not key:
        raise ConfigError("缺少 OPENCODE_API_KEY：请复制 .env.example 为 .env 并填入 API key")
    # 前缀不做强校验：OpenCode Go 已出现过 sk- / oc_sk_ 两种格式，硬编码前缀会把
    # 换 key 变成"配置错误"。这里只拦住明显异常（含空白/非 ASCII）的值。
    if any(c.isspace() for c in key) or not key.isascii():
        raise ConfigError("OPENCODE_API_KEY 格式异常（含空白或非 ASCII 字符）")

    provider = (os.getenv("ASR_PROVIDER") or "local").strip().lower()
    if provider not in ASR_PROVIDERS:
        raise ConfigError(f"ASR_PROVIDER={provider} 不合法：可选值 {'|'.join(ASR_PROVIDERS)}")

    template = (os.getenv("MINUTES_TEMPLATE") or "standard").strip().lower()
    if template not in MINUTES_TEMPLATES:
        raise ConfigError(f"MINUTES_TEMPLATE={template} 不合法："
                          f"可选值 {'|'.join(MINUTES_TEMPLATES)}")

    subtitle_mode = (os.getenv("SUBTITLE_MODE") or "bilingual").strip().lower()
    if subtitle_mode not in SUBTITLE_MODES:
        raise ConfigError(f"SUBTITLE_MODE={subtitle_mode} 不合法："
                          f"可选值 {'|'.join(SUBTITLE_MODES)}")

    output_dir = os.getenv("OUTPUT_DIR") or "会议记录"
    if not Path(output_dir).is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return Settings(
        opencode_base_url=(os.getenv("OPENCODE_BASE_URL") or "https://opencode.ai/zen/go/v1").rstrip("/"),
        opencode_api_key=key,
        translate_model=os.getenv("TRANSLATE_MODEL") or "deepseek-v4-flash",
        minutes_model=os.getenv("MINUTES_MODEL") or "deepseek-v4-pro",
        whisper_model=os.getenv("WHISPER_MODEL") or "small",
        hf_token=_clean_hf_token(os.getenv("HF_TOKEN") or ""),
        output_dir=Path(output_dir),
        root=PROJECT_ROOT,
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
        asr_provider=provider,
        audio_source=(os.getenv("AUDIO_SOURCE") or "both").strip().lower(),
        asr_batch_seconds=_float_env("ASR_BATCH_SECONDS"),
        asr_batch_max_wait=_float_env("ASR_BATCH_MAX_WAIT"),
        diarize_enabled=(os.getenv("DIARIZE_ENABLED") or "true").strip().lower() not in ("false", "0", "no", "off"),
        groq_asr_model=(os.getenv("GROQ_ASR_MODEL") or "whisper-large-v3").strip(),
        asr_language=(os.getenv("ASR_LANGUAGE") or "").strip(),
        asr_prompt=(os.getenv("ASR_PROMPT") or "").strip(),
        minutes_template=template,
        subtitle_mode=subtitle_mode,
    )
