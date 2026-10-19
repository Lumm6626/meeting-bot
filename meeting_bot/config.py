# -*- coding: utf-8 -*-
"""配置加载：读取 Windows 凭据管理器 / .env，校验必填项。

密钥优先级：**Windows 凭据管理器（keyring）> 环境变量 / .env**。
keyring 在 Windows 上的后端是 Credential Locker，无需额外服务；导入/后端
不可用（Linux 无 SecretService、被策略禁用等）时一律静默降级为 .env，
绝不因为凭据管理器异常而让程序起不来。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:                      # keyring 缺失（老安装 / 精简环境）时按「无凭据管理器」处理
    import keyring
except Exception:         # pragma: no cover - 依赖异常一律降级，不让配置加载失败
    keyring = None        # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 凭据管理器
KEYRING_SERVICE = "meeting-bot"
# 条目名 ↔ .env 变量名（顺序即 secrets_source 的探测顺序）
SECRET_ITEMS = (
    ("opencode_api_key", "OPENCODE_API_KEY"),
    ("groq_api_key", "GROQ_API_KEY"),
    ("hf_token", "HF_TOKEN"),
)

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
    # 实时说话人标注（录制期间每 60 秒跑一轮）：本机 CPU 慢于实时 1.3~1.8 倍，
    # 标签会迟到数分钟且全程占用 CPU，故 .env.example 里默认 **关闭**
    # （DIARIZE_ENABLED=false）。能力保留：置 true 即恢复实时标注。
    diarize_enabled: bool = True
    # 云端识别（Groq）：模型 / 语言提示（空=自动检测）/ 领域词表提示
    groq_asr_model: str = "whisper-large-v3"
    asr_language: str = ""
    asr_prompt: str = ""
    # 纪要模板（见 MINUTES_TEMPLATES）：决定 minutes() 用哪套 prompt
    minutes_template: str = "standard"
    # 字幕显示模式（见 SUBTITLE_MODES）：bilingual 中英对照 | zh_only 只中文
    subtitle_mode: str = "bilingual"
    # 录制落盘音频（SAVE_AUDIO）：整场 16kHz 单声道 wav 写进 <output_dir>/.recording/，
    # 约 115MB/小时。会后离线标注说话人要用它；关掉则只能拿实时标注（更慢、更迟）。
    save_audio: bool = True
    # 会后离线说话人标注（OFFLINE_DIARIZE）：录制结束、初版产物导出后，再对整场音频
    # 跑一遍 pyannote 并把「谁说的」补进逐字稿/纪要（耗时约 0.5~1.5 倍会议时长）。
    offline_diarize: bool = True


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


def _bool_env(name: str, default: bool = True) -> bool:
    """读布尔环境变量：未配置 → default；false/0/no/off（大小写不限）→ False。"""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("false", "0", "no", "off")


def _keyring_get(name: str) -> str:
    """从 Windows 凭据管理器读一项（读不到 / 后端不可用 → ""，绝不抛异常）。

    ``name`` 用条目名（``opencode_api_key`` 等，见 ``SECRET_ITEMS``），
    服务名固定 ``KEYRING_SERVICE``。
    """
    try:
        mod = keyring
        if mod is None:
            return ""
        return (mod.get_password(KEYRING_SERVICE, name) or "").strip()
    except Exception:      # 无后端 / 被禁用 / 条目损坏：一律当作「没配」
        return ""


def save_secret(name: str, value: str) -> bool:
    """把密钥写进 Windows 凭据管理器；写不进去返回 False（调用方回退 .env）。

    值为空串时删除该条目（清空配置 = 移除凭据，不留空密码占位）。
    """
    value = (value or "").strip()
    try:
        mod = keyring
        if mod is None:
            return False
        if value:
            mod.set_password(KEYRING_SERVICE, name, value)
        else:
            try:
                mod.delete_password(KEYRING_SERVICE, name)
            except Exception:   # 本来就没有该条目：删除失败不影响「空值」语义
                pass
        return True
    except Exception:
        return False


def secrets_source(env_path: Path | None = None) -> str:
    """密钥来源：``"keyring"`` / ``".env"`` / ``"none"``（供 UI 提示用）。

    三项密钥只要有一项来自凭据管理器就算 ``keyring``（迁移完成）；否则只要
    .env / 环境变量里有非空值就算 ``.env``（明文，给用户一条轻提示）。
    """
    env_path = env_path or PROJECT_ROOT / ".env"
    for item, _var in SECRET_ITEMS:
        if _keyring_get(item):
            return "keyring"
    load_dotenv(env_path)
    for _item, var in SECRET_ITEMS:
        if (os.getenv(var) or "").strip():
            return ".env"
    return "none"


def load_settings(env_path: Path | None = None) -> Settings:
    env_path = env_path or PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    # 凭据管理器优先，其次 .env / 环境变量（三类密钥一致）
    key = _keyring_get("opencode_api_key") or (os.getenv("OPENCODE_API_KEY") or "").strip()
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
        hf_token=_clean_hf_token(
            _keyring_get("hf_token") or (os.getenv("HF_TOKEN") or "")),
        output_dir=Path(output_dir),
        root=PROJECT_ROOT,
        groq_api_key=_keyring_get("groq_api_key")
        or (os.getenv("GROQ_API_KEY") or "").strip(),
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
        save_audio=_bool_env("SAVE_AUDIO", True),
        offline_diarize=_bool_env("OFFLINE_DIARIZE", True),
    )
