# -*- coding: utf-8 -*-
import pytest

from meeting_bot.config import PROJECT_ROOT, ConfigError, load_settings


def _clean_env(monkeypatch):
    """清空全部相关环境变量，避免宿主 shell 污染（修复 Task 2 deferred minor）。"""
    for var in ("OPENCODE_API_KEY", "OPENCODE_BASE_URL", "TRANSLATE_MODEL",
                "MINUTES_MODEL", "WHISPER_MODEL", "HF_TOKEN", "OUTPUT_DIR",
                "GROQ_API_KEY", "ASR_PROVIDER",
                "ASR_BATCH_SECONDS", "ASR_BATCH_MAX_WAIT"):
        monkeypatch.delenv(var, raising=False)


def test_missing_key_raises(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_bad_key_format_raises(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=not-a-key\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_valid_env_applies_defaults_and_overrides(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "OPENCODE_API_KEY=sk-test123\nTRANSLATE_MODEL=m1\nOUTPUT_DIR=records\n",
        encoding="utf-8",
    )
    s = load_settings(env)
    assert s.opencode_api_key == "sk-test123"
    assert s.translate_model == "m1"
    assert s.minutes_model == "deepseek-v4-pro"
    assert s.output_dir == PROJECT_ROOT / "records"


def test_whisper_default_is_small(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    s = load_settings(env)
    assert s.whisper_model == "small"


def test_hf_token_placeholder_cleaned(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "OPENCODE_API_KEY=sk-test123\nHF_TOKEN=hf_在此填入\n",
        encoding="utf-8",
    )
    s = load_settings(env)
    assert s.hf_token == ""


def test_hf_token_ascii_kept(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "OPENCODE_API_KEY=sk-test123\nHF_TOKEN=hf_AbCdEf123\n",
        encoding="utf-8",
    )
    s = load_settings(env)
    assert s.hf_token == "hf_AbCdEf123"


# ---------------------------------------------------------------------------
# ASR_PROVIDER / GROQ_API_KEY（云端识别模式）
# ---------------------------------------------------------------------------

def test_asr_provider_defaults_to_local(tmp_path, monkeypatch):
    """不写 ASR_PROVIDER 时默认本地识别（云端模式必须显式开启）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    s = load_settings(env)
    assert s.asr_provider == "local"
    assert s.groq_api_key == ""


def test_asr_provider_groq_and_key(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n"
                   "ASR_PROVIDER=Groq\nGROQ_API_KEY=gsk_abc123\n", encoding="utf-8")
    s = load_settings(env)
    assert s.asr_provider == "groq"          # 大小写不敏感
    assert s.groq_api_key == "gsk_abc123"


def test_invalid_asr_provider_raises(tmp_path, monkeypatch):
    """拼错的 provider 必须报错：静默回落本地会让用户以为开着云端正困惑延迟。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nASR_PROVIDER=clode\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        load_settings(env)
    assert "ASR_PROVIDER" in str(ei.value)


def test_asr_batch_thresholds_from_env(tmp_path, monkeypatch):
    """ASR_BATCH_SECONDS / ASR_BATCH_MAX_WAIT 写进 .env 必须真的生效（不是纯文档）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n"
                   "ASR_BATCH_SECONDS=12.5\nASR_BATCH_MAX_WAIT=20\n", encoding="utf-8")
    s = load_settings(env)
    assert s.asr_batch_seconds == 12.5
    assert s.asr_batch_max_wait == 20.0


def test_asr_batch_thresholds_unset_are_none(tmp_path, monkeypatch):
    """未配置时保持 None → Recorder 回落到模块默认值（30.0/35.0）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    s = load_settings(env)
    assert s.asr_batch_seconds is None and s.asr_batch_max_wait is None


def test_asr_batch_threshold_bad_value_raises(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nASR_BATCH_SECONDS=abc\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        load_settings(env)
    assert "ASR_BATCH_SECONDS" in str(ei.value)
