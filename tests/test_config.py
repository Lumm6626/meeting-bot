# -*- coding: utf-8 -*-
import pytest

from meeting_bot.config import PROJECT_ROOT, ConfigError, load_settings


def test_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_bad_key_format_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=not-a-key\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_valid_env_applies_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
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
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    s = load_settings(env)
    assert s.whisper_model == "small"


def test_hf_token_placeholder_cleaned(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OPENCODE_API_KEY=sk-test123\nHF_TOKEN=hf_在此填入\n",
        encoding="utf-8",
    )
    s = load_settings(env)
    assert s.hf_token == ""


def test_hf_token_ascii_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OPENCODE_API_KEY=sk-test123\nHF_TOKEN=hf_AbCdEf123\n",
        encoding="utf-8",
    )
    s = load_settings(env)
    assert s.hf_token == "hf_AbCdEf123"
