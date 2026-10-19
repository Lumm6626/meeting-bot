# -*- coding: utf-8 -*-
"""迁移脚本（.env → Windows 凭据管理器）回归：改写规则 + 幂等 + 失败不改文件。

不碰真实凭据库：脚本内的 ``save_secret`` 由测试打桩（成功 / 失败两种）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

NL = chr(10)


def _load_script():
    path = ROOT / "scripts" / "migrate_keys_to_keyring.py"
    spec = importlib.util.spec_from_file_location("migrate_keys_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)       # type: ignore[union-attr]
    return mod


@pytest.fixture
def migrate():
    return _load_script()


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(NL.join([
        "# 注释保留",
        "OPENCODE_BASE_URL=https://example.invalid/v1",
        "OPENCODE_API_KEY=sk-secret",
        "GROQ_API_KEY=gsk-secret",
        "HF_TOKEN=hf_secret",
        "TRANSLATE_MODEL=m1",
        "",
    ]), encoding="utf-8")
    return path


def test_migrates_values_and_blanks_env(migrate, env_file, monkeypatch):
    written: dict[str, str] = {}

    def fake_save(name, value):
        written[name] = value
        return True

    monkeypatch.setattr(migrate, "save_secret", fake_save)
    assert migrate.main([str(env_file)]) == 0

    assert written == {
        "opencode_api_key": "sk-secret",
        "groq_api_key": "gsk-secret",
        "hf_token": "hf_secret",
    }
    text = env_file.read_text(encoding="utf-8")
    assert "sk-secret" not in text and "gsk-secret" not in text and "hf_secret" not in text
    assert "OPENCODE_API_KEY=" + NL in text or text.rstrip().endswith("OPENCODE_API_KEY=")
    assert "已迁移到 Windows 凭据管理器" in text
    # 非敏感配置与注释原样保留
    assert "OPENCODE_BASE_URL=https://example.invalid/v1" in text
    assert "# 注释保留" in text


def test_idempotent_second_run(migrate, env_file, monkeypatch):
    calls: list[str] = []

    def fake_save(name, value):
        calls.append(name)
        return True

    monkeypatch.setattr(migrate, "save_secret", fake_save)
    assert migrate.main([str(env_file)]) == 0
    after_first = env_file.read_text(encoding="utf-8")

    calls.clear()
    assert migrate.main([str(env_file)]) == 0
    assert calls == [], "第二次运行不应再写凭据管理器（幂等）"
    assert env_file.read_text(encoding="utf-8") == after_first


def test_failure_leaves_env_untouched(migrate, env_file, monkeypatch):
    before = env_file.read_text(encoding="utf-8")
    monkeypatch.setattr(migrate, "save_secret", lambda name, value: False)
    assert migrate.main([str(env_file)]) == 1
    assert env_file.read_text(encoding="utf-8") == before


def test_missing_env_is_noop(migrate, tmp_path):
    assert migrate.main([str(tmp_path / "nope.env")]) == 0
