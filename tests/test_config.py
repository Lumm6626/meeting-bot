# -*- coding: utf-8 -*-
import pytest

import meeting_bot.config as config_mod
from meeting_bot.config import (KEYRING_SERVICE, PROJECT_ROOT, ConfigError,
                                load_settings, save_secret, secrets_source)

NL = chr(10)      # 显式换行：拼 .env 文本用（避免转义歧义）


def _clean_env(monkeypatch):
    """清空全部相关环境变量，避免宿主 shell 污染（修复 Task 2 deferred minor）。"""
    for var in ("OPENCODE_API_KEY", "OPENCODE_BASE_URL", "TRANSLATE_MODEL",
                "MINUTES_MODEL", "WHISPER_MODEL", "HF_TOKEN", "OUTPUT_DIR",
                "GROQ_API_KEY", "ASR_PROVIDER",
                "ASR_BATCH_SECONDS", "ASR_BATCH_MAX_WAIT", "MINUTES_TEMPLATE",
                "SUBTITLE_MODE"):
        monkeypatch.delenv(var, raising=False)


def test_missing_key_raises(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_bad_key_format_raises(tmp_path, monkeypatch):
    """含非 ASCII 的 key 判为配置错误（前缀格式不再强校验）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-测试key\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_oc_sk_prefix_key_accepted(tmp_path, monkeypatch):
    """OpenCode Go 新格式 oc_sk_... 必须被接受（旧 sk- 前缀校验会误拒）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=oc_sk_abc123\n", encoding="utf-8")
    assert load_settings(env).opencode_api_key == "oc_sk_abc123"


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


def test_diarize_enabled_flag(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nDIARIZE_ENABLED=false\n", encoding="utf-8")
    assert load_settings(env).diarize_enabled is False
    monkeypatch.delenv("DIARIZE_ENABLED", raising=False)  # load_dotenv 不覆盖已存在变量
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    assert load_settings(env).diarize_enabled is True


# ---------------------------------------------------------------------------
# MINUTES_TEMPLATE（纪要模板：standard / action_first / bilingual）
# ---------------------------------------------------------------------------

def test_minutes_template_defaults_to_standard(tmp_path, monkeypatch):
    """不写 MINUTES_TEMPLATE 时默认标准纪要（老 .env 无需改动）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    assert load_settings(env).minutes_template == "standard"


@pytest.mark.parametrize("value", ["standard", "action_first", "bilingual"])
def test_minutes_template_accepts_legal_values(tmp_path, monkeypatch, value):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(f"OPENCODE_API_KEY=sk-test123\nMINUTES_TEMPLATE={value}\n",
                   encoding="utf-8")
    assert load_settings(env).minutes_template == value


def test_minutes_template_case_insensitive(tmp_path, monkeypatch):
    """大小写不敏感：Action_First 这种手写值不该被判成配置错误。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nMINUTES_TEMPLATE=Action_First\n",
                   encoding="utf-8")
    assert load_settings(env).minutes_template == "action_first"


def test_invalid_minutes_template_raises(tmp_path, monkeypatch):
    """拼错的模板名必须报错：静默回落标准会让用户以为模板没生效。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nMINUTES_TEMPLATE=actionfirst\n",
                   encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        load_settings(env)
    assert "MINUTES_TEMPLATE" in str(ei.value)
    assert "action_first" in str(ei.value)      # 错误信息要带上合法取值

# ---------------------------------------------------------------------------
# SUBTITLE_MODE（字幕显示：bilingual 中英对照 / zh_only 只中文）
# ---------------------------------------------------------------------------

def test_subtitle_mode_defaults_to_bilingual(tmp_path, monkeypatch):
    """不写 SUBTITLE_MODE 时默认中英对照（老 .env 无需改动，行为不变）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\n", encoding="utf-8")
    assert load_settings(env).subtitle_mode == "bilingual"


@pytest.mark.parametrize("value", ["bilingual", "zh_only"])
def test_subtitle_mode_accepts_legal_values(tmp_path, monkeypatch, value):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(f"OPENCODE_API_KEY=sk-test123\nSUBTITLE_MODE={value}\n",
                   encoding="utf-8")
    assert load_settings(env).subtitle_mode == value


def test_subtitle_mode_case_insensitive(tmp_path, monkeypatch):
    """手写 ZH_Only 不该被判成配置错误（与 ASR_PROVIDER / MINUTES_TEMPLATE 一致）。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nSUBTITLE_MODE=ZH_Only\n",
                   encoding="utf-8")
    assert load_settings(env).subtitle_mode == "zh_only"


def test_invalid_subtitle_mode_raises(tmp_path, monkeypatch):
    """拼错必须报错：静默回落会让用户以为切换没生效。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-test123\nSUBTITLE_MODE=zh_only_mode\n",
                   encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        load_settings(env)
    assert "SUBTITLE_MODE" in str(ei.value)
    assert "zh_only" in str(ei.value)          # 错误信息带上合法取值


# ---------------------------------------------------------------------------
# Windows 凭据管理器（keyring）：优先于 .env，异常/不可用一律降级
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_keyring(monkeypatch):
    """整文件默认「本机凭据管理器里没有 meeting-bot 条目」。

    避免测试读/写开发者真实 Windows 凭据库（也保证断言不受本机已迁移的密钥影响）。
    单个用例可再 monkeypatch ``config_mod.keyring`` 覆盖成本用例的假后端。
    """
    class _EmptyBackend:
        def get_password(self, service, name):
            return None

        def set_password(self, service, name, value):
            raise AssertionError("测试禁止真实写 keyring")

        def delete_password(self, service, name):
            raise AssertionError("测试禁止真实写 keyring")

    monkeypatch.setattr(config_mod, "keyring", _EmptyBackend())


class _FakeKeyring:
    """最小 keyring 后端替身：固定字典 + 记录写入。"""

    def __init__(self, store=None, *, boom=False):
        self.store = dict(store or {})
        self.boom = boom
        self.writes: list[tuple[str, str, str]] = []

    def get_password(self, service, name):
        if self.boom:
            raise RuntimeError("后端不可用")
        assert service == KEYRING_SERVICE
        return self.store.get(name)

    def set_password(self, service, name, value):
        if self.boom:
            raise RuntimeError("后端不可用")
        self.writes.append((service, name, value))
        self.store[name] = value

    def delete_password(self, service, name):
        if self.boom:
            raise RuntimeError("后端不可用")
        self.store.pop(name, None)


def test_keyring_secret_wins_over_env(tmp_path, monkeypatch):
    """三类密钥都 keyring 优先：.env 里有旧值也以凭据管理器为准。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(NL.join([
        "OPENCODE_API_KEY=sk-from-env",
        "GROQ_API_KEY=gsk-from-env",
        "HF_TOKEN=hf_fromenv",
        "ASR_PROVIDER=groq",
    ]), encoding="utf-8")
    fake = _FakeKeyring({"opencode_api_key": "sk-from-keyring",
                         "groq_api_key": "gsk-from-keyring",
                         "hf_token": "hf_fromkeyring"})
    monkeypatch.setattr(config_mod, "keyring", fake)

    s = load_settings(env)
    assert s.opencode_api_key == "sk-from-keyring"
    assert s.groq_api_key == "gsk-from-keyring"
    assert s.hf_token == "hf_fromkeyring"


def test_keyring_exception_falls_back_to_env(tmp_path, monkeypatch):
    """凭据管理器抛异常（无后端 / 被禁用）时静默回退 .env，不能拖垮启动。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(NL.join(["OPENCODE_API_KEY=sk-from-env",
                            "GROQ_API_KEY=gsk-from-env"]), encoding="utf-8")
    monkeypatch.setattr(config_mod, "keyring", _FakeKeyring(boom=True))

    s = load_settings(env)
    assert s.opencode_api_key == "sk-from-env"
    assert s.groq_api_key == "gsk-from-env"


def test_keyring_bad_value_still_validated(tmp_path, monkeypatch):
    """凭据管理器里的脏值（非 ASCII）同样被既有校验拦下，不因来源不同放行。"""
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=" + NL, encoding="utf-8")
    monkeypatch.setattr(config_mod, "keyring",
                        _FakeKeyring({"opencode_api_key": "sk-坏key"}))
    with pytest.raises(ConfigError):
        load_settings(env)


def test_save_secret_success_writes_keyring():
    fake = _FakeKeyring()
    config_mod.keyring = fake
    assert save_secret("groq_api_key", "gsk_new") is True
    assert fake.store["groq_api_key"] == "gsk_new"


def test_save_secret_failure_returns_false(monkeypatch):
    monkeypatch.setattr(config_mod, "keyring", _FakeKeyring(boom=True))
    assert save_secret("opencode_api_key", "sk-x") is False


def test_save_secret_no_backend_returns_false(monkeypatch):
    monkeypatch.setattr(config_mod, "keyring", None)
    assert save_secret("opencode_api_key", "sk-x") is False


def test_secrets_source_keyring(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-env" + NL, encoding="utf-8")
    monkeypatch.setattr(config_mod, "keyring",
                        _FakeKeyring({"opencode_api_key": "sk-keyring"}))
    assert secrets_source(env) == "keyring"


def test_secrets_source_env(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("OPENCODE_API_KEY=sk-env" + NL, encoding="utf-8")
    assert secrets_source(env) == ".env"


def test_secrets_source_none(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    assert secrets_source(env) == "none"
