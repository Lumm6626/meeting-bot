# 会议翻译助手（Teams Meeting Bot）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 进度 ledger 写入 `D:/D_PI_Agent/meeting-bot/.superpowers/sdd/meeting-bot/progress.md`，每完成一个 Task 追加一条记录。

**Goal:** 构建本地运行的 Teams 会议旁听机器人：实时识别多语种发言→翻译成中文→置顶字幕浮窗；会后生成双语逐字稿（Markdown）+ 中文会议纪要（Word），桌面快捷方式双击启动。

**Architecture:** WASAPI loopback 捕获系统声音 → silero-vad 切句 → faster-whisper（CPU int8）识别 → pyannote 说话人标注 + OpenCode Go API（deepseek-v4-flash）翻译 → TranscriptStore 统一缓冲 → Qt 字幕浮窗实时显示；结束后 deepseek-v4-pro 生成纪要，导出 md + docx。

**Tech Stack:** Python 3.12 · faster-whisper · pyannote.audio · soundcard · silero-vad · openai SDK（连 OpenCode Go）· PyQt6 · python-docx · pytest

**参考 spec:** `docs/superpowers/specs/2026-10-16-meeting-bot-design.md`

---

## 文件结构（全部位于 `D:/D_PI_Agent/meeting-bot/`）

```
requirements.txt
.env.example / .env(不入库) / .gitignore
README.md
run_meeting_bot.bat
install_shortcut.py
meeting_bot/
  __init__.py  __main__.py  config.py  transcript.py  translator.py
  asr.py  diarization.py  capture.py  vad.py  recorder.py
  subtitle.py  app.py  export.py
tests/
  __init__.py  test_config.py  test_transcript.py  test_translator.py
  test_diarization.py  test_vad.py  test_recorder.py  test_export.py
scripts/
  check_asr.py  check_diarize.py  check_capture.py  check_subtitle.py  bench_translate.py
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `meeting_bot/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```
# 语音识别（Windows pip 默认装 CPU 版 torch）
faster-whisper==1.1.1
torch==2.5.1
# 说话人聚类
pyannote.audio==3.3.2
# 音频捕获（WASAPI loopback）
soundcard==0.4.3
# 语音活动检测
silero-vad==5.1.2
onnxruntime==1.20.1
# 翻译（OpenAI 兼容接口，连 OpenCode Go）
openai==1.57.0
# GUI
PyQt6==6.7.1
# 导出
python-docx==1.1.2
python-dotenv==1.0.1
numpy==2.1.3
soundfile==0.12.1
pytest==8.3.4
```

- [ ] **Step 2: 创建 .env.example**

```
# 从 C:\Users\alu\.pi\agent\models.json 里 provider "cc-switch-open-code-go" 复制 apiKey
OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_API_KEY=sk-在此填入你的key
TRANSLATE_MODEL=deepseek-v4-flash
MINUTES_MODEL=deepseek-v4-pro
WHISPER_MODEL=medium
# HuggingFace token：先去 https://huggingface.co/settings/tokens 创建，
# 再打开 https://huggingface.co/pyannote/speaker-diarization-3.1 同意使用协议
HF_TOKEN=hf_在此填入
OUTPUT_DIR=会议记录
```

- [ ] **Step 3: 创建 .gitignore**

```
.env
__pycache__/
*.pyc
.pytest_cache/
会议记录/
models/
```

- [ ] **Step 4: 创建 meeting_bot/__init__.py 和 tests/__init__.py**

`meeting_bot/__init__.py`:
```python
# -*- coding: utf-8 -*-
__version__ = "0.1.0"
```

`tests/__init__.py` 为空文件。

- [ ] **Step 5: 创建虚拟环境并安装依赖**

Run:
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install -r requirements.txt
```
Expected: 无报错（约 1-1.5GB 下载，主要 torch/ctranslate2）。

- [ ] **Step 6: 初始化 git 并提交**

```bash
cd D:/D_PI_Agent/meeting-bot
git init
git add -A && git commit -m "chore: 项目脚手架"
```

---

### Task 2: 配置模块（TDD）

**Files:**
- Create: `meeting_bot/config.py`, `tests/test_config.py`

- [ ] **Step 1: 写失败测试** `tests/test_config.py`

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL（ModuleNotFoundError: meeting_bot.config）

- [ ] **Step 3: 实现** `meeting_bot/config.py`

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add meeting_bot/config.py tests/test_config.py
git commit -m "feat: 配置模块（.env 加载与校验）"
```

---

### Task 3: 记录缓冲模块（TDD）

**Files:**
- Create: `meeting_bot/transcript.py`, `tests/test_transcript.py`

- [ ] **Step 1: 写失败测试** `tests/test_transcript.py`

```python
# -*- coding: utf-8 -*-
from datetime import datetime

from meeting_bot.transcript import Segment, TranscriptStore, fmt_ts


def test_fmt_ts():
    assert fmt_ts(0) == "[00:00:00]"
    assert fmt_ts(65) == "[00:01:05]"
    assert fmt_ts(3661) == "[01:01:01]"


def test_add_and_update():
    store = TranscriptStore()
    seg = store.add(1.0, 3.0, "hello", lang="en")
    assert seg.seq == 1
    store.update(1, speaker="SPEAKER_00", translation="你好")
    assert store.segments()[0].speaker == "SPEAKER_00"
    assert store.segments()[0].translation == "你好"


def test_speaker_names():
    store = TranscriptStore()
    store.register_label("SPEAKER_00")
    assert store.display_name("SPEAKER_00") == "SPEAKER_00"
    store.set_speaker_name("SPEAKER_00", "张三")
    assert store.display_name("SPEAKER_00") == "张三"
    assert store.display_name("") == "发言人?"


def test_to_markdown():
    store = TranscriptStore()
    store.add(5.0, 8.0, "Good morning everyone.", lang="en")
    store.set_speaker_name("SPEAKER_00", "Alice")
    md = store.to_markdown("周会", datetime(2026, 10, 16, 10, 30))
    assert "# 会议逐字稿：周会" in md
    assert "Alice" in md
    assert "[00:00:05]" in md
    assert "Good morning everyone." in md
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_transcript.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/transcript.py`

```python
# -*- coding: utf-8 -*-
"""会议记录缓冲：语句存储、说话人命名、Markdown 渲染。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def fmt_ts(seconds: float) -> str:
    """秒 → [HH:MM:SS]"""
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    lang: str = ""
    speaker: str = ""      # 声纹标签，如 SPEAKER_00
    translation: str = ""
    seq: int = 0           # 全局递增序号


@dataclass
class Speaker:
    label: str
    name: str = ""


class TranscriptStore:
    def __init__(self) -> None:
        self._segments: list[Segment] = []
        self._speakers: dict[str, Speaker] = {}
        self._next_seq = 1

    def add(self, start: float, end: float, text: str, lang: str = "") -> Segment:
        seg = Segment(start=start, end=end, text=text, lang=lang, seq=self._next_seq)
        self._next_seq += 1
        self._segments.append(seg)
        return seg

    def update(self, seq: int, **fields) -> None:
        for seg in self._segments:
            if seg.seq == seq:
                for k, v in fields.items():
                    if hasattr(seg, k):
                        setattr(seg, k, v)
                return

    def register_label(self, label: str) -> None:
        if label and label not in self._speakers:
            self._speakers[label] = Speaker(label=label)

    def set_speaker_name(self, label: str, name: str) -> None:
        self.register_label(label)
        self._speakers[label].name = name.strip()

    def display_name(self, label: str) -> str:
        if not label:
            return "发言人?"
        sp = self._speakers.get(label)
        return sp.name if sp and sp.name else label

    def segments(self) -> list[Segment]:
        return list(self._segments)

    def speakers(self) -> list[Speaker]:
        return list(self._speakers.values())

    def to_markdown(self, title: str, started_at: datetime) -> str:
        lines = [f"# 会议逐字稿：{title}", "",
                 f"> 会议时间：{started_at:%Y-%m-%d %H:%M}", "",
                 "## 发言人", ""]
        for sp in self._speakers.values():
            lines.append(f"- {sp.label}：{sp.name or '未命名'}")
        lines += ["", "## 正文", ""]
        for seg in self._segments:
            speaker = self.display_name(seg.speaker)
            lang = seg.lang or "?"
            lines.append(f"**{fmt_ts(seg.start)} · {speaker}（{lang}）**")
            lines.append(f"原文：{seg.text}")
            lines.append(f"译文：{seg.translation or '（未翻译）'}")
            lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_transcript.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add meeting_bot/transcript.py tests/test_transcript.py
git commit -m "feat: 记录缓冲模块（语句存储/说话人命名/Markdown 渲染）"
```

---

### Task 4: 翻译与纪要模块（TDD，mock 接口）

**Files:**
- Create: `meeting_bot/translator.py`, `tests/test_translator.py`

- [ ] **Step 1: 写失败测试** `tests/test_translator.py`

```python
# -*- coding: utf-8 -*-
import openai
import pytest

from meeting_bot.translator import (ActionItem, MinutesResult, TranslationError,
                                    Translator, parse_minutes)


def _settings():
    from meeting_bot.config import Settings
    return Settings(
        opencode_base_url="https://opencode.ai/zen/go/v1",
        opencode_api_key="sk-test",
        translate_model="deepseek-v4-flash",
        minutes_model="deepseek-v4-pro",
        whisper_model="medium",
        hf_token="",
        output_dir=None,
        root=None,
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


def test_translate_returns_text(monkeypatch):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return FakeResponse("  大家好，欢迎参加今天的会议。  ")

    monkeypatch.setattr("openai.OpenAI.chat.completions.create", fake_create)
    t = Translator(_settings())
    assert t.translate("Hello everyone.", "en") == "大家好，欢迎参加今天的会议。"
    assert calls[0]["max_tokens"] == 512


def test_translate_retries_then_raises(monkeypatch):
    def fake_create(**kwargs):
        raise openai.APIError("boom", request=None, body=None)

    monkeypatch.setattr("openai.OpenAI.chat.completions.create", fake_create)
    t = Translator(_settings())
    with pytest.raises(TranslationError):
        t.translate("hello", "en")


def test_thinking_disable_falls_back_on_400(monkeypatch):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if "extra_body" in kwargs:
            raise openai.APIError("unsupported", request=None, body=None)
        return FakeResponse("ok")

    # APIError 没有 status_code 属性时按 400 处理走降级路径
    monkeypatch.setattr("openai.OpenAI.chat.completions.create", fake_create)
    t = Translator(_settings())
    t._thinking_disabled_ok = None
    assert t.translate("hi", "en") == "ok"
    assert len(calls) == 2
    assert "extra_body" not in calls[1]


def test_parse_minutes_valid_json():
    raw = ('{"title": "周会", "summary": "讨论了产品发布", "key_points": ["定价"],
           "decisions": ["下月发布"], "action_items": [{"task": "写方案", "owner": "张三", "due": "10/20"}],
           "open_questions": ["渠道确认"]}')
    m = parse_minutes(raw)
    assert isinstance(m, MinutesResult)
    assert m.title == "周会"
    assert m.action_items[0] == ActionItem(task="写方案", owner="张三", due="10/20")


def test_parse_minutes_falls_back_on_bad_json():
    m = parse_minutes("这不是 JSON，是纯文本纪要。")
    assert m.title == "会议纪要"
    assert m.summary == "这不是 JSON，是纯文本纪要。"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_translator.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/translator.py`

```python
# -*- coding: utf-8 -*-
"""翻译与纪要生成：调用 OpenCode Go（OpenAI 兼容接口）。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import openai

from .config import Settings


class TranslationError(Exception):
    """翻译接口不可用（重试后仍失败）。"""


@dataclass
class ActionItem:
    task: str
    owner: str = ""
    due: str = ""


@dataclass
class MinutesResult:
    title: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


SYSTEM_TRANSLATE = (
    "你是会议实时翻译引擎。把用户发言翻译成简体中文。"
    "要求：1) 只输出译文本身，不要任何解释或前缀；"
    "2) 保留人名、产品名、数字；3) 口语化的表述译成自然流畅的中文；"
    "4) 如果输入本身就是中文，原样输出。"
)

MINUTES_PROMPT = """你是资深会议记录助理。根据下面的会议逐字稿，生成会议纪要。
严格输出 JSON，不要输出其他内容，格式：
{"title": "会议主题", "summary": "3-5句概要",
 "key_points": ["讨论要点1"],
 "decisions": ["结论1"],
 "action_items": [{"task": "任务", "owner": "责任人", "due": "截止时间（无则留空）"}],
 "open_questions": ["待跟进问题1"]}

逐字稿：
{transcript}"""


class Translator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = openai.OpenAI(
            base_url=settings.opencode_base_url,
            api_key=settings.opencode_api_key,
            timeout=30.0,
            max_retries=0,
        )
        self._thinking_disabled_ok: bool | None = None

    def _chat(self, messages: list[dict], max_tokens: int = 512, model: str | None = None) -> str:
        extra = {"thinking": {"type": "disabled"}} if self._thinking_disabled_ok is not False else None
        for attempt in range(3):
            try:
                kwargs = dict(model=model or self._settings.translate_model,
                              messages=messages, max_tokens=max_tokens, temperature=0.0)
                if extra is not None:
                    kwargs["extra_body"] = extra
                resp = self._client.chat.completions.create(**kwargs)
                if extra is not None:
                    self._thinking_disabled_ok = True
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise TranslationError("接口返回空内容")
                return content
            except TranslationError:
                raise
            except openai.APIError as e:
                if extra is not None:
                    # 接口可能不支持 thinking 参数 → 降级为普通请求重试
                    extra = None
                    self._thinking_disabled_ok = False
                    continue
                if attempt == 2:
                    raise TranslationError(f"翻译接口不可用：{e}") from e
                time.sleep(2 ** attempt)
        raise TranslationError("翻译接口不可用")

    def translate(self, text: str, lang: str = "") -> str:
        src = f"（原文语种：{lang}）" if lang and lang != "zh" else ""
        messages = [
            {"role": "system", "content": SYSTEM_TRANSLATE},
            {"role": "user", "content": f"{src}\n{text}"},
        ]
        return self._chat(messages, max_tokens=512)

    def minutes(self, transcript_md: str) -> MinutesResult:
        messages = [
            {"role": "system", "content": "你是资深会议记录助理，输出严格 JSON。"},
            {"role": "user", "content": MINUTES_PROMPT.format(transcript=transcript_md[-180000:])},
        ]
        raw = self._chat(messages, max_tokens=8000, model=self._settings.minutes_model)
        return parse_minutes(raw)


def parse_minutes(raw: str) -> MinutesResult:
    """解析纪要 JSON；失败降级为纯文本纪要。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        return MinutesResult(
            title=str(data.get("title") or "会议纪要"),
            summary=str(data.get("summary") or ""),
            key_points=[str(x) for x in data.get("key_points") or []],
            decisions=[str(x) for x in data.get("decisions") or []],
            action_items=[
                ActionItem(task=str(x.get("task", "")), owner=str(x.get("owner", "")),
                           due=str(x.get("due", "")))
                for x in data.get("action_items") or []
            ],
            open_questions=[str(x) for x in data.get("open_questions") or []],
        )
    except (json.JSONDecodeError, AttributeError):
        return MinutesResult(title="会议纪要", summary=raw)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_translator.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add meeting_bot/translator.py tests/test_translator.py
git commit -m "feat: 翻译与纪要模块（OpenCode Go 接口 + JSON 纪要解析）"
```

---

### Task 5: 说话人模块（TDD + 手动验证）

**Files:**
- Create: `meeting_bot/diarization.py`, `tests/test_diarization.py`, `scripts/check_diarize.py`

- [ ] **Step 1: 写失败测试** `tests/test_diarization.py`（纯函数 assign_speakers）

```python
# -*- coding: utf-8 -*-
from meeting_bot.diarization import assign_speakers
from meeting_bot.transcript import Segment


def test_assign_by_overlap():
    segs = [Segment(0.0, 4.0, "hello", lang="en"),
            Segment(5.0, 9.0, "world", lang="en")]
    spans = [(0.0, 4.5, "SPEAKER_00"), (5.5, 10.0, "SPEAKER_01")]
    assign_speakers(segs, spans)
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[1].speaker == "SPEAKER_01"


def test_no_overlap_stays_unlabeled():
    segs = [Segment(0.0, 4.0, "hello")]
    spans = [(10.0, 20.0, "SPEAKER_00")]
    assign_speakers(segs, spans)
    assert segs[0].speaker == ""


def test_threshold_blocks_weak_overlap():
    segs = [Segment(0.0, 4.0, "hello")]
    spans = [(3.6, 4.0, "SPEAKER_00")]  # 只覆盖 10% < 30%
    assign_speakers(segs, spans)
    assert segs[0].speaker == ""
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_diarization.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/diarization.py`

```python
# -*- coding: utf-8 -*-
"""说话人区分：pyannote 声纹聚类 + 语句归属。"""
from __future__ import annotations

import numpy as np
import torch
from pyannote.audio import Pipeline

from .config import Settings
from .transcript import Segment


class DiarizationError(Exception):
    """说话人聚类不可用（如未配置 HF token）。"""


class Diarizer:
    def __init__(self, settings: Settings):
        if not settings.hf_token:
            raise DiarizationError("缺少 HF_TOKEN，无法使用说话人区分")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )

    def label(self, audio: np.ndarray, sample_rate: int = 16000) -> list[tuple[float, float, str]]:
        """返回 [(start, end, label), ...]，时间以秒为单位。"""
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
        spans = []
        for turn, _, label in result.itertracks(yield_label=True):
            spans.append((float(turn.start), float(turn.end), str(label)))
        return spans


def assign_speakers(segments: list[Segment], spans: list[tuple[float, float, str]],
                    min_overlap: float = 0.3) -> list[Segment]:
    """按时间重叠给语句分配说话人；重叠不足则保持未标注。"""
    for seg in segments:
        best_label, best_overlap = "", 0.0
        for start, end, label in spans:
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        if best_label and best_overlap >= min_overlap * (seg.end - seg.start):
            seg.speaker = best_label
    return segments
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_diarization.py -v`
Expected: 3 passed

- [ ] **Step 5: 创建手动验证脚本** `scripts/check_diarize.py`

```python
# -*- coding: utf-8 -*-
"""手动验证说话人聚类：python scripts/check_diarize.py <多说话人音频.wav>"""
import sys

import numpy as np
import soundfile as sf

from meeting_bot.config import load_settings
from meeting_bot.diarization import Diarizer

audio, sr = sf.read(sys.argv[1])
if audio.ndim > 1:
    audio = audio.mean(axis=1)
spans = Diarizer(load_settings()).label(audio.astype(np.float32), sr)
for start, end, label in spans:
    print(f"[{start:7.1f}s - {end:7.1f}s] {label}")
```

- [ ] **Step 6: 提交**

```bash
git add meeting_bot/diarization.py tests/test_diarization.py scripts/check_diarize.py
git commit -m "feat: 说话人聚类模块（pyannote + 语句归属算法）"
```

---

### Task 6: 语句切分模块（TDD）

**Files:**
- Create: `meeting_bot/vad.py`, `tests/test_vad.py`

- [ ] **Step 1: 写失败测试** `tests/test_vad.py`（纯函数 utterance_bounds）

```python
# -*- coding: utf-8 -*-
import numpy as np

from meeting_bot.vad import utterance_bounds


def _prob_array(speech_ranges, total):
    probs = np.zeros(total, dtype=np.float32)
    for a, b in speech_ranges:
        probs[a:b] = 1.0
    return probs


def test_single_utterance():
    probs = _prob_array([(0, 50), (70, 120)], 200)
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10)
    assert bounds == [(0, 50), (70, 120)]


def test_tail_unclosed_utterance_included():
    probs = _prob_array([(30, 90)], 100)
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10)
    assert bounds == [(30, 100)]  # 结尾未出现静音，也算完整语句


def test_too_short_speech_ignored():
    probs = _prob_array([(0, 3)], 50)  # 3 帧 < min_speech=5
    assert utterance_bounds(probs, min_speech=5, min_silence=10) == []


def test_max_length_forces_split():
    probs = _prob_array([(0, 600)], 600)  # 600 帧 >= max_len=300 → 强制断句
    bounds = utterance_bounds(probs, min_speech=5, min_silence=10, max_len=300)
    assert bounds == [(0, 300), (300, 600)]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_vad.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/vad.py`

```python
# -*- coding: utf-8 -*-
"""语句切分：silero-vad 检测语音边界，输出完整语句音频。"""
from __future__ import annotations

import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME = 512                      # silero 单帧样本数 @16k
MIN_SPEECH_FRAMES = 10           # ~0.32s 起算一句
MIN_SILENCE_FRAMES = 15          # ~0.48s 静音断句
MAX_UTTERANCE_FRAMES = 600       # 最长 19s 强制断句


def utterance_bounds(probs: np.ndarray, min_speech: int = MIN_SPEECH_FRAMES,
                     min_silence: int = MIN_SILENCE_FRAMES,
                     max_len: int = MAX_UTTERANCE_FRAMES) -> list[tuple[int, int]]:
    """VAD 概率数组 → [(start_frame, end_frame), ...] 语句边界（半开区间）。"""
    bounds: list[tuple[int, int]] = []
    start = None
    silence = 0
    for i, p in enumerate(probs):
        if p >= 0.5:
            if start is None:
                start = i
                silence = 0
            else:
                silence = 0
                if i - start + 1 >= max_len:   # 超长强制断句
                    bounds.append((start, i + 1))
                    start = None
        elif start is not None:
            silence += 1
            if silence >= min_silence:
                end = i - min_silence + 1   # 语句终点 = 静音起点
                if end - start >= min_speech:
                    bounds.append((start, end))
                start = None
    if start is not None and len(probs) - start >= min_speech:
        bounds.append((start, len(probs)))
    return bounds


class UtteranceBuffer:
    """流式喂入音频块，产出 (起始样本偏移, 语句音频) 列表。

    起始样本偏移是相对录制起点的绝对偏移，由内部维护的丢弃计数保证。
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._model = load_silero_vad()
        self._frames: list[np.ndarray] = []   # 未决帧，每帧 FRAME 样本
        self._probs: list[float] = []
        self._dropped = 0                     # 已丢弃的样本数（绝对偏移基准）

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = len(arr) // FRAME * FRAME
        for i in range(0, n, FRAME):
            frame = arr[i:i + FRAME]
            prob = float(self._model(torch.from_numpy(frame), self.sample_rate))
            self._frames.append(frame)
            self._probs.append(prob)
        return self._emit()

    def _emit(self) -> list[tuple[int, np.ndarray]]:
        probs = np.asarray(self._probs, dtype=np.float32)
        bounds = utterance_bounds(probs)
        if not bounds:
            if probs.size and probs[-1] >= 0.5:
                return []   # 正在说话，缓冲不能丢弃
            # 静音中：只保留最近一段静音上下文
            keep = max(0, len(probs) - MIN_SILENCE_FRAMES * 2)
            self._drop(keep)
            return []
        last_start, last_end = bounds[-1]
        closed = last_end < len(probs) or (last_end - last_start >= MAX_UTTERANCE_FRAMES)
        if closed:
            emit_bounds, keep = bounds, len(probs)
        else:
            # 最后一段还没闭合（仍在语音），留到下一轮
            emit_bounds, keep = bounds[:-1], last_start
        out = [(self._dropped + a * FRAME, np.concatenate(self._frames[a:b]))
               for a, b in emit_bounds]
        self._drop(keep)
        return out

    def flush(self) -> list[tuple[int, np.ndarray]]:
        probs = np.asarray(self._probs, dtype=np.float32)
        out = [(self._dropped + a * FRAME, np.concatenate(self._frames[a:b]))
               for a, b in utterance_bounds(probs)]
        self._drop(len(probs))
        return out

    def _drop(self, n: int) -> None:
        del self._frames[:n]
        del self._probs[:n]
        self._dropped += n * FRAME
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_vad.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add meeting_bot/vad.py tests/test_vad.py
git commit -m "feat: 语句切分模块（silero-vad 流式边界检测）"
```

---

### Task 7: 音频捕获模块（手动验证）

**Files:**
- Create: `meeting_bot/capture.py`, `scripts/check_capture.py`

- [ ] **Step 1: 实现** `meeting_bot/capture.py`

```python
# -*- coding: utf-8 -*-
"""系统声音捕获：WASAPI loopback（截取扬声器输出，无需虚拟声卡）。"""
from __future__ import annotations

import queue
import threading

import numpy as np
import soundcard as sc


class CaptureError(Exception):
    """找不到系统扬声器/loopback 设备。"""


CHUNK_SECONDS = 0.5


class AudioCapture:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        try:
            default_speaker = sc.default_speaker()
            self._mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
        except Exception as e:
            raise CaptureError(f"无法打开系统声音捕获（请确认声卡支持 WASAPI）：{e}") from e
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=100)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frames = int(self.sample_rate * CHUNK_SECONDS)
        with self._mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
            while self._running.is_set():
                try:
                    data = rec.record(frames)
                    self._queue.put(np.asarray(data, dtype=np.float32).reshape(-1))
                except queue.Full:
                    pass  # 消费不过来时丢弃，保证实时性
                except Exception:
                    break
        self._queue.put(None)  # 结束哨兵

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """读取一块音频；捕获结束返回 None。"""
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
```

- [ ] **Step 2: 创建手动验证脚本** `scripts/check_capture.py`

```python
# -*- coding: utf-8 -*-
"""手动验证系统声音捕获：运行后请播放任意音频/视频 10 秒。

用法：python scripts/check_capture.py
输出：会议记录/_check/loopback_test.wav
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from meeting_bot.capture import AudioCapture

OUT = Path("会议记录/_check/loopback_test.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

cap = AudioCapture()
cap.start()
chunks = []
for _ in range(20):  # 10 秒
    c = cap.read(timeout=1.0)
    if c is None:
        break
    chunks.append(c)
cap.stop()
if not chunks:
    print("未捕获到任何音频！")
else:
    audio = np.concatenate(chunks)
    sf.write(str(OUT), audio, 16000)
    peak = float(np.abs(audio).max())
    print(f"已保存 {len(audio)/16000:.1f} 秒音频到 {OUT}（峰值 {peak:.3f}）")
    print("峰值 > 0.01 说明捕获正常；否则检查是否在播放声音")
```

- [ ] **Step 3: 运行验证**

Run: `python scripts/check_capture.py`（同时播放一段视频）
Expected: 输出"峰值 > 0.01"，wav 文件生成且能听到内容。

- [ ] **Step 4: 提交**

```bash
git add meeting_bot/capture.py scripts/check_capture.py
git commit -m "feat: 系统声音捕获模块（WASAPI loopback）"
```

---

### Task 8: 语音识别模块（单元 + 手动验证）

**Files:**
- Create: `meeting_bot/asr.py`, `tests/test_asr.py`, `scripts/check_asr.py`

- [ ] **Step 1: 写失败测试** `tests/test_asr.py`（mock 模型，验证段映射）

```python
# -*- coding: utf-8 -*-
import numpy as np

from meeting_bot.asr import ASREngine


def _settings():
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None)


class FakeSeg:
    start, end, text = 0.0, 1.5, " Hello world. "


class FakeTranscribe:
    def __init__(self):
        self.segments = [FakeSeg()]
        self.language = "en"


class FakeModel:
    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, **kwargs):
        assert kwargs["language"] is None, "必须开启自动语种检测"
        return [FakeSeg()], type("Info", (), {"language": "en"})()


def test_transcribe_maps_segments(monkeypatch):
    monkeypatch.setattr("meeting_bot.asr.WhisperModel", FakeModel)
    engine = ASREngine(_settings())
    segs = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert len(segs) == 1
    assert segs[0].text == "Hello world."
    assert segs[0].lang == "en"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_asr.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/asr.py`

```python
# -*- coding: utf-8 -*-
"""本地语音识别：faster-whisper，CPU int8，自动语种检测。"""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from .config import Settings
from .transcript import Segment


class ASREngine:
    def __init__(self, settings: Settings):
        self._model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[Segment]:
        """识别语句，返回不含 speaker/translation 的 Segment 列表。"""
        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            language=None,           # 自动检测语种
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        result = []
        for s in segments:
            text = s.text.strip()
            if text:
                result.append(Segment(start=float(s.start), end=float(s.end),
                                      text=text, lang=info.language or ""))
        return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_asr.py -v`
Expected: 1 passed

- [ ] **Step 5: 创建手动验证脚本** `scripts/check_asr.py`

```python
# -*- coding: utf-8 -*-
"""手动验证识别（首次运行会下载 whisper medium 模型 ~1.5GB）。

用法：python scripts/check_asr.py <音频文件.wav>
"""
import sys

import numpy as np
import soundfile as sf

from meeting_bot.asr import ASREngine
from meeting_bot.config import load_settings

audio, sr = sf.read(sys.argv[1])
if audio.ndim > 1:
    audio = audio.mean(axis=1)
segments = ASREngine(load_settings()).transcribe(audio.astype(np.float32), sr)
for s in segments:
    print(f"[{s.start:6.1f}-{s.end:6.1f}] ({s.lang}) {s.text}")
```

- [ ] **Step 6: 运行验证**

Run: `python scripts/check_asr.py 会议记录/_check/loopback_test.wav`（用 Task 7 捕获的音频）
Expected: 打印识别文本，语种正确（如 en）。

- [ ] **Step 7: 提交**

```bash
git add meeting_bot/asr.py tests/test_asr.py scripts/check_asr.py
git commit -m "feat: 语音识别模块（faster-whisper CPU int8）"
```

---

### Task 9: 录制编排模块（TDD 集成测试，fake 注入）

**Files:**
- Create: `meeting_bot/recorder.py`, `tests/test_recorder.py`

- [ ] **Step 1: 写失败测试** `tests/test_recorder.py`

```python
# -*- coding: utf-8 -*-
import time

import numpy as np
import pytest

from meeting_bot.recorder import Recorder
from meeting_bot.transcript import Segment


def _settings():
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", None, None)


class FakeCapture:
    """喂入 3 块静音后结束。"""
    def __init__(self):
        self.chunks = [np.zeros(8000, dtype=np.float32)] * 3
        self._i = 0
        self.started = False

    def start(self):
        self.started = True

    def read(self, timeout=1.0):
        if self._i < len(self.chunks):
            c = self.chunks[self._i]
            self._i += 1
            return c
        return None

    def stop(self):
        pass


class FakeASR:
    def transcribe(self, audio, sample_rate=16000):
        return [Segment(0.5, 2.5, "hello world", lang="en")]


class FakeTranslator:
    def translate(self, text, lang=""):
        return "你好世界"

    def minutes(self, md):
        raise NotImplementedError


class FakeDiarizer:
    def label(self, audio, sample_rate=16000):
        return [(0.0, 10.0, "SPEAKER_00")]


class FakeVAD:
    """替换真实 silero：第一次 push 返回一条语句。"""
    def __init__(self):
        self.pushed = 0

    def push(self, chunk):
        self.pushed += 1
        if self.pushed == 1:
            return [(0, np.zeros(16000, dtype=np.float32))]
        return []

    def flush(self):
        return []


def test_full_pipeline(monkeypatch):
    monkeypatch.setattr("meeting_bot.recorder.UtteranceBuffer", FakeVAD)
    monkeypatch.setattr("meeting_bot.recorder.DIARIZE_INTERVAL", 0.1)
    r = Recorder(_settings(), capture=FakeCapture(), asr=FakeASR(),
                 translator=FakeTranslator(), diarizer=FakeDiarizer())
    events = []
    r.on_segment(events.append)
    r.on_status(lambda m: None)
    r.start("测试会议")
    time.sleep(0.6)  # 等 diarize worker 跑一轮
    r.stop()

    segs = r.store.segments()
    assert len(segs) >= 1
    seg = segs[0]
    assert seg.text == "hello world"
    assert seg.lang == "en"
    assert seg.translation == "你好世界"
    assert seg.speaker == "SPEAKER_00"
    assert len(events) >= 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_recorder.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/recorder.py`

```python
# -*- coding: utf-8 -*-
"""录制编排：串联捕获→切句→识别→标注→翻译→存储。"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from queue import Queue
from typing import Callable

import numpy as np

from .asr import ASREngine
from .capture import AudioCapture
from .config import Settings
from .diarization import Diarizer, assign_speakers
from .transcript import Segment, TranscriptStore
from .translator import TranslationError, Translator
from .vad import UtteranceBuffer

SegmentEvent = Callable[[Segment], None]
StatusEvent = Callable[[str], None]

DIARIZE_INTERVAL = 15.0      # 每 15 秒跑一次说话人聚类


class Recorder:
    def __init__(self, settings: Settings,
                 capture: AudioCapture | None = None,
                 asr: ASREngine | None = None,
                 translator: Translator | None = None,
                 diarizer: Diarizer | None = None):
        self._settings = settings
        self._capture = capture or AudioCapture()
        self._asr = asr
        self._translator = translator
        self._diarizer = diarizer
        self._store = TranscriptStore()
        self._events: list[SegmentEvent] = []
        self._status: list[StatusEvent] = []
        self._stop = threading.Event()
        self._translate_queue: Queue[Segment] = Queue()
        self._diarize_audio: list[np.ndarray] = []
        self._started_at: datetime | None = None
        self.title = ""

    # --- 事件 ---
    def on_segment(self, fn: SegmentEvent) -> None:
        self._events.append(fn)

    def on_status(self, fn: StatusEvent) -> None:
        self._status.append(fn)

    def _emit(self, seg: Segment) -> None:
        for fn in self._events:
            fn(seg)

    def _say(self, msg: str) -> None:
        for fn in self._status:
            fn(msg)

    @property
    def store(self) -> TranscriptStore:
        return self._store

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def translator(self) -> Translator | None:
        return self._translator

    def start(self, title: str = "") -> None:
        self.title = title
        self._started_at = datetime.now()
        self._say("开始录制（等待语音…）")
        self._capture.start()
        threading.Thread(target=self._translate_worker, daemon=True).start()
        threading.Thread(target=self._diarize_worker, daemon=True).start()
        self._main_loop()

    def _main_loop(self) -> None:
        buf = UtteranceBuffer()
        while not self._stop.is_set():
            chunk = self._capture.read(timeout=0.5)
            if chunk is None:
                break
            self._diarize_audio.append(chunk)
            for offset, utterance in buf.push(chunk):
                self._process_utterance(utterance, offset / 16000)
        for offset, utterance in buf.flush():
            self._process_utterance(utterance, offset / 16000)

    def _process_utterance(self, audio: np.ndarray, start_offset: float) -> None:
        try:
            segments = self._asr.transcribe(audio) if self._asr else []
        except Exception as e:
            self._say(f"识别失败：{e}")
            return
        for s in segments:
            seg = self._store.add(start_offset + s.start, start_offset + s.end,
                                  s.text, lang=s.lang)
            if self._translator:
                self._translate_queue.put(seg)
            else:
                self._emit(seg)

    def _translate_worker(self) -> None:
        while not self._stop.is_set() or not self._translate_queue.empty():
            try:
                seg = self._translate_queue.get(timeout=0.5)
            except Exception:
                continue
            try:
                if seg.lang == "zh":
                    seg.translation = seg.text
                else:
                    seg.translation = self._translator.translate(seg.text, seg.lang)
            except TranslationError:
                seg.translation = ""   # 会议结束后统一补译
                self._say(f"[警告] 翻译失败（{seg.text[:20]}…），已保留原文")
            self._emit(seg)

    def _diarize_worker(self) -> None:
        if not self._diarizer:
            return
        last_n = 0
        while not self._stop.is_set():
            time.sleep(DIARIZE_INTERVAL)
            if len(self._diarize_audio) == last_n:
                continue
            last_n = len(self._diarize_audio)
            audio = np.concatenate(self._diarize_audio)
            try:
                spans = self._diarizer.label(audio)
            except Exception as e:
                self._say(f"[警告] 说话人标注失败：{e}")
                continue
            assign_speakers(self._store.segments(), spans)
            for seg in self._store.segments():
                if seg.speaker:
                    self._store.register_label(seg.speaker)
            self._say(f"已识别说话人：{len(self._store.speakers())} 人")

    def stop(self) -> None:
        self._stop.set()
        self._capture.stop()
        deadline = time.time() + 20
        while not self._translate_queue.empty() and time.time() < deadline:
            time.sleep(0.2)
        if self._translator:
            for seg in self._store.segments():
                if not seg.translation and seg.lang != "zh":
                    try:
                        seg.translation = self._translator.translate(seg.text, seg.lang)
                    except TranslationError:
                        pass
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_recorder.py -v`
Expected: 1 passed

- [ ] **Step 5: 提交**

```bash
git add meeting_bot/recorder.py tests/test_recorder.py
git commit -m "feat: 录制编排模块（多线程流水线 + 事件回调）"
```

---

### Task 10: 字幕浮窗（手动验证）

**Files:**
- Create: `meeting_bot/subtitle.py`, `scripts/check_subtitle.py`

- [ ] **Step 1: 实现** `meeting_bot/subtitle.py`

```python
# -*- coding: utf-8 -*-
"""字幕浮窗：置顶、半透明、可拖拽、滚轮调字号。"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .transcript import Segment


class SubtitleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 160); border-radius: 10px;")
        self._font_size = 20
        self._speaker = QLabel("发言人")
        self._zh = QLabel("")
        self._orig = QLabel("")
        self._speaker.setStyleSheet("color: #4FC3F7; font-weight: bold; background: transparent;")
        self._zh.setStyleSheet("color: white; background: transparent;")
        self._orig.setStyleSheet("color: #B0BEC5; background: transparent;")
        for w in (self._speaker, self._zh, self._orig):
            w.setWordWrap(True)
        lay = QVBoxLayout(self)
        for w in (self._speaker, self._zh, self._orig):
            lay.addWidget(w)
        self.resize(720, 180)
        self._apply_font()
        self._drag = None

    def _apply_font(self):
        f = self.font()
        f.setPointSize(self._font_size)
        self._speaker.setFont(f)
        self._zh.setFont(f)
        f2 = self.font()
        f2.setPointSize(max(10, self._font_size - 4))
        self._orig.setFont(f2)

    def show_segment(self, seg: Segment, display_name: str):
        self._speaker.setText(display_name)
        self._zh.setText(seg.translation or seg.text)
        self._orig.setText(f"({seg.lang}) {seg.text}" if seg.translation else "")
        self.adjustSize()
        self.show()
        self.raise_()

    def change_font(self, delta: int):
        self._font_size = max(12, min(40, self._font_size + delta))
        self._apply_font()

    # --- 拖拽 ---
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        self.change_font(1 if e.angleDelta().y() > 0 else -1)
```

- [ ] **Step 2: 创建手动验证脚本** `scripts/check_subtitle.py`

```python
# -*- coding: utf-8 -*-
"""手动验证字幕浮窗：python scripts/check_subtitle.py"""
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from meeting_bot.subtitle import SubtitleWindow
from meeting_bot.transcript import Segment

app = QApplication(sys.argv)
win = SubtitleWindow()
samples = [
    Segment(0, 3, "Good morning everyone, let's get started.", lang="en",
            translation="大家早上好，我们开始吧。", speaker="SPEAKER_00"),
    Segment(3, 6, "Buenos días, gracias por venir.", lang="es",
            translation="早上好，感谢各位到来。", speaker="SPEAKER_01"),
]
i = 0


def tick():
    global i
    seg = samples[i % len(samples)]
    win.show_segment(seg, f"说话人{i % len(samples) + 1}")
    i += 1


timer = QTimer()
timer.timeout.connect(tick)
timer.start(3000)
tick()
sys.exit(app.exec())
```

- [ ] **Step 3: 运行验证**

Run: `python scripts/check_subtitle.py`
Expected: 置顶黑底浮窗循环显示中英双语字幕；可拖拽、滚轮调字号。

- [ ] **Step 4: 提交**

```bash
git add meeting_bot/subtitle.py scripts/check_subtitle.py
git commit -m "feat: 字幕浮窗（置顶/半透明/可拖拽/字号可调）"
```

---

### Task 11: 导出模块（TDD）

**Files:**
- Create: `meeting_bot/export.py`, `tests/test_export.py`

- [ ] **Step 1: 写失败测试** `tests/test_export.py`

```python
# -*- coding: utf-8 -*-
from datetime import datetime

from docx import Document

from meeting_bot.export import export_all, meeting_dir, minutes_to_docx, write_transcript
from meeting_bot.transcript import TranscriptStore
from meeting_bot.translator import ActionItem, MinutesResult


def _settings(tmp_path):
    from meeting_bot.config import Settings
    return Settings("https://x", "sk-test", "m1", "m2", "medium", "", tmp_path / "会议记录", tmp_path)


def test_meeting_dir_created(tmp_path):
    s = _settings(tmp_path)
    d = meeting_dir(s, "周会", datetime(2026, 10, 16, 9, 30))
    assert d.exists()
    assert d.name == "2026-10-16_0930_周会"


def test_write_transcript(tmp_path):
    s = _settings(tmp_path)
    d = meeting_dir(s, "周会", datetime(2026, 10, 16, 9, 30))
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")
    p = write_transcript(store, d, "周会", datetime(2026, 10, 16, 9, 30))
    content = p.read_text(encoding="utf-8")
    assert "hello" in content
    assert "译文" in content


def test_minutes_to_docx(tmp_path):
    m = MinutesResult(title="周会纪要", summary="概要",
                      key_points=["要点1"], decisions=["结论1"],
                      action_items=[ActionItem("写方案", "张三", "10/20")],
                      open_questions=["问题1"])
    p = minutes_to_docx(m, tmp_path / "纪要.docx")
    doc = Document(str(p))
    texts = "\n".join(par.text for par in doc.paragraphs)
    assert "周会纪要" in texts
    assert "张三" in texts
    assert "10/20" in texts


def test_export_all(tmp_path):
    s = _settings(tmp_path)
    store = TranscriptStore()
    store.add(0.0, 2.0, "hello", lang="en")

    class FakeTranslator:
        def minutes(self, md):
            return MinutesResult(title="周会纪要", summary="概要")

    t_path, m_path = export_all(store, s, "周会", datetime(2026, 10, 16, 9, 30),
                                translator=FakeTranslator())
    assert t_path.exists() and m_path.exists()
    assert t_path.suffix == ".md" and m_path.suffix == ".docx"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL

- [ ] **Step 3: 实现** `meeting_bot/export.py`

```python
# -*- coding: utf-8 -*-
"""导出：逐字稿 Markdown + 纪要 Word。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from docx import Document

from .config import Settings
from .transcript import TranscriptStore
from .translator import MinutesResult, Translator


def meeting_dir(settings: Settings, title: str, started_at: datetime) -> Path:
    d = settings.output_dir / f"{started_at:%Y-%m-%d_%H%M}_{title}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_transcript(store: TranscriptStore, directory: Path,
                     title: str, started_at: datetime) -> Path:
    path = directory / "逐字稿.md"
    path.write_text(store.to_markdown(title, started_at), encoding="utf-8")
    return path


def minutes_to_docx(minutes: MinutesResult, path: Path) -> Path:
    doc = Document()
    doc.add_heading(minutes.title, level=1)
    doc.add_paragraph(minutes.summary)
    doc.add_heading("讨论要点", level=2)
    for p in minutes.key_points:
        doc.add_paragraph(p, style="List Bullet")
    doc.add_heading("会议结论", level=2)
    for p in minutes.decisions:
        doc.add_paragraph(p, style="List Bullet")
    doc.add_heading("行动项", level=2)
    for a in minutes.action_items:
        due = f"（截止：{a.due}）" if a.due else ""
        doc.add_paragraph(f"{a.task} — 负责人：{a.owner or '待定'}{due}", style="List Bullet")
    doc.add_heading("待跟进问题", level=2)
    for p in minutes.open_questions:
        doc.add_paragraph(p, style="List Bullet")
    doc.save(str(path))
    return path


def export_all(store: TranscriptStore, settings: Settings, title: str,
               started_at: datetime,
               translator: Translator | None = None) -> tuple[Path, Path]:
    directory = meeting_dir(settings, title, started_at)
    transcript_path = write_transcript(store, directory, title, started_at)
    if translator is not None:
        minutes = translator.minutes(store.to_markdown(title, started_at))
    else:
        minutes = MinutesResult(title=title, summary="（翻译接口不可用，未生成纪要）")
    minutes_path = minutes_to_docx(minutes, directory / "纪要.docx")
    return transcript_path, minutes_path
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_export.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add meeting_bot/export.py tests/test_export.py
git commit -m "feat: 导出模块（逐字稿 md + 纪要 docx）"
```

---

### Task 12: 主窗口与启动入口（手动验证）

**Files:**
- Create: `meeting_bot/app.py`, `meeting_bot/__main__.py`

- [ ] **Step 1: 实现** `meeting_bot/app.py`

```python
# -*- coding: utf-8 -*-
"""主窗口：开始/结束录制、说话人改名、状态显示。"""
from __future__ import annotations

import sys
import threading
from datetime import datetime

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from .config import ConfigError, Settings, load_settings
from .diarization import Diarizer, DiarizationError
from .export import export_all
from .recorder import Recorder
from .subtitle import SubtitleWindow
from .translator import Translator


class RecorderBridge(QObject):
    """把工作线程的事件桥接到 Qt 主线程（线程安全）。"""
    segment = pyqtSignal(object)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("会议翻译助手")
        self.recorder: Recorder | None = None
        self.subtitle = SubtitleWindow()
        self.bridge = RecorderBridge()
        self.bridge.segment.connect(self._on_segment)
        self.bridge.status.connect(self._on_status)
        self.bridge.finished.connect(self._on_finished)

        central = QWidget()
        lay = QVBoxLayout(central)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("会议名称（用于文件命名，可留空）")
        lay.addWidget(self.title_input)

        row = QHBoxLayout()
        self.start_btn = QPushButton("开始记录")
        self.stop_btn = QPushButton("结束并生成纪要")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)

        self.status = QLabel("就绪")
        lay.addWidget(self.status)

        self.speaker_table = QTableWidget(0, 2)
        self.speaker_table.setHorizontalHeaderLabels(["声纹标签", "名字"])
        lay.addWidget(self.speaker_table)

        self.setCentralWidget(central)
        self.resize(520, 420)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.speaker_table.itemChanged.connect(self._rename_speaker)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_speakers)
        self._timer.start(5000)

    def _start(self):
        try:
            translator = Translator(self.settings)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"翻译接口初始化失败：{e}")
            return
        try:
            diarizer = Diarizer(self.settings)
        except DiarizationError as e:
            QMessageBox.warning(self, "说话人功能不可用", f"{e}\n\n将继续记录，但无法区分说话人。")
            diarizer = None
        self.recorder = Recorder(self.settings, translator=translator, diarizer=diarizer)
        self.recorder.on_segment(self.bridge.segment.emit)
        self.recorder.on_status(self.bridge.status.emit)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        threading.Thread(
            target=self.recorder.start,
            args=(self.title_input.text().strip(),),
            daemon=True,
        ).start()

    def _on_segment(self, seg):
        name = self.recorder.store.display_name(seg.speaker)
        self.subtitle.show_segment(seg, name)

    def _on_status(self, msg):
        self.status.setText(msg)

    def _refresh_speakers(self):
        if not self.recorder:
            return
        self.speaker_table.blockSignals(True)
        speakers = self.recorder.store.speakers()
        self.speaker_table.setRowCount(len(speakers))
        for i, sp in enumerate(speakers):
            self.speaker_table.setItem(i, 0, QTableWidgetItem(sp.label))
            self.speaker_table.setItem(i, 1, QTableWidgetItem(sp.name))
        self.speaker_table.blockSignals(False)

    def _rename_speaker(self, item):
        if item.column() != 1 or not self.recorder:
            return
        label_item = self.speaker_table.item(item.row(), 0)
        if label_item:
            self.recorder.store.set_speaker_name(label_item.text(), item.text())

    def _stop(self):
        if not self.recorder:
            return
        self.stop_btn.setEnabled(False)
        self.status.setText("正在收尾（补译 + 生成纪要）…")

        def finish():
            self.recorder.stop()
            title = self.title_input.text().strip() or self.recorder.title or "未命名会议"
            try:
                transcript_path, minutes_path = export_all(
                    self.recorder.store, self.settings, title,
                    self.recorder.started_at or datetime.now(),
                    translator=self.recorder.translator)
                self.bridge.finished.emit(True,
                    f"已生成：\n逐字稿：{transcript_path}\n纪要：{minutes_path}")
            except Exception as e:
                self.bridge.finished.emit(False, f"导出失败：{e}")

        threading.Thread(target=finish, daemon=True).start()

    def _on_finished(self, ok: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("完成" if ok else "导出失败")
        if ok:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)


def main():
    app = QApplication(sys.argv)
    try:
        settings = load_settings()
    except ConfigError as e:
        QMessageBox.critical(None, "配置错误", f"{e}\n\n请复制 .env.example 为 .env 并填写配置。")
        sys.exit(1)
    win = MainWindow(settings)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现** `meeting_bot/__main__.py`

```python
# -*- coding: utf-8 -*-
from .app import main

main()
```

- [ ] **Step 3: 运行验证（无 .env 时应弹配置错误）**

Run: `python -m meeting_bot`
Expected: 弹出"配置错误"对话框后退出（.env 尚未创建时）。

- [ ] **Step 4: 提交**

```bash
git add meeting_bot/app.py meeting_bot/__main__.py
git commit -m "feat: 主窗口与启动入口（Qt 线程桥接）"
```

---

### Task 13: 桌面快捷方式

**Files:**
- Create: `run_meeting_bot.bat`, `install_shortcut.py`

- [ ] **Step 1: 创建** `run_meeting_bot.bat`

```bat
@echo off
chcp 65001 >nul
cd /d %~dp0
"%~dp0.venv\Scripts\python.exe" -m meeting_bot
if errorlevel 1 pause
```

- [ ] **Step 2: 创建** `install_shortcut.py`

```python
# -*- coding: utf-8 -*-
"""在桌面创建快捷方式。用法：python install_shortcut.py"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BAT = ROOT / "run_meeting_bot.bat"
LNK = Path.home() / "Desktop" / "会议翻译助手.lnk"

ps = (
    "$ws = New-Object -ComObject WScript.Shell;"
    f"$s = $ws.CreateShortcut('{LNK}');"
    f"$s.TargetPath = '{BAT}';"
    f"$s.WorkingDirectory = '{ROOT}';"
    "$s.Save()"
)
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
print(f"已创建桌面快捷方式：{LNK}")
```

- [ ] **Step 3: 运行并验证**

Run: `python install_shortcut.py`
Expected: 输出快捷方式路径；桌面出现"会议翻译助手"图标；双击能弹出主窗口（.env 配置好后）。

- [ ] **Step 4: 提交**

```bash
git add run_meeting_bot.bat install_shortcut.py
git commit -m "feat: 一键启动脚本与桌面快捷方式安装器"
```

---

### Task 14: 延迟基准 + 端到端演练 + README

**Files:**
- Create: `scripts/bench_translate.py`, `README.md`

- [ ] **Step 1: 创建翻译延迟基准脚本** `scripts/bench_translate.py`

```python
# -*- coding: utf-8 -*-
"""翻译延迟基准：python scripts/bench_translate.py

对 .env 里的 TRANSLATE_MODEL 连发 3 条测试翻译，输出平均耗时。
若平均耗时 > 5 秒，建议在 .env 中更换更快模型（如 deepseek-v4-flash）。
"""
import time

from meeting_bot.config import load_settings
from meeting_bot.translator import Translator

SAMPLES = [
    ("Good morning everyone, let's get started with the allergy clinic update.", "en"),
    ("El envío llegará la próxima semana, sin falta.", "es"),
    ("We need to finalize the pricing before Friday.", "en"),
]

t = Translator(load_settings())
total = 0.0
for text, lang in SAMPLES:
    t0 = time.perf_counter()
    out = t.translate(text, lang)
    dt = time.perf_counter() - t0
    total += dt
    print(f"[{dt:5.2f}s] {text}\n        → {out}")
print(f"\n平均耗时：{total / len(SAMPLES):.2f} 秒/句")
```

- [ ] **Step 2: 运行基准并记录结果**

Run: `python scripts/bench_translate.py`
Expected: 输出 3 条译文与平均耗时；将结果记入 ledger。若平均 > 5s，把 .env 的 `TRANSLATE_MODEL` 改为备选模型（如 `glm-5.2`）重测一次，取更快者。

- [ ] **Step 3: 端到端演练（手动，按清单执行）**

准备：播放一段 5 分钟的多说话人英文/西语访谈视频（系统正常出声）。
1. 双击桌面快捷方式 → 主窗口出现
2. 输入会议名 → 点"开始记录" → 播放视频
3. 预期：10 秒内字幕浮窗出现中文，说话人标签出现在表格中（约 15~30 秒后）
4. 在表格里给说话人改名，确认字幕上的名字同步变化
5. 视频放完 → 点"结束并生成纪要" → 等待弹窗
6. 检查 `会议记录/YYYY-MM-DD_HHMM_<会议名>/逐字稿.md` 和 `纪要.docx` 内容正确
7. 记录异常项到 ledger，按 systematic-debugging 流程修复

- [ ] **Step 4: 编写** `README.md`

```markdown
# 会议翻译助手（Teams Meeting Bot）

本地旁听 Teams 会议：实时识别多语种发言 → 翻译成中文 → 置顶字幕浮窗；
会议结束后自动生成双语逐字稿（Markdown）与中文会议纪要（Word）。

## 首次安装

1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. 复制 `.env.example` 为 `.env`：
   - `OPENCODE_API_KEY`：从 `C:\Users\<你>\.pi\agent\models.json` 中 provider
     `cc-switch-open-code-go` 的 `apiKey` 复制
   - `HF_TOKEN`：到 https://huggingface.co/settings/tokens 创建 token，
     并打开 https://huggingface.co/pyannote/speaker-diarization-3.1
     点击 "Agree and access repository" 接受使用协议
3. `python install_shortcut.py` 创建桌面快捷方式
4. 首次运行会自动下载语音模型（约 1.5GB，请耐心等待）

## 使用

1. 开会前：双击桌面"会议翻译助手"；确认 Teams 扬声器输出走系统默认设备
2. 填写会议名 → 点"开始记录"
3. 字幕浮窗可拖拽，滚轮调字号；在说话人表格里把声纹标签改成真实名字
4. 会议结束 → 点"结束并生成纪要"，产物在 `会议记录/日期_时间_会议名/` 下

## 已知限制

- 字幕延迟约 3~8 秒（识别 + 翻译耗时）
- 说话人区分基于声纹聚类，多人抢话时准确率下降
- 只捕获系统播放的声音；你的麦克风声音不会进入记录
```

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过。

- [ ] **Step 6: 最终提交**

```bash
git add -A && git commit -m "docs: README 与基准脚本；端到端演练通过"
```

---

## 风险与对策

| 风险 | 对策 |
|---|---|
| pyannote 模型下载需要 HF token + 接受协议 | README 给出步骤；无 token 时降级为不分说话人继续工作 |
| OpenCode Go 对 `thinking` 参数不支持 | translator 已实现 400 自动降级重试 |
| flash 模型翻译延迟过高 | Task 14 基准测试，.env 一键换模型 |
| whisper medium 在 CPU 上识别慢于实时 | 先测 realtime factor；超标则 .env 改 `small` 或 `large-v3-turbo` 对比 |
| Teams 输出设备不是默认扬声器 | README 说明；捕获模块取默认设备 |
| 会议声音经过降噪耳机（蓝牙） | 蓝牙耳机播放时 loopback 仍可捕获默认设备输出（WASAPI 层面）；验证阶段确认 |
