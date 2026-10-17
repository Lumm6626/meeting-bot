# -*- coding: utf-8 -*-
"""语音识别：本地 faster-whisper（默认）+ Groq 云端 whisper（低延迟模式）。

本地模式攒批识别（ASR_BATCH_SECONDS/ASR_BATCH_MAX_WAIT），字幕成批出现；
云端模式（ASR_PROVIDER=groq）每条语句到达即识别，字幕延迟约 5~10 秒，
代价是音频会上传 Groq 云端，且免费额度有限速（约 20 req/min）。
"""
from __future__ import annotations

import io
import os
import time
import wave

import numpy as np
import openai
from faster_whisper import WhisperModel

from .config import ConfigError, Settings
from .transcript import Segment

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "whisper-large-v3-turbo"
CLOUD_MAX_ATTEMPTS = 3                  # 首次 + 2 次重试
CLOUD_RETRY_BACKOFF = (1.0, 2.0)        # 第 n 次失败后等几秒再试


class CloudASRError(Exception):
    """云端识别不可用（重试后仍失败）或响应结构异常。"""


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


def encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """float32 [-1,1] 单声道音频 → 16-bit PCM WAV 字节流。

    Groq 的 transcriptions 只接受文件上传，故这里自己打包。采样率沿用入参
    （Recorder 恒传 16000），不做重采样以免在实时链路上再添一段延迟。
    """
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)       # 单声道
            wf.setsampwidth(2)       # 16 bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


def _field(obj, name: str, default=None):
    """从 pydantic 对象或 dict 里取字段（兼容 fake client 的两种返回形态）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _segments_from_response(resp) -> list[Segment]:
    """verbose_json 响应 → Segment 列表；结构异常一律 CloudASRError。"""
    lang = _field(resp, "language", "") or ""
    raw = _field(resp, "segments", None)
    if not raw:
        # 没有 segments 就无从生成字幕：视为接口异常（重试无益，由调用方丢弃该批）
        raise CloudASRError("云端识别响应缺少 segments（接口格式异常）")
    result: list[Segment] = []
    for s in raw:
        text = str(_field(s, "text", "") or "").strip()
        if not text:
            continue                       # 空文本过滤（静音段/呼吸声）
        start = float(_field(s, "start", 0.0) or 0.0)
        end = float(_field(s, "end", start) or start)
        result.append(Segment(start=start, end=max(start, end), text=text, lang=str(lang)))
    return result


class CloudASREngine:
    """Groq whisper 云端识别：与 ASREngine 同接口，换取低延迟（5-10s 字幕）。"""

    def __init__(self, settings: Settings):
        key = ((os.getenv("GROQ_API_KEY") or "").strip()
               or (getattr(settings, "groq_api_key", "") or "").strip())
        if not key:
            raise ConfigError("缺少 GROQ_API_KEY：请到 https://console.groq.com 免费注册"
                              "取 key，填入 .env 的 GROQ_API_KEY")
        self._client = openai.OpenAI(base_url=GROQ_BASE_URL, api_key=key,
                                     timeout=30.0, max_retries=0)
        self._model = GROQ_MODEL

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[Segment]:
        """音频 → Segment 列表（语种自动检测、带时间戳）。

        重试 2 次后仍失败则抛 CloudASRError（Recorder 侧丢弃该批并告警）。
        """
        wav = encode_wav(audio, sample_rate)
        last_error: Exception | None = None
        for attempt in range(CLOUD_MAX_ATTEMPTS):
            try:
                resp = self._client.audio.transcriptions.create(
                    model=self._model,
                    file=("utterance.wav", wav, "audio/wav"),
                    response_format="verbose_json",   # 只有 verbose_json 带 segments/language
                )
            except openai.APIError as e:
                # APIError 覆盖限速/5xx/连接失败 → 退避重试
                last_error = e
                if attempt + 1 < CLOUD_MAX_ATTEMPTS:
                    time.sleep(CLOUD_RETRY_BACKOFF[min(attempt,
                                                       len(CLOUD_RETRY_BACKOFF) - 1)])
                continue
            except Exception as e:
                # 非 APIError（SDK 边界异常：参数/编码/连接池）→ 不重试，直接包装。
                # 若让它直穿，识别线程会当场死掉，整场再无字幕。
                raise CloudASRError(f"云端识别请求异常：{e}") from e
            try:
                return _segments_from_response(resp)
            except CloudASRError:
                raise
            except Exception as e:
                # 结构异常（segments 元素缺字段/类型不对）：确定性错误，重试无益
                raise CloudASRError(f"云端识别响应结构异常：{e}") from e
        raise CloudASRError(
            f"云端识别不可用（重试 {CLOUD_MAX_ATTEMPTS - 1} 次仍失败）：{last_error}"
        ) from last_error
