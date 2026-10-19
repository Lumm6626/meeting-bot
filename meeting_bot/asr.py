# -*- coding: utf-8 -*-
"""语音识别：本地 faster-whisper（默认）+ Groq 云端 whisper（低延迟模式）。

本地模式攒批识别（ASR_BATCH_SECONDS/ASR_BATCH_MAX_WAIT），字幕成批出现；
云端模式（ASR_PROVIDER=groq）每条语句到达即识别，字幕延迟约 5~10 秒，
代价是音频会上传 Groq 云端，且免费额度有限速（约 20 req/min）；
单次请求超时 10s，失败按 1s/2s 退避重试（可被停止信号打断）。
"""
from __future__ import annotations

import io
import os
import time
import wave
from typing import Callable

import numpy as np
import openai
from faster_whisper import WhisperModel

from .config import ConfigError, Settings
from .transcript import Segment

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "whisper-large-v3-turbo"
CLOUD_TIMEOUT = 10.0                    # 单次请求超时（秒）：目标字幕延迟 5~10s
CLOUD_MAX_ATTEMPTS = 3                  # 首次 + 2 次重试
CLOUD_RETRY_BACKOFF = (1.0, 2.0)        # 第 n 次失败后等几秒再试
CLOUD_RETRYABLE_STATUS = (408, 409, 429)  # 超时/冲突/限速：4xx 里唯三值得重试的
CLOUD_STOP_POLL = 0.1                   # 可中断退避的轮询粒度（秒）


class CloudASRError(Exception):
    """云端识别不可用（重试后仍失败）或响应结构异常。"""


class CloudASRConfigError(ConfigError):
    """云端识别的配置错误（key 无效/无权限）：重试无益，需用户改配置。

    继承 ConfigError：Recorder 侧对「配置类错误」已有「只告警一次」的处理路径
    （缺 key 与本类错误的处置完全相同），无需再分叉。
    """


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


def _normalize_lang(lang: str) -> str:
    """Groq 返回全称（English/Spanish/Chinese），统一为 ISO 码。

    下游用 ``lang == "zh"`` 跳过中文翻译，不一致会导致中文白多走一次翻译请求。
    """
    table = {"chinese": "zh", "english": "en", "spanish": "es",
             "french": "fr", "german": "de", "japanese": "ja",
             "korean": "ko", "russian": "ru", "portuguese": "pt",
             "italian": "it", "arabic": "ar"}
    return table.get(str(lang).strip().lower(), str(lang).strip()[:2].lower() or "")


def _is_mostly_cjk(text: str) -> bool:
    """判断文本是否以中文为主（用于语种混说时决定跳过翻译）。

    Groq 的 verbose_json 只给一个文件级 language：中英混说的会议里英文句会被
    标成 zh 而跳过翻译。这里按字符占比逐句判定，避免中文会议里的英文不被翻译。
    """
    letters = [c for c in text if not c.isspace() and (c.isalpha() or "一" <= c <= "鿿")]
    if not letters:
        return False
    cjk = sum(1 for c in letters if "一" <= c <= "鿿")
    return cjk / len(letters) >= 0.5


def _segments_from_response(resp) -> list[Segment]:
    """verbose_json 响应 → Segment 列表；结构异常一律 CloudASRError。"""
    lang = _normalize_lang(_field(resp, "language", "") or "")
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
        # 中文句标 zh（跳过翻译）；非中文句不能沿用文件级 zh 标签（否则会漏翻），
        # 标成空串让下游按"未知语种"处理（照常翻译、不显示语种前缀）
        if _is_mostly_cjk(text):
            seg_lang = "zh"
        elif str(lang) == "zh":
            seg_lang = ""
        else:
            seg_lang = str(lang)
        result.append(Segment(start=start, end=max(start, end), text=text, lang=seg_lang))
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
                                     timeout=CLOUD_TIMEOUT, max_retries=0)
        # 模型/语言/领域提示词都可配：large-v3 比 turbo 更准且实测不更慢；
        # 指定 language 能避免两三秒短句被误判成其它语种；prompt 喂领域词表提升专名准确率
        self._model = (getattr(settings, "groq_asr_model", "") or GROQ_MODEL).strip()
        self._language = (getattr(settings, "asr_language", "") or "").strip()
        self._prompt = (getattr(settings, "asr_prompt", "") or "").strip()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   should_stop: Callable[[], bool] | None = None) -> list[Segment]:
        """音频 → Segment 列表（语种自动检测、带时间戳）。

        重试 2 次后仍失败则抛 CloudASRError（Recorder 侧丢弃该批并告警）。
        确定性错误不重试：key 无效/无权限 → CloudASRConfigError（一次性告警），
        其余 4xx（400/404/413/422…）→ CloudASRError（重试只会白耗配额与延迟）。
        should_stop 给出时，退避睡眠切片轮询，停止信号能立刻打断重试。
        """
        wav = encode_wav(audio, sample_rate)
        last_error: Exception | None = None
        for attempt in range(CLOUD_MAX_ATTEMPTS):
            try:
                kwargs: dict = dict(
                    model=self._model,
                    file=("utterance.wav", wav, "audio/wav"),
                    response_format="verbose_json",   # 只有 verbose_json 带 segments/language
                    temperature=0.0,                  # 减少幻觉式改写
                )
                if self._language:
                    kwargs["language"] = self._language
                if self._prompt:
                    kwargs["prompt"] = self._prompt
                resp = self._client.audio.transcriptions.create(**kwargs)
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                # 401/403：key 无效或无权限。确定性错误，重试 3 次只是白耗配额
                # 与 3 倍延迟，用户必须改配置才能恢复 → 交给 ConfigError 路径一次性告警。
                raise CloudASRConfigError(
                    f"GROQ_API_KEY 无效或无权限（HTTP {e.status_code}）：请到 "
                    "https://console.groq.com 重新生成 key 并更新 .env"
                ) from e
            except openai.APIStatusError as e:
                # 限速/请求超时/冲突值得退避；其余 4xx（400 参数错、404 模型不存在、
                # 413 音频过大、422 不可处理）都是确定性错误，重试 3 次纯属浪费。
                if e.status_code not in CLOUD_RETRYABLE_STATUS and e.status_code < 500:
                    raise CloudASRError(
                        f"云端识别请求被拒绝（HTTP {e.status_code}）：{e}"
                    ) from e
                last_error = e
                if attempt + 1 < CLOUD_MAX_ATTEMPTS:
                    self._sleep_backoff(attempt, should_stop)
                continue
            except openai.APIConnectionError as e:
                # 连接失败/超时（APITimeoutError 是它的子类）→ 瞬时故障，退避重试
                last_error = e
                if attempt + 1 < CLOUD_MAX_ATTEMPTS:
                    self._sleep_backoff(attempt, should_stop)
                continue
            except openai.APIError as e:
                # 兜底：无法分类的 APIError（如响应校验失败）按瞬时故障处理
                last_error = e
                if attempt + 1 < CLOUD_MAX_ATTEMPTS:
                    self._sleep_backoff(attempt, should_stop)
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

    @staticmethod
    def _sleep_backoff(attempt: int, should_stop: Callable[[], bool] | None = None) -> None:
        """退避睡眠；给了 should_stop 就切片轮询，停止信号可立刻打断。

        原实现 time.sleep(1) + time.sleep(2) 无法打断：录制结束时识别线程卡在退避里，
        stop() 只能干等 join，用户感知是「停不下来」。切成 0.1s 一轮后最多多等 0.1s。
        """
        backoff = CLOUD_RETRY_BACKOFF[min(attempt, len(CLOUD_RETRY_BACKOFF) - 1)]
        if should_stop is None:
            time.sleep(backoff)
            return
        deadline = time.monotonic() + backoff
        while time.monotonic() < deadline:
            if should_stop():
                raise CloudASRError("识别被停止")
            time.sleep(CLOUD_STOP_POLL)
