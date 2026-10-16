# -*- coding: utf-8 -*-
"""录制编排：串联捕获→切句→识别→标注→翻译→存储。"""
from __future__ import annotations

import queue
import re
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

DIARIZE_INTERVAL = 60.0       # 每 60 秒跑一次说话人聚类
DIARIZE_WINDOW = 300.0        # 每次只处理最近 5 分钟音频（滑动窗口，避免长会 O(T²) 重算）
DIARIZE_JOIN_TIMEOUT = 180.0  # stop() 等说话人标注线程收尾的上限
PRODUCER_DONE_TIMEOUT = 10.0  # stop() 等主循环完成尾部 flush 的上限
TRANSLATE_DRAIN_TIMEOUT = 20.0  # stop() 等翻译 worker 排空退出的上限
TRANSLATE_POLL_TIMEOUT = 0.5  # 翻译 worker 队列轮询间隔


def map_local_labels(prev_labeled: list[tuple[float, float, str]],
                     new_spans: list[tuple[float, float, str]],
                     overlap_start: float) -> dict[str, str]:
    """把新窗口的局部标签映射到全局标签。

    prev_labeled: list[(start, end, global_label)] 重叠区域内已标注的语句（旧全局标签）
    new_spans: list[(start, end, local_label)] 新窗口 pyannote 结果（绝对时间轴）
    overlap_start: float 重叠区起点（秒）

    对每个 local_label：在 overlap_start 之后的 prev_labeled 中，累计与该 label 的
    span 重叠的时长，按 global_label 分组求和，取最大者（平局取先出现者）。
    返回 {local_label: global_label}；无法映射的 local_label 不出现在结果中。
    """
    # 保持局部标签首次出现顺序（平局判「先出现者」时用到）
    local_labels: list[str] = []
    for _, _, label in new_spans:
        if label not in local_labels:
            local_labels.append(label)

    mapping: dict[str, str] = {}
    for local in local_labels:
        spans = [(s, e) for s, e, lbl in new_spans if lbl == local]
        votes: dict[str, float] = {}   # 插入序 = prev_labeled 顺序 → 平局时 max() 取先出现者
        for p_start, p_end, global_label in prev_labeled:
            if p_end <= overlap_start:
                continue               # 整条都在重叠区之前 → 不参与投票
            overlap = 0.0
            for s, e in spans:
                overlap += max(0.0, min(e, p_end) - max(s, p_start))
            if overlap > 0.0:
                votes[global_label] = votes.get(global_label, 0.0) + overlap
        if votes:
            mapping[local] = max(votes, key=votes.__getitem__)
    return mapping


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
        # producer-done：主循环（生产者）已完成尾部 flush，翻译 worker 方可排空退出。
        # 只靠 _stop 会有窗口期：队列一度为空时 worker 退出，之后 flush 的尾部语句无人翻译。
        self._producer_done = threading.Event()
        self._worker_done = threading.Event()   # 翻译 worker 已退出（排空完成）
        self._running = threading.Event()       # 主循环是否在跑（start 时 set、退出时 clear）
        self._translate_thread: threading.Thread | None = None
        self._diarize_thread: threading.Thread | None = None
        self._translate_queue: Queue[Segment] = Queue()
        # 已 emit 的 seq：stop() 兜底补发时去重（store 跨 start 不重置，seq 全局唯一 → 无需 clear）
        self._emitted: set[int] = set()
        # _emit_once 的记账锁：worker 与 stop() 会并发对同一 seq 兜底补发，
        # 「检查 + 记账」必须原子，否则两者都判定「未发过」→ 重复 emit（审查 Minor-1）
        self._emit_lock = threading.Lock()
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

    def _emit_once(self, seg: Segment) -> None:
        """发送语句事件并记账（每个 seq 最多一次）。

        回调异常只告警：翻译 worker 与 stop() 都不能因消费者异常而中断（审查 N2）。
        去重同时兜住「stop() 兜底补发」与「worker 迟到 emit」的竞态。

        先记账再发送（at-most-once）：「检查 + 记账」若不加锁，worker 与 stop() 会并发
        判定「未发过」→ 同一 seq 冒出两条事件（审查 Minor-1）。锁只护记账不包住 _emit：
        用户回调可能很慢甚至反向调用本类，在锁内执行有死锁风险。
        """
        with self._emit_lock:
            if seg.seq in self._emitted:
                return
            self._emitted.add(seg.seq)
        try:
            self._emit(seg)
        except Exception as e:
            self._say(f"[警告] 语句事件回调异常：{e}")

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
        if self._stop.is_set():
            # stop() 先于 start() 落地（用户点「开始」后立刻点「结束」，start 所在线程晚一步执行）：
            # 此时若照常开采集，主循环会因 _stop 已置位而立刻退出，且再无人调用 stop() ——
            # 采集被打开却永不关闭（loopback 占用到进程退出）。守卫必须在 _capture.start() 之前。
            self._say("录制已取消")
            return
        self.title = title
        self._started_at = datetime.now()
        self._say("开始录制（等待语音…）")
        self._producer_done.clear()
        self._worker_done.clear()
        self._running.set()
        self._capture.start()
        self._translate_thread = threading.Thread(target=self._translate_worker, daemon=True)
        self._translate_thread.start()
        if self._diarizer:
            self._diarize_thread = threading.Thread(target=self._diarize_worker, daemon=True)
            self._diarize_thread.start()
        self._main_loop()

    def _main_loop(self) -> None:
        buf = UtteranceBuffer()
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._capture.read(timeout=0.5)
                except queue.Empty:
                    # read() 契约：超时抛 queue.Empty，采集仍在进行 → 继续读
                    continue
                if chunk is None:
                    break        # 结束哨兵：采集已停止
                self._diarize_audio.append(chunk)
                for offset, utterance in buf.push(chunk):
                    self._process_utterance(utterance, offset / 16000)
            for offset, utterance in buf.flush():
                self._process_utterance(utterance, offset / 16000)
        finally:
            # 尾部 flush 结束（或主循环异常）才算生产者完结；顺序不能反：
            # 必须先 set(producer_done) 再 clear(running)，否则 stop() 可能跳过等待。
            self._producer_done.set()
            self._running.clear()

    def _process_utterance(self, audio: np.ndarray, start_offset: float) -> None:
        try:
            if self._asr is None:
                # 惰性创建：重型模型只在首句到来时加载（测试注入 fake 时不走此行）
                self._asr = ASREngine(self._settings)
            segments = self._asr.transcribe(audio)
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
        try:
            # producer-done 模式：只认「生产者完结 + 队列排空」才退出。
            # 若用 _stop 判定，尾段仍在内核 ASR（0.5~3s）时 worker 就会因队列暂空而退出，
            # flush 产出的 Segment 将永远无人翻译、无人 emit。
            while not self._producer_done.is_set() or not self._translate_queue.empty():
                try:
                    seg = self._translate_queue.get(timeout=TRANSLATE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                try:
                    if seg.lang == "zh":
                        seg.translation = seg.text
                    else:
                        seg.translation = self._translator.translate(seg.text, seg.lang)
                except TranslationError:
                    seg.translation = ""   # 会议结束后统一补译
                    self._say(f"[警告] 翻译失败（{seg.text[:20]}…），已保留原文")
                except Exception as e:
                    # 意外异常（openai 库边界异常、回调异常等）不得杀死 worker：
                    # 否则后续语句只入库、永不翻译、永不 emit（审查 N2）。
                    seg.translation = ""
                    self._say(f"[警告] 翻译异常（{seg.text[:20]}…）：{e}，已保留原文")
                self._emit_once(seg)
        finally:
            self._worker_done.set()

    def _take_diarize_window(self) -> tuple[np.ndarray, float]:
        """取最近 DIARIZE_WINDOW 秒音频，返回 (窗口音频, 窗口起点绝对秒数)。

        长会每次对全量累积音频重跑 pyannote 是 O(T²)；只喂滑动窗口后单次开销恒定。
        """
        n = int(DIARIZE_WINDOW * 16000)
        chunks = list(self._diarize_audio)      # 快照：主循环可能正在 append
        if not chunks:
            return np.zeros(0, dtype=np.float32), 0.0
        audio = np.concatenate(chunks)
        total = len(audio)
        if total > n:
            audio = audio[total - n:]
        return audio, (total - len(audio)) / 16000.0

    def _next_speaker_num(self) -> int:
        """下一个可用的全局说话人编号 = 现有标签最大数字后缀 + 1。"""
        best = -1
        for sp in self._store.speakers():
            m = re.match(r"SPEAKER_(\d+)$", sp.label)
            if m:
                best = max(best, int(m.group(1)))
        return best + 1

    def _globalize_spans(self, spans: list[tuple[float, float, str]], base: float
                         ) -> list[tuple[float, float, str]]:
        """把窗口内 (相对时间, 局部标签) 的 spans 换成 (绝对时间, 全局标签)。

        pyannote 的 SPEAKER_xx 只在单次窗口内自洽、跨窗口会换标签，必须用重叠区
        旧全局标签投票映射；映射不到的局部标签视为新说话人，分配新全局标签。
        """
        abs_spans = [(s + base, e + base, lbl) for s, e, lbl in spans]
        prev_labeled = [(seg.start, seg.end, seg.speaker)
                        for seg in self._store.segments()
                        if seg.speaker and seg.start >= base]
        mapping = map_local_labels(prev_labeled, abs_spans, base)
        next_num = self._next_speaker_num()
        for lbl in dict.fromkeys(lbl for _, _, lbl in abs_spans):
            if lbl in mapping:
                continue
            # 只分配、不注册：该标签最终是否真有语句落上，由随后的 assign_speakers 决定。
            # 若在此无条件 register_label，与语句重叠 <30% 的短 turn 会注册出无人使用的
            # 「幽灵说话人」（纪要列出幽灵发言人、编号被顶高）——审查 N1。
            # 真正的注册在 _diarize_worker 末尾遍历「有 speaker 的语句」时完成。
            mapping[lbl] = f"SPEAKER_{next_num:02d}"
            next_num += 1
        return [(s, e, mapping[lbl]) for s, e, lbl in abs_spans]

    def _diarize_worker(self) -> None:
        if not self._diarizer:
            return
        last_n = 0
        while not self._stop.is_set():
            # _stop.wait 代替 time.sleep：stop() 时立即醒来，不必等满一个周期
            if self._stop.wait(DIARIZE_INTERVAL):
                break
            if not self._diarize_audio or len(self._diarize_audio) == last_n:
                continue
            last_n = len(self._diarize_audio)
            audio, base = self._take_diarize_window()
            try:
                spans = self._diarizer.label(audio)
                # label() 成功后的映射 / 分配 / 注册同样不能抛穿：异常会杀死标注线程，
                # 此后整场会议再无说话人标注，且本轮注册全部丢失（审查 Minor-3）。
                mapped_spans = self._globalize_spans(spans, base)
                assign_speakers(self._store.segments(), mapped_spans)
                for seg in self._store.segments():
                    if seg.speaker:
                        self._store.register_label(seg.speaker)
                self._say(f"已识别说话人：{len(self._store.speakers())} 人")
            except Exception as e:
                self._say(f"[警告] 说话人标注失败：{e}")
                continue

    def stop(self) -> None:
        self._stop.set()
        self._capture.stop()
        if self._running.is_set():
            # start() 在调用线程里同步跑主循环，stop() 无法 join 它；
            # 等 producer_done 即可确保尾部 flush 已经把 Segment 投进翻译队列。
            # stop() 早于/独立于 start()（无人在跑主循环）时 _running 为 False，直接跳过。
            self._producer_done.wait(timeout=PRODUCER_DONE_TIMEOUT)
        worker = self._translate_thread
        if worker is not None and worker.is_alive():
            # 只等 queue.empty() 不够：worker 可能已取走最后一条却还没 emit。
            self._worker_done.wait(timeout=TRANSLATE_DRAIN_TIMEOUT)
        if self._translator:
            for seg in self._store.segments():
                if not seg.translation and seg.lang != "zh":
                    try:
                        seg.translation = self._translator.translate(seg.text, seg.lang)
                    except TranslationError:
                        pass
                # 兜底补发：worker 猝死/超时未 emit 的语句在这里补上（审查 N2）
                self._emit_once(seg)

        # 尾部：等说话人标注线程收尾（把手上这一轮窗口跑完即退出）。
        # 当前线程就是标注线程时不 join 自己；未启动（无 diarizer）则跳过。
        diar = self._diarize_thread
        if diar is not None and diar is not threading.current_thread() and diar.is_alive():
            diar.join(timeout=DIARIZE_JOIN_TIMEOUT)
            if diar.is_alive():
                self._say(f"[警告] 说话人标注线程未在 {DIARIZE_JOIN_TIMEOUT:.0f} 秒内退出")
