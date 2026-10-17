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

from .asr import ASREngine, CloudASREngine
from .capture import SOURCE_BOTH, AudioCapture
from .config import ConfigError, Settings
from .diarization import Diarizer, assign_speakers
from .transcript import Segment, TranscriptStore
from .translator import TranslationError, Translator
from .vad import UtteranceBuffer

SegmentEvent = Callable[[Segment], None]
StatusEvent = Callable[[str], None]

DIARIZE_INTERVAL = 60.0       # 每 60 秒跑一次说话人聚类
DIARIZE_WINDOW = 300.0        # 每次只处理最近 5 分钟音频（滑动窗口，避免长会 O(T²) 重算）
DIARIZE_BUFFER_SECONDS = DIARIZE_WINDOW + 60.0  # 滚动缓冲上限 = 窗口 + 一轮余量（防内存无上限增长）
DIARIZE_JOIN_TIMEOUT = 180.0  # stop() 等说话人标注线程收尾的上限
PRODUCER_DONE_TIMEOUT = 10.0  # stop() 等主循环完成尾部 flush 的上限
TRANSLATE_DRAIN_TIMEOUT = 20.0  # stop() 等翻译 worker 排空退出的上限
TRANSLATE_POLL_TIMEOUT = 0.5  # 翻译 worker 队列轮询间隔
BACKFILL_MAX_CONSECUTIVE_FAILS = 3  # stop() 补译连续失败上限（超过则放弃补译，防无界重试）
STREAM_UPDATE_INTERVAL = 0.5  # 流式译文刷新间隔（秒）：0.5s 一次是「均匀打字」的手感，再密
                              # 就是 token 碎片闪屏（用户反馈的拼接感），再疏则像卡住不掉字
CONTEXT_RECENT_SEGMENTS = 4   # 翻译上文参考取最近几条语句（统一术语/人名 + 纠口音错字）
ASR_BATCH_SECONDS = 30.0      # 攒够 30 秒音频触发一批识别
ASR_BATCH_MAX_WAIT = 35.0     # 静默会议兜底：最老语句积压 35 秒墙钟也强制识别
ASR_POLL_TIMEOUT = 0.5        # 识别 worker 轮询间隔（无新语句也要醒来查 max-wait）
ASR_JOIN_TIMEOUT = 180.0      # stop() 等识别线程把收尾批（最后 30s+ 音频）识别完的上限


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
                 diarizer: Diarizer | None = None,
                 audio_source: str = SOURCE_BOTH):
        """audio_source 只在 capture 为 None（自行建捕获）时生效：

        - ``"both"``：扬声器 loopback + 默认麦克风混合（默认，线上/线下通用）
        - ``"speaker"``：只录系统扬声器 loopback（线上会议）
        - ``"mic"``：只录默认麦克风（现场对话）

        显式注入 capture 时该参数被忽略 —— 上层（app.py）会先构造探测过的实例
        再传进来，此处不再重复打开设备。
        """
        self._settings = settings
        self._audio_source = audio_source
        # 云端识别（groq）：每条语句到达即识别（不再攒批），getattr 兜住旧构造的 Settings
        self._asr_provider = (getattr(settings, "asr_provider", "local") or "local").strip().lower()
        # 攒批阈值：.env 里显式配置则以后者为准，否则用模块常量（测试 monkeypatch 常量生效）
        self._asr_batch_seconds = (getattr(settings, "asr_batch_seconds", None)
                                   or ASR_BATCH_SECONDS)
        self._asr_batch_max_wait = (getattr(settings, "asr_batch_max_wait", None)
                                    or ASR_BATCH_MAX_WAIT)
        self._capture = capture or AudioCapture(source=audio_source)
        self._asr = asr
        self._translator = translator
        self._diarizer = diarizer
        self._store = TranscriptStore()
        self._events: list[SegmentEvent] = []
        # 译文填充后的「原地更新」通道（两阶段显示）：与 _events 分离，
        # 因为同一条语句会先发原文（_events 一次），再发译文更新（_updates）。
        self._updates: list[SegmentEvent] = []
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
        # --- 识别批处理（替代逐句识别）---
        # whisper 每次 transcribe 有 ~10s 固定开销（30s 窗口 padding）：逐句识别 RTF 高达
        # 5.6，连续对话（每 5~10s 一句）积压以 2 倍实时增长 → 字幕永远追不上。攒够一批
        # 一次识别后 RTF < 1（34s 音频 19s，RTF 0.55），代价是字幕成批延迟出现。
        self._asr_batch: list[tuple[int, np.ndarray]] = []   # (绝对样本偏移, 语句音频)
        self._asr_batch_first_at = 0.0        # 当前批最老条目的入队时刻（monotonic 秒）
        self._asr_batch_lock = threading.Lock()   # 主循环 append / worker 取走 → 必须原子
        self._asr_batch_event = threading.Event()   # 有积压时唤醒 worker
        self._asr_thread: threading.Thread | None = None
        # 引擎构造失败（缺 GROQ_API_KEY 等配置错误）只告警一次：这是持续性问题，
        # 每句都报会把状态栏刷满、把字幕挤没（云端模式下每句都触发一次识别尝试）
        self._asr_engine_error_said = False
        # 识别线程已收尾（批队列全部产出）：翻译 worker 必须等到它才排空退出，
        # 否则收尾批的语句入队时 worker 已因「生产者完结 + 队列空」退出 → 无人翻译。
        self._asr_finished = threading.Event()
        # 已 emit 的 seq：stop() 兜底补发时去重（store 跨 start 不重置，seq 全局唯一 → 无需 clear）
        self._emitted: set[int] = set()
        # _emit_once 的记账锁：识别线程（原文 emit）与 stop()（兜底补发）会并发处理
        # 同一 seq，「检查 + 记账」必须原子，否则两者都判定「未发过」→ 重复 emit（审查 Minor-1）
        self._emit_lock = threading.Lock()
        self._diarize_audio: list[np.ndarray] = []
        # 已被 _trim_diarize_buffer 裁掉的样本数：滑动窗口的绝对时间原点
        self._diarize_base = 0
        # 已 append 的块数（只增不减）：标注线程用它判断「有无新增音频」。
        # 不能用 len(_diarize_audio) 当判据 —— 稳态下每轮 append 1 块、trim 又丢 1 块，
        # 块数恒定 → 门控从首个稳态 tick 起永远判定「无新音频」→ 说话人标注永久停摆
        # （录制 6-7 分钟、缓冲进入裁剪稳态后开始，最终复审 N-1）。
        self._diarize_appended = 0
        # 缓冲的写（主循环 append + trim）与读（标注线程快照 + base）必须原子：
        # 否则 list() 快照与读 _diarize_base 之间被插队 append+trim，base 会多记本次
        # 被裁样本 → 窗口绝对起点每次后移一个块累积漂移（最终复审 N-2）。
        self._diarize_lock = threading.Lock()
        self._started_at: datetime | None = None
        self.title = ""

    # --- 事件 ---
    def on_segment(self, fn: SegmentEvent) -> None:
        self._events.append(fn)

    def on_update(self, fn: SegmentEvent) -> None:
        """注册「译文已填充」回调（两阶段显示的第二阶段）。

        与 ``on_segment`` 分开：``on_segment`` 每条语句只发一次（原文上屏），
        ``on_update`` 在翻译完成（含 zh 直填、收尾补译）后再发一次，供面板原地替换。
        """
        self._updates.append(fn)

    def on_status(self, fn: StatusEvent) -> None:
        self._status.append(fn)

    def _emit(self, seg: Segment) -> None:
        for fn in self._events:
            fn(seg)

    def _notify_update(self, seg: Segment) -> None:
        """通知译文已填充。回调异常只告警：绝不能杀死翻译 worker / 中断收尾补译。"""
        for fn in self._updates:
            try:
                fn(seg)
            except Exception as e:
                self._say(f"[警告] 译文更新回调异常：{e}")

    def _emit_once(self, seg: Segment) -> None:
        """发送语句事件并记账（每个 seq 最多一次）。

        回调异常只告警：识别线程与 stop() 都不能因消费者异常而中断（审查 N2）。
        去重兜住「原文阶段 emit」与「stop() 兜底补发」的竞态：一条语句必须先出原文
        事件（两阶段显示的第一阶段），译文只走 :meth:`_notify_update`，两条通道互不重叠。

        先记账再发送（at-most-once）：「检查 + 记账」若不加锁，识别线程与 stop() 会并发
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
        self._asr_finished.clear()
        self._running.set()
        self._capture.start()
        self._asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self._asr_thread.start()
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
                with self._diarize_lock:
                    self._diarize_audio.append(chunk)
                    self._diarize_appended += 1
                    self._trim_diarize_buffer()
                for offset, utterance in buf.push(chunk):
                    self._enqueue_asr(offset, utterance)
            for offset, utterance in buf.flush():
                self._enqueue_asr(offset, utterance)
        finally:
            # 采集侧故障对外可见：record() 抛错只会让采集线程退出，主循环收到的是
            # 「正常结束哨兵」，若不检查就表现为「录到一半音频没了」且无任何提示。
            err = getattr(self._capture, "error", None)
            if err is not None:
                self._say(f"[警告] 声音采集中断（后续音频未记录）：{err}")
            dropped = getattr(self._capture, "dropped_chunks", 0)
            if dropped:
                self._say(f"[警告] 处理跟不上实时，已丢弃 {dropped} 块音频（约 {dropped * 0.5:.0f} 秒）")
            # 尾部 flush 结束（或主循环异常）才算生产者完结；顺序不能反：
            # 必须先 set(producer_done) 再 clear(running)，否则 stop() 可能跳过等待。
            self._producer_done.set()
            self._running.clear()
            # TOCTOU 闭环（审查 New-1）：start() 的 _stop 守卫通过之后、_capture.start()
            # 真正打开采集之前，stop() 可能抢先跑完 —— 那一瞬 stream 还没打开，
            # stop() 里的 _capture.stop() 是空操作，此后无人再关采集（loopback 占用到
            # 进程退出）。主循环退出即补一次 _capture.stop()（AudioCapture.stop 幂等：
            # clear + join 已结束的线程，重复调用无副作用）。
            self._capture.stop()

    # --- 识别批处理 ---
    def _enqueue_asr(self, offset: int, utterance: np.ndarray) -> None:
        """把一条 VAD 语句追加进识别批（主循环唯一入口，不阻塞主循环）。"""
        with self._asr_batch_lock:
            if not self._asr_batch:
                # 记「本批最老条目」的入队时刻：max-wait 兜底以它为基准
                self._asr_batch_first_at = time.monotonic()
            self._asr_batch.append((offset, utterance))
        self._asr_batch_event.set()

    def _asr_batch_ready(self, force: bool) -> bool:
        """是否该识别这一批了：够长 / 等够久 / 收尾（force）。"""
        with self._asr_batch_lock:
            if not self._asr_batch:
                return False
            if self._asr_provider == "groq":
                # 云端识别单次延迟低（约 1s），攒批只会白白增加字幕延迟：
                # 有积压就出批（通常恰好 1 条 VAD 语句）。阈值退化为 0 的效果。
                return True
            pending = sum(len(a) for _, a in self._asr_batch)
            waited = time.monotonic() - self._asr_batch_first_at
        if force:
            return True                      # 停止录制：残余全部带走
        return (pending >= int(self._asr_batch_seconds * 16000)
                or waited >= self._asr_batch_max_wait)

    def _take_asr_batch(self) -> list[tuple[int, np.ndarray]]:
        """取走当前全部积压（原子：与主循环 append 互斥）。"""
        with self._asr_batch_lock:
            batch = self._asr_batch
            self._asr_batch = []
            self._asr_batch_first_at = 0.0
        return batch

    def _asr_worker(self) -> None:
        """批处理识别线程：攒批 → 一次 transcribe → 按批内相对时间回填绝对时间轴。"""
        try:
            # 退出条件 = 已收到停止信号且积压已清空（收尾批必须识别完）
            while not (self._stop.is_set() and not self._asr_batch):
                # 事件唤醒（新语句入批）+ 固定轮询（静默会议靠 max-wait 兜底醒来）
                self._asr_batch_event.wait(timeout=ASR_POLL_TIMEOUT)
                self._asr_batch_event.clear()
                if not self._asr_batch_ready(self._stop.is_set()):
                    continue
                self._transcribe_batch(self._take_asr_batch())
        finally:
            # 不论正常退出还是异常，都要放行翻译 worker 的排空条件
            self._asr_finished.set()

    def _transcribe(self, audio: np.ndarray) -> list[Segment]:
        """调识别引擎。

        云端引擎多收一个 should_stop：它的失败退避 sleep 会切片轮询停止信号，
        录制结束时不必干等 1s+2s（本地引擎没有这个参数，用 isinstance 区分，
        不用 try/except TypeError 兜 —— 那会把 transcribe 内部真出现的
        TypeError 一起吞掉，掩盖 bug）。
        """
        if isinstance(self._asr, CloudASREngine):
            return self._asr.transcribe(audio, should_stop=self._stop.is_set)
        return self._asr.transcribe(audio)

    def _transcribe_batch(self, batch: list[tuple[int, np.ndarray]]) -> None:
        if not batch:
            return
        # 按绝对样本偏移排序（顺带修 m2）：拼接顺序、段映射表、批起点三者的
        # 前提都是「批内有序」，乱序会让映射表与音频错位。
        batch = sorted(batch, key=lambda it: it[0])
        # 拼接会抹掉语句之间的静音（10s 静音 = 0 样本），whisper 返回的段起点/终点
        # 因此是「拼接缓冲内的相对时间」；直接拿 min(offset) 当统一 base 会让批内
        # 第 2 条及以后语句的时间系统性偏早（最多提前全部间隙之和）→ 逐字稿时间戳
        # 错误，且 assign_speakers 与 pyannote 的真实时间轴错位（漏标/错标说话人）。
        # 解：记录每条语句在拼接缓冲里的 (起, 止) 与它的绝对样本偏移，
        # 段的相对时间按样本落点回填到所属语句的绝对时间轴上。
        spans: list[tuple[int, int, int]] = []   # (拼接内起, 拼接内止, 语句绝对样本偏移)
        acc = 0
        for off, a in batch:
            spans.append((acc, acc + len(a), off))
            acc += len(a)
        audio = (batch[0][1] if len(batch) == 1
                 else np.concatenate([u for _, u in batch]))
        try:
            if self._asr is None:
                # 惰性创建：重型模型只在首批语句到来时加载（测试注入 fake 时不走此行）。
                # groq 模式走云端 client：没有 500MB 模型下载，也不用等首句识别预热。
                self._asr = (CloudASREngine(self._settings)
                             if self._asr_provider == "groq" else ASREngine(self._settings))
            segments = self._transcribe(audio)
        except Exception as e:
            # 单批失败只丢这一批（告警可见），不得杀掉识别线程 —— 否则整场再无字幕。
            # 云端模式的 CloudASRError（重试后仍失败/结构异常/被停止）与本地异常同路处理；
            # 缺 key 时 CloudASREngine 构造抛 ConfigError 也落在这里（告警里带原因）。
            # CloudASRConfigError 是 ConfigError 的子类（key 无效/无权限），
            # 所以下面这一个分支同时覆盖「缺 key」与「key 无效」两种配置类错误。
            if isinstance(e, ConfigError):
                # 配置错误是持续性的：首次失败后每句都会再失败，只报一次即可，
                # 否则状态栏被同一句警告刷满、字幕反而看不见。
                if self._asr_engine_error_said:
                    return
                self._asr_engine_error_said = True
                self._say(f"[错误] 识别引擎不可用，本场不再有字幕：{e}")
                return
            self._say(f"[警告] 识别失败（本批 {len(batch)} 条语句已丢弃）：{e}")
            return

        def to_abs(t_rel: float, is_end: bool = False) -> float:
            """拼接缓冲内的相对秒 → 会议绝对秒（按语句分段映射回填间隙）。

            区间开闭必须区分起点/终点：语句区间是半开的 [rs, re)，但段终点恰等于某条
            语句的终点时（re_ == s），用半开判定会「跳」到下一条语句的起点 —— 例：两条
            语句偏移 0 和 11s，whisper 返回 (0, 1)，终点 1.0 落在语句 1 的终点边界，
            半开判定把它映射到 11.0 → 该段时长膨胀 11 倍，assign_speakers 的重叠比例
            随之错乱（说话人漏标/错标）。故终点用 (rs, re] 闭右、起点用 [rs, re) 闭左。
            """
            s = int(t_rel * 16000)
            for rs, re_, off in spans:
                hit = (rs < s <= re_) if is_end else (rs <= s < re_)
                if hit:
                    return (off + s - rs) / 16000.0
            i = spans[-1]
            if s >= i[1]:
                # 段落在最后一条语句之后：沿最后一条语句的时间轴等量延伸
                return (i[2] + i[1] - i[0] + s - i[1]) / 16000.0
            # 段落在首条语句之前（s < 0 或浮点误差）：沿首条语句时间轴回退；时间轴
            # 不允许为负（负起点会让 UI 排序/切片异常），夹到 0。
            return max(0.0, (spans[0][2] + s) / 16000.0)

        for s in segments:
            # 单段入库失败只丢这一段（告警可见），不得杀掉识别线程（审查 N3）：
            # store.add / to_abs 的任何意外异常若直穿 worker，整场后续批次永不出库。
            try:
                start = to_abs(s.start)
                end = max(start, to_abs(s.end, is_end=True))   # 时间轴单调，防倒挂
                seg = self._store.add(start, end, s.text, lang=s.lang)
            except Exception as e:
                self._say(f"[警告] 语句入库失败：{e}")
                continue
            # 两阶段显示（第一阶段）：原文立即上屏 —— 此刻 seg.translation 为空，
            # 面板主标签落回原文，不必干等翻译接口的 3~6 秒。译文到达后走
            # _notify_update 原地替换（第二阶段）。
            # 必须走 _emit_once：回调异常若直穿 worker，识别线程当场死亡且无告警，
            # 此后所有批次永不出库（无声「字幕停更」）。_emit_once 内部记账 + 兜异常。
            self._emit_once(seg)
            if self._translator:
                self._translate_queue.put(seg)

    def _recent_context(self, seg: Segment) -> str:
        """取该句之前最近几条语句的「原文/译文」，作为翻译的上文参考。

        为什么读 store 而不在 worker 里另存滚动缓冲：store 是唯一事实源、天然按
        seq 有序，worker 与 stop() 都能复用；``segments()`` 返回副本（``list(...)``），
        与 ``_globalize_spans`` / ``_diarize_worker`` 的既有读法一样是线程安全的。

        优先取「已有译文的语句」：译文里已固化的术语 / 人名才是要统一的锚点；
        前几句都还没译出来时退回全部更早语句 —— 原文本身也能帮模型纠正口音错字。
        只取 ``seq < seg.seq``：绝不能把待译句自己塞进上文（模型会照抄自己）。
        """
        prev = [s for s in self._store.segments() if s.seq < seg.seq]
        if not prev:
            return ""
        translated = [s for s in prev if (s.translation or "").strip()]
        picked = (translated or prev)[-CONTEXT_RECENT_SEGMENTS:]
        return "\n".join(
            f"原文：{s.text}\n译文：{s.translation or '（无）'}" for s in picked)

    def _stream_translate(self, seg: Segment) -> None:
        """流式翻译一条语句：译文逐字上屏，节流刷新，末次必刷。

        为什么要节流：deepseek-v4-pro 流式约 20~60 块/句，每块都推一次 Qt 重绘
        会把事件循环刷满（面板卡顿、丢字）；而按块刷新在视觉上就是「碎片拼接」，
        所以改成固定 0.5 秒节奏（STREAM_UPDATE_INTERVAL）：约每半秒推一次当时
        的累积译文，看起来是匀速打字，而不是 token 一抖一抖地冒。

        为什么「末次必刷」：最后一块往往落在节流窗口内被跳过，不补一次的话面板
        会停在倒数第二块 —— 字幕缺尾巴，且 store 里是全量译文、屏幕上是残句。

        流中断（TranslationError 等）：半句译文可能已经上屏，必须清空并再通知一次，
        让面板退回原文；否则屏幕上留着一段永远补不上的残句（会议纪要里是空译文，
        两边不一致）。

        已知取舍：若流卡死到 stop() 等满 TRANSLATE_DRAIN_TIMEOUT，store 里可能留下
        半句译文（面板看到的也是半句 —— 两边一致）。此时不再补译：补译会对同一个
        已卡住的接口再发一次请求，且有第二个写者覆盖 seg.translation。原文始终在
        store 里，纪要上下文不缺。
        """
        # 上文参考在进入流之前取一次：流式过程中 store 可能新增语句（别的 worker
        # 正在翻译更靠后的句子），中途换上下文会让同一句的术语前后不一致。
        context = self._recent_context(seg)
        last_at = 0.0
        last_text = ""
        try:
            for partial in self._translator.translate_stream(seg.text, seg.lang,
                                                              context=context):
                seg.translation = partial
                now = time.monotonic()
                if now - last_at >= STREAM_UPDATE_INTERVAL:
                    last_at = now
                    last_text = partial
                    self._notify_update(seg)
        except Exception:
            if seg.translation:
                seg.translation = ""
                self._notify_update(seg)   # 回退：面板回到原文显示
            raise
        # 末次必刷：只要屏上还不是最终文本就再推一次（与最后一块通知的时间无关，
        # 尾块被节流挡掉时面板会缺尾巴）；文本已相同说明屏上即最终，无需空重绘。
        if seg.translation and seg.translation != last_text:
            self._notify_update(seg)

    def _translate_worker(self) -> None:
        try:
            # producer-done 模式：只认「生产者完结 + 队列排空」才退出。
            # 若用 _stop 判定，尾段仍在内核 ASR（0.5~3s）时 worker 就会因队列暂空而退出，
            # flush 产出的 Segment 将永远无人翻译、无人 emit。
            # 除 producer-done 外还必须等识别线程收尾：批处理下最末一批语句
            # 在 _producer_done 之后才产出，那一刻队列是空的。
            while not (self._producer_done.is_set()
                       and self._asr_finished.is_set()
                       and self._translate_queue.empty()):
                try:
                    seg = self._translate_queue.get(timeout=TRANSLATE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                try:
                    if seg.lang == "zh":
                        seg.translation = seg.text
                        # 中文段无需翻译：一次通知即完成第二阶段（没有流可节流）
                        if seg.translation:
                            self._notify_update(seg)
                    else:
                        # 非中文段走流式：译文逐字上屏（_stream_translate 内部负责
                        # 节流刷新、末次必刷、流中断时把半句回退成原文）
                        self._stream_translate(seg)
                except TranslationError:
                    seg.translation = ""   # 会议结束后统一补译
                    self._say(f"[警告] 翻译失败（{seg.text[:20]}…），已保留原文")
                except Exception as e:
                    # 意外异常（openai 库边界异常、回调异常等）不得杀死 worker：
                    # 否则后续语句只入库、永不翻译、永不 emit（审查 N2）。
                    seg.translation = ""
                    self._say(f"[警告] 翻译异常（{seg.text[:20]}…）：{e}，已保留原文")
        finally:
            self._worker_done.set()

    def _trim_diarize_buffer(self) -> None:
        """裁剪滚动缓冲：只保留最近 DIARIZE_BUFFER_SECONDS 秒音频。

        长会（1 小时 ≈ 0.5G 样本）不裁剪则内存无上限增长，且每轮 diarize 都要
        np.concatenate 全量音频（O(T²) 拷贝）。裁剪后缓冲恒定，concatenate 也只处理
        窗口量级；被裁掉的样本数记入 _diarize_base，保证窗口起点仍是绝对时间轴。
        """
        total = sum(len(a) for a in self._diarize_audio)
        limit = int(DIARIZE_BUFFER_SECONDS * 16000)
        if total <= limit:
            return
        excess = total - limit
        kept: list[np.ndarray] = []
        for a in self._diarize_audio:
            if excess <= 0:
                kept.append(a)
                continue
            if len(a) <= excess:
                excess -= len(a)
            else:
                kept.append(a[excess:])
                excess = 0
        self._diarize_audio = kept
        self._diarize_base += total - limit

    def _take_diarize_window(self) -> tuple[np.ndarray, float]:
        """取最近 DIARIZE_WINDOW 秒音频，返回 (窗口音频, 窗口起点绝对秒数)。

        长会每次对全量累积音频重跑 pyannote 是 O(T²)；只喂滑动窗口后单次开销恒定。
        """
        n = int(DIARIZE_WINDOW * 16000)
        # 快照与 base 必须同一临界区读取：主循环 append+trim 若插在两者之间，
        # base 会多记本次被裁样本 → 窗口绝对起点后移（N-2）。concatenate 与切片
        # 放锁外（base 已固化，纯本地数据），避免长时间持锁拖住主循环。
        with self._diarize_lock:
            chunks = list(self._diarize_audio)  # 快照：主循环可能正在 append
            base = self._diarize_base
        if not chunks:
            return np.zeros(0, dtype=np.float32), 0.0
        audio = np.concatenate(chunks)
        total = len(audio)
        if total > n:
            audio = audio[total - n:]
        # 绝对起点 = 已被裁掉的样本 + 本次窗口前的样本（_diarize_base 是裁剪记账）
        return audio, (base + (total - len(audio))) / 16000.0

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
            # 判据必须是单调递增的「已 append 块数」：裁剪稳态下块数恒定，
            # 用 len() 比较会让标注从首个稳态 tick 起永久停摆（N-1）。
            if not self._diarize_audio or self._diarize_appended == last_n:
                continue
            last_n = self._diarize_appended
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
        # 识别收尾：此刻尾部 flush 的语句都已入批 → 唤醒 worker 把残余批一次识别完，
        # 再排空翻译队列（顺序不能反：翻译 worker 的退出条件含 _asr_finished）。
        self._asr_batch_event.set()
        asr_thread = self._asr_thread
        if asr_thread is not None and asr_thread is not threading.current_thread():
            if asr_thread.is_alive():
                asr_thread.join(timeout=ASR_JOIN_TIMEOUT)
                if asr_thread.is_alive():
                    self._say(f"[警告] 识别线程未在 {ASR_JOIN_TIMEOUT:.0f} 秒内退出"
                              "（尾部语句可能缺失）")
            # join 超时兜底：翻译 worker 不能因为识别线程卡死而永远等下去
            self._asr_finished.set()
        worker = self._translate_thread
        if worker is not None and worker.is_alive():
            # 只等 queue.empty() 不够：worker 可能已取走最后一条却还没 emit。
            self._worker_done.wait(timeout=TRANSLATE_DRAIN_TIMEOUT)
        if self._translator:
            # 补译免疫（最终审查 Critical-1）：只捕 TranslationError 时，接口返回结构异常
            # （choices=None 等 → TypeError）会直接逃逸出 stop() → 调用方 export_all 整段
            # 被跳过 → 整场会议零产物。改捕 Exception 并降级为告警；同时设连续失败上限，
            # 避免接口全程不可用时逐句重试把收尾 UI 拖到不可预期（无界重试）。
            backfill_fails = 0
            backfill_given_up = False
            for seg in self._store.segments():
                if not backfill_given_up and not seg.translation and seg.lang != "zh":
                    try:
                        seg.translation = self._translator.translate(
                            seg.text, seg.lang, context=self._recent_context(seg))
                        backfill_fails = 0          # 成功即清零：偶发失败不累计
                        # 补译成功 → 面板要拿到最终译文（原文阶段早已上屏）
                        self._notify_update(seg)
                    except Exception as e:
                        backfill_fails += 1
                        self._say(f"[警告] 收尾补译失败：{e}")
                        if backfill_fails >= BACKFILL_MAX_CONSECUTIVE_FAILS:
                            backfill_given_up = True
                            self._say("[警告] 收尾补译连续失败 "
                                      f"{BACKFILL_MAX_CONSECUTIVE_FAILS} 次，放弃本场补译（原文已保留）")
                # 兜底补发：worker 猝死/超时未 emit 的语句在这里补上（审查 N2）。
                # 注意：放弃补译只跳过「重试翻译」，本循环仍对每条语句执行补发，
                # 否则放弃分支会让未 emit 的语句永久丢失（N2 回归）。
                self._emit_once(seg)

        # 尾部：等说话人标注线程收尾（把手上这一轮窗口跑完即退出）。
        # 当前线程就是标注线程时不 join 自己；未启动（无 diarizer）则跳过。
        diar = self._diarize_thread
        if diar is not None and diar is not threading.current_thread() and diar.is_alive():
            diar.join(timeout=DIARIZE_JOIN_TIMEOUT)
            if diar.is_alive():
                self._say(f"[警告] 说话人标注线程未在 {DIARIZE_JOIN_TIMEOUT:.0f} 秒内退出")
