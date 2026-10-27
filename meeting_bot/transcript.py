# -*- coding: utf-8 -*-
"""会议记录缓冲：语句存储、说话人命名、Markdown 渲染、崩溃恢复日志。"""
from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_SPEAKER_RE = re.compile(r"^SPEAKER_(\d+)$")


def pretty_speaker_label(label: str) -> str:
    """SPEAKER_00 → 发言人A，SPEAKER_01 → 发言人B…（超 26 人退回编号）。"""
    m = _SPEAKER_RE.match(label or "")
    if not m:
        return label
    idx = int(m.group(1))
    if 0 <= idx < 26:
        return f"发言人{chr(ord('A') + idx)}"
    return f"发言人{idx + 1}"


MERGE_GAP_SECONDS = 0.8      # 未标注时：停顿 < 0.8s 视为同一段话
MAX_BLOCK_CHARS = 320        # 单个发言段最多约 320 字（超了再起一段，避免一大坨）

_END_PUNCT = "。！？；：，、….!?;:,"
_CJK_CHAR = re.compile(r"[一-鿿]")


def _last_word_char(text: str) -> str:
    """取末尾最后一个「字/字母/数字」（跳过标点空白），用来判断中文还是英文语境。"""
    for ch in reversed(text):
        if ch.isalnum():
            return ch
    return ""


def join_sentences(parts: list[str], *, sep_empty: str = "，") -> str:
    """把同一发言人的多句拼成一段（中文补「，」，英文补空格，避免单词/句子粘连）。"""
    out = ""
    for raw in parts:
        piece = (raw or "").strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        prev_char = out[-1]
        next_char = piece[0]
        cjk_context = bool(_CJK_CHAR.match(_last_word_char(out))
                           or _CJK_CHAR.match(next_char))
        if prev_char in "。！？；：，、…":
            out += piece                      # 中文句末/句中标点：直接相接
        elif prev_char in ".!?;:,":
            out += ("" if cjk_context else " ") + piece
        elif cjk_context:
            out += sep_empty + piece
        else:
            out += " " + piece
    return out


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


# ---------------------------------------------------------------------------
# 崩溃恢复：语句变更的追加式 JSONL 日志
# ---------------------------------------------------------------------------

HEADER = "header"
FOOTER = "footer"


def _write_jsonl_line(fp, obj: dict) -> None:
    """写一行 JSON 并立刻 flush（崩溃时最多丢最后一行）。"""
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def _parse_line(line: str) -> dict | None:
    """解析一行；坏行（崩溃时写了半行、编码异常）返回 None 由调用方跳过。"""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


class Journal:
    """把语句变更追加写入 JSONL，用于进程崩溃后的恢复。

    为什么是追加式日志而不是「每句重写整个 JSON」：会议中途崩溃时最后一条 JSON
    是残缺的，整文件读写会让此前的全部内容一起失效；逐行追加则只有最后一行作废，
    回放时跳过坏行即可（见 :func:`load_journal`）。

    事件格式（每行一条 JSON，``ensure_ascii=False`` 便于人工排障）::

        {"t":"header","title":str,"started_at":iso}   构造时写一次
        {"t":"add","seq","start","end","text","lang","speaker","translation"}
        {"t":"update","seq",...只写改动的字段}
        {"t":"speaker","label","name"}
        {"t":"voice","label","vector"}                该声纹的声纹向量（认人用）
        {"t":"footer"}                                正常结束（已导出）

    ``footer`` 是「本场已完整落盘」的标记：下次启动扫描工作区时，没有 footer 的
    文件才算「上次会议未正常结束」，才需要询问用户是否恢复。
    """

    def __init__(self, path: Path | str, title: str = "",
                 started_at: datetime | None = None, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()     # 识别/翻译/标注/UI 四个线程都会写
        self._append = bool(append) and self.path.is_file()
        self._fp = self.path.open("a" if self._append else "w",
                                  encoding="utf-8", newline="\n")
        self._closed = False
        self._finished = False
        # 回放（load_journal）期间挂起写入：否则「读日志 → 重建 store」会把回放
        # 动作（update / speaker）当新事件再追加回去，同一条日志每开一次翻一倍
        # （实测撑到 2.5 GB）
        self._suspend_depth = 0
        # 写失败的「最近一次异常」：日志是防丢的安全网，不能反过来把录制搞崩
        self.error: Exception | None = None
        if not self._append:
            # 追加模式不重写 header（沿用原日志开头，避免双 header）
            self._write({"t": HEADER, "title": title,
                         "started_at": (started_at or datetime.now()).isoformat(timespec="seconds")})

    def _write(self, obj: dict) -> None:
        with self._lock:
            if self._closed or self._suspend_depth > 0:
                return
            try:
                _write_jsonl_line(self._fp, obj)
            except (OSError, ValueError) as e:
                # 磁盘满 / 只读 / 关闭竞态：只记一次，录制照常继续
                if self.error is None:
                    self.error = e

    def record_add(self, seg: Segment) -> None:
        self._write({"t": "add", "seq": seg.seq, "start": seg.start, "end": seg.end,
                     "text": seg.text, "lang": seg.lang, "speaker": seg.speaker,
                     "translation": seg.translation})

    def record_update(self, seq: int, fields: dict) -> None:
        # 只写调用方显式声明的字段（update 的语义就是「改动了这些」），
        # 回放时按序覆盖即可，不必重写整条语句
        self._write({"t": "update", "seq": seq, **fields})

    def record_speaker(self, label: str, name: str) -> None:
        self._write({"t": "speaker", "label": label, "name": name})

    def record_voice(self, label: str, vector) -> None:
        """记一条声纹向量（每场每人一条，认人/学习用；写失败也不影响主流程）。"""
        try:
            values = [round(float(x), 6) for x in vector]
        except (TypeError, ValueError):
            return
        if not label or not values:
            return
        self._write({"t": "voice", "label": str(label), "vector": values})

    @contextmanager
    def suspended(self):
        """上下文内不写盘（回放重建 store 时用）。可嵌套。"""
        with self._lock:
            self._suspend_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._suspend_depth = max(0, self._suspend_depth - 1)

    def footer(self) -> None:
        """标记本场已完整落盘/导出（幂等：重复调用只写一次）。"""
        if self._finished:
            return
        self._finished = True
        self._write({"t": FOOTER})

    @property
    def finished(self) -> bool:
        return self._finished

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fp.close()
            except OSError:
                pass

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def journal_meta(path: Path | str) -> tuple[str, datetime | None]:
    """只读 header：给「是否恢复上次会议」提示框取标题与开始时间。

    不必回放整个日志（大会几百 KB）；读不到 header 时返回 ``("", None)``。
    """
    try:
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                ev = _parse_line(line)
                if ev and ev.get("t") == HEADER:
                    title = str(ev.get("title") or "")
                    try:
                        started_at = datetime.fromisoformat(str(ev.get("started_at")))
                    except (TypeError, ValueError):
                        started_at = None
                    return title, started_at
    except OSError:
        pass
    return "", None


def journal_finished(path: Path | str) -> bool:
    """该日志是否已含 footer（已完整落盘/导出）。文件不存在/读不到 → False。"""
    try:
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                ev = _parse_line(line)
                if ev and ev.get("t") == FOOTER:
                    return True
    except OSError:
        return False
    return False


def write_footer(path: Path | str) -> None:
    """给已有日志追加 footer（恢复流程里没有 Journal 实例，直接追加一行）。"""
    with Path(path).open("a", encoding="utf-8", newline="\n") as fp:
        _write_jsonl_line(fp, {"t": FOOTER})


def load_journal(path: Path | str, attach_journal: bool = False) ->         tuple[str, datetime, "TranscriptStore"]:
    """回放日志重建 store，返回 (标题, 开始时间, store)。

    - 坏行逐行跳过（崩溃时半行 JSON 是预期内的）；
    - 缺 header 抛 ``ValueError``（不是本程序写的日志 / 文件被截断到开头都没有）；
    - ``attach_journal=False`` 时不挂 journal（避免把回放动作再写一遍）；
      ``attach_journal=True`` 时以追加模式挂回原日志——之后的改名等事件会写回
      该文件（用于「对历史会议补标记发言人」）。
    """
    store = TranscriptStore(journal=Journal(path, append=True)
                            if attach_journal else None)
    # 先把行全部读进来再回放：挂日志（追加模式）会在回放 update/speaker 事件时
    # 往同一文件追加新行，若边读边写会把刚追加的行再读回来 → 无限循环（实测卡死）
    with Path(path).open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    # 回放期间禁止写盘：否则 set_speaker_name / update 会把回放动作再追加回日志，
    # 同一条日志每打开一次就翻一倍（曾把一场会议撑到 2.5 GB）
    journal = store.journal
    ctx = journal.suspended() if journal is not None else _null_ctx()
    with ctx:
        _replay_lines(lines, store)
    title = store.session_title
    if not title:
        raise ValueError(f"日志缺少 header，无法恢复：{path}")
    return title, _started_of(lines) or datetime.now(), store


def _null_ctx():
    """空上下文（没有挂 journal 时用，避免分支写两份回放代码）。"""
    from contextlib import nullcontext
    return nullcontext()


def _started_of(lines) -> datetime | None:
    for line in lines:
        ev = _parse_line(line)
        if ev and ev.get("t") == HEADER:
            try:
                return datetime.fromisoformat(str(ev.get("started_at")))
            except (TypeError, ValueError):
                return None
    return None


def _replay_lines(lines, store) -> None:
    for line in lines:
            ev = _parse_line(line)
            if ev is None:
                continue                      # 坏行 / 空行：跳过继续
            kind = ev.get("t")
            if kind == HEADER:
                store.session_title = str(ev.get("title") or "")
            elif kind == "add":
                store._restore(Segment(
                    start=float(ev.get("start") or 0.0),
                    end=float(ev.get("end") or 0.0),
                    text=str(ev.get("text") or ""),
                    lang=str(ev.get("lang") or ""),
                    speaker=str(ev.get("speaker") or ""),
                    translation=str(ev.get("translation") or ""),
                    seq=int(ev.get("seq") or 0)))
            elif kind == "update":
                fields = {k: v for k, v in ev.items() if k not in ("t", "seq")}
                store.update(int(ev.get("seq") or 0), **fields)
            elif kind == "speaker":
                store.set_speaker_name(str(ev.get("label") or ""),
                                       str(ev.get("name") or ""))
            elif kind == "voice":
                label = str(ev.get("label") or "")
                try:
                    vector = [float(x) for x in (ev.get("vector") or [])]
                except (TypeError, ValueError):
                    vector = []
                if label and vector:
                    store.voice_vectors[label] = vector
            # footer 只是标记，回放时无需处理


def _block_chars(segs: list[Segment]) -> int:
    return sum(len((s.text or "") + (s.translation or "")) for s in segs)


def _flush_turn(lines: list[str], segs: list[Segment], store) -> None:
    """输出一个发言回合（Teams 转录样式）：

    **发言人A**　[00:01:31]
    原文每句一行…

    > 译文每句一行…
    """
    label = segs[0].speaker or ""
    speaker = store.display_name(label)
    lines.append(f"**{speaker}**　{fmt_ts(segs[0].start)}")
    lines.append("")
    # 原文：每句单独一行（保留说话节奏，不糊成一大段）
    for seg in segs:
        text = (seg.text or "").strip()
        if text:
            lines.append(text)
    lines.append("")
    # 译文：每句单独一行，引用块与原文区分；纯中文发言不重复显示
    translated = [(seg.translation or "").strip() for seg in segs
                  if (seg.translation or "").strip()
                  and (seg.translation or "").strip() != (seg.text or "").strip()]
    if translated:
        for t in translated:
            lines.append(f"> {t}")
        lines.append("")


class TranscriptStore:
    def __init__(self, journal: Journal | None = None) -> None:
        self._segments: list[Segment] = []
        self._speakers: dict[str, Speaker] = {}
        # 本场每个声纹标签的声纹向量（label → 向量）：用于「按声纹认人 / 学习」
        self.voice_vectors: dict[str, list[float]] = {}
        # 回放时 header 里的会议标题（load_journal 取它做返回值）
        self.session_title: str = ""
        self._next_seq = 1
        # 有日志则每次变更同步落盘：进程被杀/断电后可经 load_journal 重建
        self._journal = journal

    @property
    def journal(self) -> Journal | None:
        return self._journal

    def add(self, start: float, end: float, text: str, lang: str = "") -> Segment:
        seg = Segment(start=start, end=end, text=text, lang=lang, seq=self._next_seq)
        self._next_seq += 1
        self._segments.append(seg)
        if self._journal is not None:
            self._journal.record_add(seg)
        return seg

    def update(self, seq: int, **fields) -> None:
        for seg in self._segments:
            if seg.seq == seq:
                changed = {}
                for k, v in fields.items():
                    if hasattr(seg, k):
                        setattr(seg, k, v)
                        changed[k] = v
                if changed.get("speaker"):
                    # 标注结果也会进 speakers()：历史会议回放后只有 speaker 字段，
                    # 不登记的话声纹面板 / 声纹比对全都看不到这些人
                    self.register_label(str(changed["speaker"]))
                if changed and self._journal is not None:
                    self._journal.record_update(seq, changed)
                return

    def record_voice_vector(self, label: str, vector) -> bool:
        """登记一个声纹标签的向量（写内存 + 日志；同名只记一次，避免重复写盘）。"""
        try:
            values = [round(float(x), 6) for x in vector]
        except (TypeError, ValueError):
            return False
        if not label or not values:
            return False
        self.voice_vectors[str(label)] = values
        if self._journal is not None:
            self._journal.record_voice(str(label), values)
        return True

    def register_label(self, label: str) -> None:
        # 不写日志：标签可由 add 事件里的 speaker 字段推导，多记一行纯属冗余。
        # 用 setdefault 而不是「先查后写」：标注线程（recorder/app）也会注册标签，
        # 与 GUI 线程的改名并发时，查-写两步之间可能被插入一个空 Speaker，
        # 把用户刚填的名字覆盖掉。
        if label:
            self._speakers.setdefault(label, Speaker(label=label))

    def set_speaker_name(self, label: str, name: str) -> None:
        # setdefault 同上：改名与「标签首次出现」并发时不能把名字弄丢
        speaker = self._speakers.setdefault(label, Speaker(label=label))
        cleaned = name.strip()
        speaker.name = cleaned
        if self._journal is not None:
            self._journal.record_speaker(label, cleaned)

    def display_name(self, label: str) -> str:
        if not label:
            return "发言人?"
        sp = self._speakers.get(label)
        if sp and sp.name:
            return sp.name
        return pretty_speaker_label(label)

    def segments(self) -> list[Segment]:
        return list(self._segments)

    def speakers(self) -> list[Speaker]:
        return list(self._speakers.values())

    def _restore(self, seg: Segment) -> None:
        """回放日志用：按日志里的 seq 原样放回（不重编号，保证后续 update 能命中）。"""
        self._segments.append(seg)
        if seg.speaker:
            self.register_label(seg.speaker)   # 让 speakers() 认得日志里带的声纹标签
        if seg.seq >= self._next_seq:
            self._next_seq = seg.seq + 1

    def to_markdown(self, title: str, started_at: datetime) -> str:
        """逐字稿：按「发言人回合」分段（同一个人连续说的合并成一段）。"""
        lines = [f"# 会议逐字稿：{title}", "",
                 f"> 会议时间：{started_at:%Y-%m-%d %H:%M}", ""]

        # ---- 发言人汇总（谁说了多久、多少句）----
        turns = self._turns()
        lines += ["## 发言人", ""]
        summary: dict[str, dict] = {}
        for label, segs in turns:
            if not label:
                continue
            info = summary.setdefault(label, {"n": 0, "first": segs[0].start,
                                              "last": segs[-1].end})
            info["n"] += len(segs)
            info["last"] = max(info["last"], segs[-1].end)
        for label, info in summary.items():
            span = max(0, int(info["last"] - info["first"]))
            dur = f"{span // 60} 分 {span % 60} 秒" if span >= 60 else f"{span} 秒"
            lines.append(f"- {self.display_name(label)}：{info['n']} 句，"
                         f"跨度 {dur}（首次发言 {fmt_ts(info['first'])}）")
        if len(lines) and lines[-1] == "":
            lines.pop()
        lines += ["", "## 正文", ""]

        # ---- 正文：同类发言人合并成段 ----
        buf: list[Segment] = []
        for seg in self._segments:
            if not buf:
                buf = [seg]
                continue
            same_speaker = buf[-1].speaker == seg.speaker and bool(seg.speaker)
            # 未标注（发言人?）时也按「停顿很短」合并成段：避免一句一行太碎
            close_enough = (seg.start - buf[-1].end) <= MERGE_GAP_SECONDS
            if same_speaker or close_enough:
                buf.append(seg)
            else:
                _flush_turn(lines, buf, self)
                buf = [seg]
        if buf:
            _flush_turn(lines, buf, self)
        return chr(10).join(lines)

    def _turns(self) -> list[tuple[str, list[Segment]]]:
        """把语句按「同一发言人连续出现」切成回合（'' 视为未标注，独立成段）。"""
        turns: list[tuple[str, list[Segment]]] = []
        for seg in self._segments:
            label = seg.speaker or ""
            if turns and turns[-1][0] == label:
                turns[-1][1].append(seg)
            else:
                turns.append((label, [seg]))
        return turns
