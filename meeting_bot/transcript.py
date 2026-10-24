# -*- coding: utf-8 -*-
"""会议记录缓冲：语句存储、说话人命名、Markdown 渲染、崩溃恢复日志。"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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
        # 写失败的「最近一次异常」：日志是防丢的安全网，不能反过来把录制搞崩
        self.error: Exception | None = None
        if not self._append:
            # 追加模式不重写 header（沿用原日志开头，避免双 header）
            self._write({"t": HEADER, "title": title,
                         "started_at": (started_at or datetime.now()).isoformat(timespec="seconds")})

    def _write(self, obj: dict) -> None:
        with self._lock:
            if self._closed:
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
    title: str | None = None
    started_at: datetime | None = None
    store = TranscriptStore(journal=Journal(path, append=True)
                            if attach_journal else None)
    # 先把行全部读进来再回放：挂日志（追加模式）会在回放 update/speaker 事件时
    # 往同一文件追加新行，若边读边写会把刚追加的行再读回来 → 无限循环（实测卡死）
    with Path(path).open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    for line in lines:
            ev = _parse_line(line)
            if ev is None:
                continue                      # 坏行 / 空行：跳过继续
            kind = ev.get("t")
            if kind == HEADER:
                title = str(ev.get("title") or "")
                try:
                    started_at = datetime.fromisoformat(str(ev.get("started_at")))
                except (TypeError, ValueError):
                    started_at = None
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
            # footer 只是标记，回放时无需处理
    if title is None:
        raise ValueError(f"日志缺少 header，无法恢复：{path}")
    return title, started_at or datetime.now(), store


class TranscriptStore:
    def __init__(self, journal: Journal | None = None) -> None:
        self._segments: list[Segment] = []
        self._speakers: dict[str, Speaker] = {}
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
                if changed and self._journal is not None:
                    self._journal.record_update(seq, changed)
                return

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
        return sp.name if sp and sp.name else label

    def segments(self) -> list[Segment]:
        return list(self._segments)

    def speakers(self) -> list[Speaker]:
        return list(self._speakers.values())

    def _restore(self, seg: Segment) -> None:
        """回放日志用：按日志里的 seq 原样放回（不重编号，保证后续 update 能命中）。"""
        self._segments.append(seg)
        if seg.seq >= self._next_seq:
            self._next_seq = seg.seq + 1

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
