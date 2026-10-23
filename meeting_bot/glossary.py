# -*- coding: utf-8 -*-
"""术语库与纠错表：语音识别的"自动学习"层（全部本地，可查看可清空）。

- ``GlossaryLibrary``：从历史逐字稿里自动挖掘领域术语（产品/客户/行业词），
  下次开会前把高频+近期的术语注入识别提示词（whisper prompt），提升专名识别率。
- ``CorrectionTable``：从「原始识别 vs 润色后」的差异里学习纠错对（如
  识别成"全不能跳过" → 润色成"这个不能跳过"），出现多次后成为确定规则，
  下次识别出该错词时零成本直接替换。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 术语库
# ---------------------------------------------------------------------------

@dataclass
class TermEntry:
    term: str
    count: int = 0
    updated: str = ""          # ISO 时间，最近一次出现/确认时间


class GlossaryLibrary:
    """本地术语库（<项目根>/术语库.json，gitignore）。

    术语来源：① 用户手写（ASR_PROMPT）② 会后 LLM 自动挖掘 ③ 可编程 add。
    """

    def __init__(self, path: Path | None = None):
        from .config import PROJECT_ROOT
        self.path = Path(path) if path else PROJECT_ROOT / "术语库.json"
        self._data: dict[str, dict] = {}
        self.last_error: str | None = None
        self._load()

    # ---- IO ----
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = raw.get("terms", {}) if isinstance(raw, dict) else {}
        except (OSError, ValueError) as e:
            self.last_error = str(e)
            self._data = {}

    def save(self) -> bool:
        try:
            self.path.write_text(
                json.dumps({"version": 1, "terms": self._data},
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
            return True
        except OSError as e:
            self.last_error = str(e)
            return False

    # ---- 操作 ----
    def add(self, term: str, *, count: int = 1, when: str = "") -> None:
        """登记术语；重复登记累计计数。空白/过短/纯标点直接忽略。"""
        from datetime import datetime
        term = _norm(term)
        if len(term) < 2:
            return                      # 单字词噪声多，不进术语库
        entry = self._data.setdefault(
            term, {"count": 0, "updated": when or datetime.now().isoformat(timespec="seconds")})
        entry["count"] += max(1, int(count))
        entry["updated"] = when or datetime.now().isoformat(timespec="seconds")

    def add_many(self, terms: list[str], *, when: str = "") -> None:
        for t in terms:
            self.add(t, when=when)

    def remove(self, term: str) -> None:
        self._data.pop(_norm(term), None)

    def terms(self) -> list[TermEntry]:
        return [TermEntry(t, d.get("count", 0), d.get("updated", ""))
                for t, d in sorted(self._data.items(),
                                   key=lambda kv: (kv[1].get("count", 0),
                                                   kv[1].get("updated", "")),
                                   reverse=True)]

    def top(self, n: int = 30) -> list[str]:
        """高频 + 近期优先的前 n 个术语（拼 ASR prompt 用）。"""
        return [e.term for e in self.terms()[:max(0, int(n))]]

    def clear(self) -> None:
        self._data = {}
        self.save()

    def size(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# 纠错表
# ---------------------------------------------------------------------------

@dataclass
class Correction:
    wrong: str
    right: str
    count: int = 0


class CorrectionTable:
    """本地纠错表（<项目根>/纠错表.json，gitignore）。

    学习规则：从「原始识别 vs 润色后」的文本差异里抽候选对；**同一对出现
    ≥ MIN_CONFIRM 次**才生效（防单次误改污染）。生效后对识别原文做确定性
    字符串替换（零成本零延迟）。
    """

    MIN_CONFIRM = 2

    def __init__(self, path: Path | None = None):
        from .config import PROJECT_ROOT
        self.path = Path(path) if path else PROJECT_ROOT / "纠错表.json"
        self._pairs: dict[str, dict] = {}
        self.last_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._pairs = raw.get("pairs", {}) if isinstance(raw, dict) else {}
        except (OSError, ValueError) as e:
            self.last_error = str(e)
            self._pairs = {}

    def save(self) -> bool:
        try:
            self.path.write_text(
                json.dumps({"version": 1, "pairs": self._pairs},
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
            return True
        except OSError as e:
            self.last_error = str(e)
            return False

    def learn(self, wrong: str, right: str, *, count: int = 1) -> None:
        wrong, right = _norm(wrong), _norm(right)
        if not wrong or not right or wrong == right:
            return
        if len(wrong) > 24 or len(right) > 24:
            return                      # 太长的片段不是"词级纠错"，不学
        entry = self._pairs.setdefault(wrong, {"right": right, "count": 0})
        entry["count"] += max(1, int(count))

    def learn_from_diff(self, raw: str, polished: str) -> int:
        """从一段「识别原文 vs 润色后」抽取词级纠错对。返回学到的对数。"""
        pairs = extract_diff_pairs(raw, polished)
        for w, r in pairs:
            self.learn(w, r)
        return len(pairs)

    def active(self) -> dict[str, str]:
        """已确认（出现 ≥ MIN_CONFIRM 次）的纠错规则。"""
        return {w: d["right"] for w, d in self._pairs.items()
                if d.get("count", 0) >= self.MIN_CONFIRM}

    def apply(self, text: str) -> str:
        """对识别原文应用全部已确认纠错（按错词长度降序，避免子串互相覆盖）。"""
        rules = sorted(self.active().items(), key=lambda kv: -len(kv[0]))
        out = text
        for wrong, right in rules:
            out = out.replace(wrong, right)
        return out

    def pairs(self) -> list[Correction]:
        return [Correction(w, d.get("right", ""), d.get("count", 0))
                for w, d in sorted(self._pairs.items(),
                                   key=lambda kv: -kv[1].get("count", 0))]

    def clear(self) -> None:
        self._pairs = {}
        self.save()


# ---------------------------------------------------------------------------
# 差异抽取（纯函数，可单测）
# ---------------------------------------------------------------------------

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_NUM = re.compile(r"[A-Za-z][A-Za-z0-9.+-]{1,23}")
_STOP = {"啊", "哦", "嗯", "呃", "那", "这", "了", "的", "是", "在", "就", "都",
         "和", "与", "一个", "这个", "那个"}


def _norm(t: str) -> str:
    return (t or "").strip().strip("，。！？、；：,.!?;: \t")


def extract_diff_pairs(raw: str, polished: str) -> list[tuple[str, str]]:
    """提取「词级」纠错候选：只保留 2~12 字的中文词 / 英文词（长度差异 ≤ 6）。

    这是启发式抽取：真正生效还要靠 CorrectionTable 的 ≥2 次确认，所以允许
    这里适度多抽。
    """
    pairs: list[tuple[str, str]] = []
    if not raw or not polished:
        return pairs
    raw_tok = set(_CJK.findall(raw) + _LATIN_NUM.findall(raw))
    pol_tok = set(_CJK.findall(polished) + _LATIN_NUM.findall(polished))
    for w in raw_tok:
        w = _norm(w)
        if not (2 <= len(w) <= 12) or w in _STOP:
            continue
        for r in pol_tok:
            r = _norm(r)
            if w == r or not (2 <= len(r) <= 12):
                continue
            if abs(len(w) - len(r)) > 6:
                continue
            pairs.append((w, r))
    return pairs
