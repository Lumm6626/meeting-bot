# -*- coding: utf-8 -*-
"""本地声纹库：记住「谁的声音」，会后自动认人；用户改名即刻学习。

为什么是「向量池 + 最大相似度」而不是训练分类器：

- 一场会议只给每人一条声纹向量，样本量根本不足以训练任何模型，但足够做
  「最近邻 + 阈值」匹配；
- 同一个人的音色会随设备 / 房间 / 状态漂移，所以每人保留多条历史向量、
  取与 query 最相似的那条作为该人得分（max-pooling），比取平均更抗设备差异；
- 用户改名是最强的标注信号：改名即入库（正样本）；把「自动匹配错的人」改成
  别人时剔除误归属样本（负样本修正），库才会越用越准。

隐私：声纹属生物特征，只存本机 ``声纹库.json``（已 git 忽略），不随会议记录
导出、不上传任何服务。删掉该文件 = 全部遗忘（下次需要重新命名）。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

VOICEPRINT_FILENAME = "声纹库.json"
FORMAT_VERSION = 1
# 每人保留的历史样本上限：再多只会拖慢匹配、放大「陈年设备差异」，收益递减
MAX_VECTORS = 20
DEFAULT_THRESHOLD = 0.75
# 落盘时保留的小数位：1e-6 对余弦相似度的影响远小于阈值粒度，文件却能小一半
VECTOR_DECIMALS = 6

# 声纹模型与「每段 / 每人」的取音上限（见 compute_speaker_embeddings）
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SEGMENT_SECONDS = 20.0
TOTAL_SECONDS = 60.0

def cosine(a, b) -> float:
    """余弦相似度（纯函数）。零向量 / 维度不一致 / 非法输入 → 0.0。"""
    try:
        va = np.asarray(a, dtype=np.float64).ravel()
        vb = np.asarray(b, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return 0.0
    if va.size == 0 or va.size != vb.size:
        return 0.0
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 0.0 or nb <= 0.0:
        return 0.0
    score = float(np.dot(va, vb) / (na * nb))
    # 浮点误差可能给出 1.0000000000000002：截到 [-1, 1]，别让「相似度 > 1」吓到人
    return max(-1.0, min(1.0, score))

def _clean_vector(vector) -> list[float] | None:
    """规整成「有限浮点列表」；空 / 非数 / 全零（无法算夹角）→ None。"""
    try:
        arr = np.asarray(vector, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if float(np.linalg.norm(arr)) <= 0.0:
        return None
    return [round(float(x), VECTOR_DECIMALS) for x in arr]

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

class VoiceprintLibrary:
    """本地声纹池：``{姓名: {"vectors": [[...]], "updated": iso, "meetings": n}}``。

    所有方法都**不抛异常**：IO / 数据损坏只记到 ``self.last_error`` 后安全返回。
    声纹是「让产品更聪明」的增强项，绝不能因为它把录制、导出、改名这些主流程搞崩。
    内存变更（add / penalize / remove / rename）不自动落盘，由调用方在合适的时机
    :meth:`save`（改名时一次、会后认人时一次，避免高频写盘）。
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else PROJECT_ROOT / VOICEPRINT_FILENAME
        self.last_error: Exception | None = None
        self._lock = threading.RLock()      # GUI 线程（改名）与标注线程（会后入库）都会碰
        self._people: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------ 读
    def _load(self) -> None:
        """读库：文件不存在 = 空库（正常）；损坏 = 空库 + last_error（不删原文件）。"""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as e:
            self.last_error = e
            return
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            self.last_error = e
            return
        people = data.get("people") if isinstance(data, dict) else None
        if not isinstance(people, dict):
            return
        with self._lock:
            for name, entry in people.items():
                cleaned = _clean_entry(name, entry)
                if cleaned is not None:
                    self._people[str(name).strip()] = cleaned

    def people(self) -> list[str]:
        """已认识的姓名（按字典序，便于 UI / 断言稳定）。"""
        with self._lock:
            return sorted(self._people)

    def vectors_of(self, name: str) -> list[np.ndarray]:
        """某人的历史向量快照（float64 数组；不存在 → 空列表）。"""
        with self._lock:
            entry = self._people.get((name or "").strip())
            if not entry:
                return []
            return [np.asarray(v, dtype=np.float64) for v in entry["vectors"]]

    def meetings_of(self, name: str) -> int:
        """某人累计被登记的次数（粗略「见过几场」）。"""
        with self._lock:
            return int((self._people.get((name or "").strip()) or {}).get("meetings") or 0)

    # ------------------------------------------------------------ 写（内存）
    def add(self, name: str, vector) -> bool:
        """登记一条声纹样本（每人最多 20 条，超出丢最旧的）。"""
        key = (name or "").strip()
        if not key:
            return False                    # 没名字 = 没人可记
        clean = _clean_vector(vector)
        if clean is None:
            return False
        with self._lock:
            entry = self._people.setdefault(
                key, {"vectors": [], "updated": "", "meetings": 0})
            vectors = entry["vectors"]
            vectors.append(clean)
            if len(vectors) > MAX_VECTORS:
                del vectors[: len(vectors) - MAX_VECTORS]
            entry["updated"] = _now_iso()
            entry["meetings"] = int(entry.get("meetings") or 0) + 1
        return True

    def match(self, vector, threshold: float = DEFAULT_THRESHOLD) -> list[tuple[str, float]]:
        """返回 ``[(姓名, 分数), ...]``：每人取「最像的一条」得分，≥ threshold 才保留，按分数降序。"""
        query = _clean_vector(vector)
        if query is None:
            return []
        try:
            limit = float(threshold)
        except (TypeError, ValueError):
            limit = DEFAULT_THRESHOLD
        with self._lock:
            snapshot = [(name, list(entry["vectors"]))
                        for name, entry in self._people.items()]
        hits: list[tuple[str, float]] = []
        for name, vectors in snapshot:
            score = max((cosine(query, v) for v in vectors), default=-1.0)
            if score >= limit:
                hits.append((name, score))
        # 分数降序；同分按姓名排，保证结果稳定（测试 / UI 都不会随机换顺序）
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits

    def penalize(self, name: str, vector, tol: float = 0.15) -> bool:
        """剔除一条「误归属」样本：删掉该人名下与 vector **最相似的那一条**。

        用于「自动匹配认错人、用户改成别人」的纠正：留着那条样本，下次还会认错。

        ``tol`` 是「毫不相干」的下限：最相似度连 tol 都不到，说明这条向量与该人
        档案里的任何样本都没有关系（多半是不同人 / 坏数据），此时不动库 —— 盲目
        删除只会把真样本删掉、让库越学越薄。
        """
        key = (name or "").strip()
        query = _clean_vector(vector)
        if not key or query is None:
            return False
        try:
            floor = float(tol)
        except (TypeError, ValueError):
            floor = 0.15
        with self._lock:
            entry = self._people.get(key)
            if not entry or not entry["vectors"]:
                return False
            scores = [cosine(query, v) for v in entry["vectors"]]
            best = max(range(len(scores)), key=lambda i: scores[i])
            if scores[best] < floor:
                return False
            del entry["vectors"][best]
            entry["updated"] = _now_iso()
            if not entry["vectors"]:
                self._people.pop(key, None)   # 删空 = 这个人从库里消失（不留空壳）
        return True

    def remove(self, name: str) -> bool:
        """彻底忘掉一个人（返回是否真的删掉了）。"""
        key = (name or "").strip()
        with self._lock:
            return self._people.pop(key, None) is not None

    def rename(self, old: str, new: str) -> bool:
        """把 old 的档案改名成 new；new 已存在则合并向量（取最近 20 条）。"""
        src = (old or "").strip()
        dst = (new or "").strip()
        if not src or not dst or src == dst:
            return False
        with self._lock:
            entry = self._people.pop(src, None)
            if entry is None:
                return False
            existing = self._people.get(dst)
            if existing is None:
                self._people[dst] = entry
            else:
                merged = (existing["vectors"] + entry["vectors"])[-MAX_VECTORS:]
                existing["vectors"] = merged
                existing["updated"] = _now_iso()
                # 合并只为了不丢样本，累计场次取「较多的一边」而不是相加
                existing["meetings"] = max(int(existing.get("meetings") or 0),
                                           int(entry.get("meetings") or 0))
        return True

    # ------------------------------------------------------------ 落盘
    def save(self) -> bool:
        """写 JSON（UTF-8、缩进 1、先写临时文件再原子替换）；失败记 last_error 返回 False。"""
        with self._lock:
            data = {"version": FORMAT_VERSION,
                    "people": {name: {"vectors": [list(v) for v in entry["vectors"]],
                                      "updated": entry["updated"],
                                      "meetings": entry["meetings"]}
                               for name, entry in self._people.items()}}
        tmp = self.path.parent / (self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self.path)      # 原子替换：崩在写一半也不会留下坏库
        except (OSError, TypeError, ValueError) as e:
            self.last_error = e
            try:
                tmp.unlink()                # 清掉半截临时文件（清理失败也不影响结论）
            except OSError:
                pass
            return False
        self.last_error = None
        return True

def _clean_entry(name, entry) -> dict | None:
    """校验 / 归一化一条档案；无效（无名 / 无向量）→ None（丢弃，避免 match 白算）。"""
    if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
        return None
    raw = entry.get("vectors")
    vectors: list[list[float]] = []
    if isinstance(raw, list):
        for item in raw[-MAX_VECTORS:]:
            vector = _clean_vector(item)
            if vector is not None:
                vectors.append(vector)
    if not vectors:
        return None
    try:
        meetings = max(0, int(entry.get("meetings")))
    except (TypeError, ValueError):
        meetings = 0
    return {"vectors": vectors, "updated": str(entry.get("updated") or ""),
            "meetings": meetings}

# ---------------------------------------------------------------------------
# 会后声纹向量（pyannote wespeaker）
# ---------------------------------------------------------------------------

def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)

def _plan_segments(spans) -> dict[str, list[tuple[float, float]]]:
    """按 label 归并语音段，并截断到「每段 ≤20s、每人累计 ≤60s」。

    截断是必须的：wespeaker 每段推理几十到几百毫秒，两小时的会若不封顶会白等
    几分钟；而 60 秒语音对声纹已经足够（再多不涨准确率，只涨耗时）。
    """
    plan: dict[str, list[tuple[float, float]]] = {}
    used: dict[str, float] = {}
    for span in spans or ():
        try:
            start, end, label = float(span[0]), float(span[1]), str(span[2] or "").strip()
        except (IndexError, TypeError, ValueError):
            continue
        if not label:
            continue
        start = max(0.0, start)
        if end <= start:
            continue
        room = TOTAL_SECONDS - used.get(label, 0.0)
        if room <= 0.0:
            continue
        end = min(end, start + min(SEGMENT_SECONDS, room))
        if end <= start:
            continue
        plan.setdefault(label, []).append((start, end))
        used[label] = used.get(label, 0.0) + (end - start)
    return plan

# 模型只加载一次：一次会议几百段都走同一个实例（加载一次 ~10 秒，别每次重来）
_EMBEDDER = None

def _load_embed_model(token: str | None = None):
    """加载 wespeaker 声纹模型，兼容新旧 huggingface_hub。

    pyannote 3.x 内部固定用 ``use_auth_token=`` 调 ``hf_hub_download``，而
    huggingface_hub ≥1.0 把这个参数改名为 ``token``：直接调用会抛 TypeError
    （pyannote 没兜住，因为它只对 RepositoryNotFoundError 做了处理）。这里先按
    常规写法试一次，不行就自己用新 API 把权重取回本地缓存、再让 pyannote 从
    本地文件加载 —— 只影响本模块，不改全局、不动 pyannote。
    """
    from pyannote.audio import Model

    try:
        model = Model.from_pretrained(EMBED_MODEL, use_auth_token=token)
        if model is not None:
            return model
    except TypeError:
        pass                    # 新版 hub 不认 use_auth_token：走下面的本地加载
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(EMBED_MODEL, "pytorch_model.bin", token=token)
    return Model.from_pretrained(weights)

def _build_embedder(token: str | None = None):
    """惰性构造 wespeaker 声纹模型 → ``fn(chunk, sr) -> np.ndarray``。

    只在真要算声纹时才 import pyannote/torch（顶层 import 会让启动慢十几秒，见
    app.py 的 _ensure_heavy）。token 缺省取环境变量 HF_TOKEN（密钥在 Windows
    凭据管理器里时由调用方传进来）：模型页需先点 Agree，缓存命中后离线也能用。
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    import torch
    from pyannote.audio import Inference

    token = (token or os.getenv("HF_TOKEN") or "").strip() or None
    model = _load_embed_model(token)
    if model is None:
        raise RuntimeError("声纹模型不可用")
    inference = Inference(model, window="whole")

    def _embed(chunk, sample_rate: int) -> np.ndarray:
        waveform = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).unsqueeze(0)
        out = inference({"waveform": waveform, "sample_rate": int(sample_rate)})
        return np.asarray(out, dtype=np.float32).ravel()

    _EMBEDDER = _embed
    return _EMBEDDER

def compute_speaker_embeddings(audio, sample_rate: int, spans,
                               embed_fn=None, token: str | None = None
                               ) -> dict[str, np.ndarray]:
    """按 label 聚合语音段 → 每个 label 一条声纹向量（各段平均后 L2 归一化）。

    ``embed_fn(chunk, sr) -> 向量`` 可注入（测试用它替掉真模型，全程不 import
    torch）；``token`` 是 HF token（缺省读环境变量）。返回 ``{label: np.ndarray}``；
    模型不可用（无 token / 无缓存 / 网络失败）→ ``{}``，调用方（会后标注流程）
    据此跳过「认人」，产物不受影响。
    """
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        return {}
    if sample_rate <= 0 or audio is None:
        return {}
    try:
        wave = np.asarray(audio, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return {}
    if wave.size == 0:
        return {}
    plan = _plan_segments(spans)
    if not plan:
        return {}
    if embed_fn is None:
        try:
            embed_fn = _build_embedder(token)
        except Exception:      # 没 token / 没缓存 / 没装 torch：声纹这条线整体放弃
            return {}
    result: dict[str, np.ndarray] = {}
    for label, ranges in plan.items():
        vectors: list[np.ndarray] = []
        for start, end in ranges:
            first = int(round(start * sample_rate))
            last = min(wave.size, int(round(end * sample_rate)))
            if last <= first:
                continue
            try:
                vector = np.asarray(embed_fn(wave[first:last], sample_rate),
                                    dtype=np.float32).ravel()
            except Exception:  # 单段失败（太短 / 太吵 / 模型内部异常）：跳过，别拖垮整场
                continue
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                continue
            vectors.append(vector)
        if not vectors:
            continue
        mean = np.mean(np.stack(vectors, axis=0), axis=0)
        normalized = _l2_normalize(mean)
        if float(np.linalg.norm(normalized)) > 0.0:
            result[label] = normalized
    return result

