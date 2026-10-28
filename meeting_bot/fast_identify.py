# -*- coding: utf-8 -*-
"""快速说话人识别：用「发言回合 + 声纹向量」在一两分钟内分出谁在说。

和 pyannote 全量聚类（会后要 0.5~1.5 倍会议时长，50 分钟的会跑 75 分钟）不同，
这里利用一个现成事实：**VAD 已经把一个发言回合切出来了，一个回合里基本只有
一个人在说**。于是整件事变成：

1. 每个回合截 2~6 秒音频 → 一条声纹向量（实测 3 秒约 150 ms，几百个回合约 1 分钟）；
2. 先跟本地声纹库比对：够像某个熟人，这个簇就叫他的名字（下次开会直接显示 Peter）；
3. 不像熟人的向量做在线聚类（余弦 ≥ 阈值归簇，否则开新簇）→ 发言人A/B/C…；
4. 每个簇的**平均向量**再跟库比一次：簇比单句稳，能捞回单句认不出的人。

输出与 pyannote 完全同构（``[(start, end, label)]`` + ``{label: 姓名}``），所以
逐字稿 / 纪要 / 标记面板都不用改。秒级的 pyannote 全量标注仍保留（OFFLINE_DIARIZE），
作为事后精修。

设计约束：
- 本模块**只做纯计算**（numpy + 时间区间），不 import torch / pyannote，不碰 Qt；
  向量由调用方（子进程 ``diarize_worker --fast``）算好后传进来 —— 这样既能在主进程
  安全跑，也能被测试用合成向量覆盖。
- 认人宁缺毋滥：单段阈值默认 0.80，簇级 0.75；达不到就留给用户命名（面板）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MERGE_GAP = 0.8            # 相邻两句停顿 ≤ 0.8s 视为同一个发言回合（与逐字稿口径一致）
MAX_TURN_SECONDS = 15.0    # 单个回合最长 15s（再长就切开，声纹不需要那么长）
MIN_WINDOW = 1.2           # 短于 1.2s 的片段声纹不稳，跳过
MAX_WINDOW = 6.0           # 每段最多取 6s（实测 5s 起推理明显变慢，准确率不再涨）
# 两个簇的质心余弦 ≥ 0.62 就并成一簇。实测（200 段真实会议音频）：
# 0.72 会把同一个人拆成几十个单段簇，0.62 得到 6~9 簇，与参会人数吻合。
CLUSTER_THRESHOLD = 0.62
CLUSTER_KNOWN_THRESHOLD = 0.72   # 簇平均向量比对声纹库的阈值（簇比单段稳，可略松）
SEGMENT_KNOWN_THRESHOLD = 0.75   # 单段认人的阈值（再加簇级一致性校验，见 classify）
MIN_VOTES_RATIO = 0.6      # 簇内至少 60% 的片段投给同一个名字才算认出来
STRONG_CENTROID = 0.80     # 质心相似度足够高时，一个人也能直接命名（不必投票）


@dataclass
class IdentifyResult:
    """快速识别结果（与 pyannote 输出同构，便于直接套用）。"""

    spans: list[tuple[float, float, str]] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)        # 标签 → 认出的姓名
    scores: dict[str, float] = field(default_factory=dict)     # 标签 → 置信度
    counts: dict[str, int] = field(default_factory=dict)       # 标签 → 片段数
    embeddings: dict[str, list[float]] = field(default_factory=dict)  # 标签 → 簇平均向量

    @property
    def speakers(self) -> int:
        return len(self.counts)


def build_turns(segments, merge_gap: float = MERGE_GAP,
                max_seconds: float = MAX_TURN_SECONDS) -> list[tuple[float, float]]:
    """把逐句记录合并成「发言回合」（同一人连续说话的近似时间段）。

    只用时间信息：相邻两句停顿 ≤ ``merge_gap`` 且合起来不超 ``max_seconds`` 就并成
    一个回合。回合是「单人假设段」——声纹识别的输入单位。
    """
    turns: list[tuple[float, float]] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        start, end = float(seg.start), float(seg.end)
        if end <= start:
            continue
        if turns and start - turns[-1][1] <= merge_gap \
                and max(end, turns[-1][1]) - turns[-1][0] <= max_seconds:
            turns[-1] = (turns[-1][0], max(end, turns[-1][1]))
        else:
            turns.append((start, end))
    return turns


def plan_windows(turns: list[tuple[float, float]],
                 min_seconds: float = MIN_WINDOW,
                 max_seconds: float = MAX_WINDOW) -> list[tuple[int, float, float]]:
    """决定每个回合取哪一小段音频算声纹：返回 ``[(回合下标, 起, 止)]``。

    太短的回合并进下一个（否则白白丢掉）；仍不够长就跳过（声纹不可靠）。
    """
    windows: list[tuple[int, float, float]] = []
    pending: tuple[int, float, float] | None = None
    for idx, (start, end) in enumerate(turns):
        span_end = min(end, start + max_seconds)
        if pending is not None:
            p_idx, p_start, _p_end = pending
            # 与上一个太短的回合拼接：中间没有大停顿才拼（否则是两个人）
            if start - _p_end <= MERGE_GAP:
                span_end = min(end, p_start + max_seconds)
                if span_end - p_start >= min_seconds:
                    windows.append((p_idx, p_start, span_end))
                    pending = None
                    continue
            else:
                pending = None
        if span_end - start >= min_seconds:
            windows.append((idx, start, span_end))
        else:
            pending = (idx, start, span_end)
    return windows


def _unit(vector) -> np.ndarray | None:
    """L2 归一化；非法/零向量返回 None。"""
    try:
        arr = np.asarray(vector, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not np.isfinite(norm) or norm <= 0.0:
        return None
    return arr / norm


def _mean_vector(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return _unit(np.mean(np.stack(vectors, axis=0), axis=0))


def _match(library, vector, threshold: float) -> tuple[str, float]:
    """库比对，异常一律当「认不出」（认人是增强项，不能拖垮标注）。"""
    if library is None or vector is None:
        return "", 0.0
    try:
        hits = library.match(vector, threshold=threshold)
    except Exception:
        return "", 0.0
    if not hits:
        return "", 0.0
    try:
        name, score = str(hits[0][0]), float(hits[0][1])
    except (IndexError, TypeError, ValueError):
        return "", 0.0
    return name, score


def agglomerate(vectors: list[np.ndarray], threshold: float = CLUSTER_THRESHOLD
                ) -> list[int]:
    """凝聚聚类（质心 + 平均连接近似），返回每个向量的簇号（0 起，按首次出现排序）。

    为什么不用在线贪心：贪心的质心会随「先遇到的向量」漂移，实测把一个 50 分钟的
    会聚成「1 个 132 段的杂烩 + 30 多个单段簇」。改成反复合并「最像的两个簇」
    （average linkage 的质心近似）后，同一批音频稳定得到 6~9 簇，与参会人数吻合。

    矩阵化计算：每轮 k×k 相似度一次算完，200 段规模 ≈ 毫秒级/轮。
    """
    if not vectors:
        return []
    centroids = [np.asarray(v, dtype=np.float32) for v in vectors]
    members = [[i] for i in range(len(vectors))]
    weights = [1] * len(vectors)
    while len(centroids) > 1:
        matrix = np.stack(centroids)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit = matrix / np.maximum(norms, 1e-8)
        sim = unit @ unit.T
        np.fill_diagonal(sim, -1.0)
        flat = int(np.argmax(sim))
        i, j = divmod(flat, len(centroids))
        if float(sim[i, j]) < threshold:
            break
        total = weights[i] + weights[j]
        merged = (centroids[i] * weights[i] + centroids[j] * weights[j]) / total
        keep, drop = (i, j) if i < j else (j, i)
        unit_merged = _unit(merged)
        if unit_merged is not None:
            centroids[keep] = unit_merged
        weights[keep] = total
        members[keep].extend(members[drop])
        del centroids[drop], weights[drop], members[drop]
    # 按「首次出现的向量下标」重排簇号，保证结果稳定（不随合并顺序抖动）
    order = sorted(range(len(members)), key=lambda c: min(members[c]))
    labels = [0] * len(vectors)
    for new_index, cluster in enumerate(order):
        for idx in members[cluster]:
            labels[idx] = new_index
    return labels


def _absorb_small_clusters(clusters: list[dict], turns: list[tuple[float, float]],
                           min_members: int = 2, min_seconds: float = 3.0,
                           absorb_threshold: float = 0.5) -> list[dict]:
    """把小簇并进最像的大簇；并不进去的直接丢弃（宁可留空，也不硬塞一个「发言人」）。

    实测 200 段里会有十来个「只有一个片段」的簇：多是应答词（"yeah"）、咳嗽、
    两个人的声音重叠，或是真正的插话者但只说了一句。给它们各开一个发言人会让
    逐字稿冒出「发言人J/K/L」，反而不如留空（会后 pyannote 精修或人工命名）。
    """
    big = [c for c in clusters
           if len(c["vectors"]) >= min_members
           and sum(turns[i][1] - turns[i][0] for i in c["turns"]) >= min_seconds]
    big_ids = {id(c) for c in big}          # 用 id 比较：簇 dict 里装了 numpy 数组
    small = [c for c in clusters if id(c) not in big_ids]
    if not big:
        return clusters
    for cluster in small:
        centroid = cluster["centroid"]
        best, best_score = None, -1.0
        if centroid is not None:
            for target in big:
                if target["centroid"] is None:
                    continue
                score = float(np.dot(target["centroid"], centroid))
                if score > best_score:
                    best, best_score = target, score
        if best is not None and best_score >= absorb_threshold:
            best["vectors"].extend(cluster["vectors"])
            best["turns"].extend(cluster["turns"])
            best["centroid"] = _mean_vector(best["vectors"])
        # 并不进去 → 丢弃（这些回合的 speaker 留空）
    return big


def _label_of(index: int, first_index: int) -> str:
    return f"SPEAKER_{first_index + index:02d}"


def classify(windows: list[tuple[int, float, float]], vectors: list,
             turns: list[tuple[float, float]], library=None,
             first_index: int = 0,
             cluster_threshold: float = CLUSTER_THRESHOLD,
             known_threshold: float = SEGMENT_KNOWN_THRESHOLD,
             cluster_known_threshold: float = CLUSTER_KNOWN_THRESHOLD
             ) -> IdentifyResult:
    """把「每段声纹向量」聚成说话人，并尽量贴上学过的名字。

    ``windows``：``plan_windows`` 的输出（顺序即时间顺序）；
    ``vectors``：与 windows 一一对应的声纹向量（可为 None / 空，跳过即可）；
    ``turns``：``build_turns`` 的输出（用来给**没采样到**的回合补标签）；
    ``library``：``VoiceprintLibrary`` 或任何有 ``match(vector, threshold=)`` 的对象；
    ``first_index``：新分配的 SPEAKER_xx 起始编号（避免与录制中已有的标签撞车）。

    返回 :class:`IdentifyResult`：每个回合一条 span（同一个人相邻回合合并），
    加上「标签 → 姓名」的名字表 + 置信度 + 每簇平均向量（供声纹学习）。
    """
    result = IdentifyResult()
    if not turns:
        return result
    marks: list[str] = [""] * len(turns)          # 每个回合的标签

    # 收集「回合下标 + 单位向量」，再一次性凝聚聚类（顺序无偏）
    entries: list[tuple[int, np.ndarray]] = []
    for (idx, _start, _end), raw in zip(windows, list(vectors) + [None] * len(windows)):
        if idx < 0 or idx >= len(turns):
            continue
        vec = _unit(raw)
        if vec is not None:
            entries.append((idx, vec))
    if not entries:
        return result
    labels = agglomerate([vec for _idx, vec in entries],
                         threshold=float(cluster_threshold))
    clusters: list[dict] = []
    for (idx, vec), cluster_no in zip(entries, labels):
        while len(clusters) <= cluster_no:
            clusters.append({"vectors": [], "centroid": None, "turns": []})
        cluster = clusters[cluster_no]
        cluster["vectors"].append(vec)
        cluster["turns"].append(idx)
    for cluster in clusters:
        cluster["centroid"] = _mean_vector(cluster["vectors"])
    clusters = _absorb_small_clusters(clusters, turns)

    # 簇级认人：成员投票 + 平均向量复比，两个渠道都试，票多的赢
    named_labels: dict[str, str] = {}             # 姓名 → 标签（同名簇合并成一个标签）
    for order, cluster in enumerate(clusters):
        label = _label_of(order, first_index)
        votes: dict[str, list[float]] = {}
        for vec in cluster["vectors"]:
            name, score = _match(library, vec, known_threshold)
            if name:
                votes.setdefault(name, []).append(score)
        centroid_name, centroid_score = _match(library, cluster["centroid"],
                                               cluster_known_threshold)
        name, score = "", 0.0
        if votes:
            best_name = max(votes, key=lambda n: (len(votes[n]), max(votes[n])))
            enough = len(votes[best_name]) >= max(1, round(MIN_VOTES_RATIO * len(cluster["vectors"])))
            if best_name == centroid_name or enough:
                name = best_name
                score = max(votes[best_name])
        if not name and centroid_name and centroid_score >= STRONG_CENTROID:
            name, score = centroid_name, centroid_score
        if name:
            # 同一个人被切成两个簇（中间换了音量/麦）：合并成同一个标签，
            # 否则逐字稿里会出现「两个 Peter Ren」。
            label = named_labels.setdefault(name, label)
            result.names[label] = name
            result.scores[label] = max(result.scores.get(label, 0.0), round(float(score), 4))
        centroid = cluster["centroid"]
        if centroid is not None:
            result.embeddings[label] = [float(x) for x in centroid]
        result.counts[label] = result.counts.get(label, 0) + len(cluster["vectors"])
        for idx in cluster["turns"]:
            marks[idx] = label

    # 太短没采样的回合：贴给时间上最近的已分类回合（比留空更符合「谁在说」）
    if clusters:
        for idx in range(len(turns)):
            if marks[idx]:
                continue
            for offset in range(1, len(turns)):
                prev_idx, next_idx = idx - offset, idx + offset
                if 0 <= prev_idx < len(turns) and marks[prev_idx]:
                    marks[idx] = marks[prev_idx]
                    break
                if 0 <= next_idx < len(turns) and marks[next_idx]:
                    marks[idx] = marks[next_idx]
                    break

    # 相邻同标签回合合并成一条 span（与 pyannote 的 turn 口径一致）
    for idx, label in enumerate(marks):
        if not label:
            continue
        start, end = turns[idx]
        if result.spans and result.spans[-1][2] == label and start - result.spans[-1][1] <= MERGE_GAP:
            result.spans[-1] = (result.spans[-1][0], max(end, result.spans[-1][1]), label)
        else:
            result.spans.append((start, end, label))
    return result
