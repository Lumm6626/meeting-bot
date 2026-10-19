# -*- coding: utf-8 -*-
"""字幕浮窗：Teams 会议转录式滚动面板。

设计要点（对外接口：``SubtitleWindow(mode, name_resolver=)``、``show_segment(seg,
name)``、``update_segment(seg, name)`` 与 ``refresh_names()``——前三条 app.py 一直在
用、不可变更；``name_resolver`` / ``refresh_names`` 是「改名回填整场字幕」新增的
两条可选接口，默认行为与旧版完全一致）：

* 两阶段显示：识别完即 ``show_segment``（此时无译文 → 主标签显示原文），
  翻译到达后 ``update_segment`` 按 seq 找到同一句，**原地**替换并重渲染该块
  （不新增块、不重复文本；同 seq 重复到达按更新处理 → 幂等）；
* 回合分块历史（Teams 转录风）：同一说话人连续说的话合成一个「回合块」——
  头行时间+说话人、正文为回合内各句译文**连续拼成的段落**（不再逐句换行，
  免得空行把译文挤得看不全）、副标签为回合内各句原文用空格拼成的小段落；
  换人（声纹标签变化）或停顿超过 ``TURN_GAP_SECONDS`` 才开新块；同回合追加
  语句时原地更新该块（不新增条目）；最多保留 ``MAX_TURNS`` 个块，丢弃最旧；
* 自动跟随：停在底部时新记录自动滚到底；用户向上翻看时暂停，滚回底部恢复；
* 自由拉伸：无边框窗口自绘圆角黑底，8px 边缘/四角命中即改光标拖拽缩放；
  非边缘区域按住左键移动窗口（沿用原实现）；
* 字号自适配：基础字号 = 窗口宽度 × ``BASE_RATIO``（12~40 钳制），
  滚轮 / +/- 键调整倍率（0.6~2.2），实际字号 = 基础字号 × 倍率，全部条目同步；
* 显示模式：``bilingual`` 中文段落 + 灰色小字原文（默认）/ ``zh_only`` 只显示中文
  （原文段整段隐藏、块高度自适应）。右键菜单切换，选择记在 QSettings
  (``meeting-bot/subtitle``) 的 ``mode`` 键里，下次启动优先用它。
* 说话人名字实时解析（``name_resolver``）：块头行的名字不再固定成「入块那一刻的
  字符串」，而是每帧按当前解析器算 —— 用户在主窗口表格里把 ``SPEAKER_00`` 改成
  「张三」后调 :meth:`SubtitleWindow.refresh_names`，整场已显示的块立刻换名字
  （正文文本、块数量、滚动位置都不动）。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6.QtCore import QEvent, QPoint, QRectF, QSettings, QTimer, Qt
from PyQt6.QtGui import QActionGroup, QColor, QPainter
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)

from .config import SUBTITLE_MODES
from .transcript import Segment, fmt_ts

# 半透明黑底参数（paintEvent 与样式表共用，避免两处硬编码漂移）
BG_RGBA = (0, 0, 0, 160)
BG_RADIUS = 10
BG_STYLE = (
    f"background-color: rgba({BG_RGBA[0]}, {BG_RGBA[1]}, {BG_RGBA[2]}, {BG_RGBA[3]}); "
    f"border-radius: {BG_RADIUS}px;"
)

COLOR_META = "#4FC3F7"      # 时间 + 说话人
COLOR_MAIN = "#FFFFFF"      # 译文（主字号）
COLOR_SUB = "#B0BEC5"       # 原文 + 语种
COLOR_ZOOM = "#90A4AE"      # 角落倍率提示
COLOR_REC_DOT = "#FF5252"   # 录制红点
REC_PILL_STYLE = ("background-color: rgba(229, 57, 53, 110);"
                  " border-radius: 9px;")

TURN_GAP_SECONDS = 4.0      # 相邻语句起始时间间隔上限，超过即视为新回合（新块）
# 2.0 太小：连续说话时句中稍作停顿（2~4 秒）就被切成新块，字幕一块块跳；
# 4.0 让一段连续发言基本落在一个回合块里（换人仍然立刻切块，见 _same_turn）。
MAX_TURNS = 60              # 历史回合块上限（一个回合 = 一个条目块）
MAX_ENTRIES = MAX_TURNS     # 兼容旧命名（历史上限，现按回合块计数）
MIN_W, MIN_H = 320, 200     # 最小窗口尺寸
EDGE = 8                    # 边缘拖拽判定宽度（px）
ZOOM_MIN, ZOOM_MAX = 0.6, 2.2
ZOOM_STEP = 0.1
BASE_RATIO = 0.022          # 基础字号 = 宽度 × 该比例
FONT_MIN, FONT_MAX = 12, 40
ZOOM_BADGE_MS = 3000        # 倍率提示自动隐藏时间

# ---------------------------------------------------------------- 显示模式
# bilingual：中文段落 + 灰色小字原文段（默认，与既有行为一致）
# zh_only  ：只显示中文段落，原文段整段隐藏（块高度自适应变矮）
MODE_BILINGUAL = "bilingual"
MODE_ZH_ONLY = "zh_only"
# 右键菜单顺序（标签 → 模式）；勾选态由当前模式决定
MODE_MENU = ((MODE_ZH_ONLY, "中文"), (MODE_BILINGUAL, "中英对照"))

SETTINGS_ORG = "meeting-bot"     # QSettings 作用域：记忆用户上次的选择
SETTINGS_APP = "subtitle"
SETTINGS_MODE_KEY = "mode"


def normalize_mode(mode: str | None) -> str:
    """收敛成合法模式：空值 / 拼错 / None 一律回落 bilingual。

    .env 的合法性由 config.load_settings 严格校验（非法直接 ConfigError）；
    这里只服务 UI 层（构造参数、QSettings 里的历史脏值），绝不因此崩溃。
    """
    m = (mode or "").strip().lower()
    return m if m in SUBTITLE_MODES else MODE_BILINGUAL


def _subtitle_settings() -> QSettings:
    """QSettings 工厂（独立成函数：测试可 monkeypatch 到临时作用域，不污染注册表）。"""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def load_saved_mode() -> str | None:
    """读上次记住的模式；没记过 / 值非法 / 后端不可用 → None（交给构造参数）。"""
    try:
        raw = _subtitle_settings().value(SETTINGS_MODE_KEY, "")
    except Exception:          # 记忆失败绝不能拖垮字幕窗口
        return None
    mode = str(raw or "").strip().lower()
    return mode if mode in SUBTITLE_MODES else None


def save_mode(mode: str) -> None:
    """把模式写进 QSettings（立即 sync，异常吞掉：只是「记忆」，不是功能）。"""
    try:
        settings = _subtitle_settings()
        settings.setValue(SETTINGS_MODE_KEY, normalize_mode(mode))
        settings.sync()
    except Exception:          # 同上：注册表/配置文件不可写时静默忽略
        pass

SCROLL_STYLE = """
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical {
    background: rgba(176, 190, 197, 120); border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""

_CURSORS = {
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
}


# ---------------------------------------------------------------- 段落拼接
# 块内不再逐句换行：同一回合的各句连续拼成一段（译文段用 "，"，原文段用 " "）。
# 补分隔符的判据 —— 只有「上一句没有句末标点」且「下一句不以标点开头」时才补：
#   "你好" + "世界"           → "你好，世界"
#   "你好" + "！"             → "你好！"          （下一句以标点开头，不补）
#   "今天下雨。" + "我们改期"  → "今天下雨。我们改期"（已有句末标点，直接相接）
#   "hello," + "world"        → "hello, world"   （punct_gap=" "：英文词边界不能丢）
SENTENCE_END_PUNCT = "。！？；：，、…" + ".!?;:,"
SENTENCE_START_PUNCT = "。！？；：，、…" + ".!?;:,)）】」』”’"
CJK_PUNCT = "。！？；：，、…“”‘’（）【】《》"   # 全角标点自带留白，后面不再补空格
MAIN_JOIN_SEP = "，"     # 译文段分隔符
SUB_JOIN_SEP = " "       # 原文段分隔符（英文自然）


def join_paragraph(parts: Iterable[str], sep: str, *, punct_gap: str = "") -> str:
    """把一句句文本拼成一整段（不插换行、不留空行）。

    ``sep``：上一句没有句末标点、下一句也不以标点开头时补的连接符；
    ``punct_gap``：上一句以**半角**标点结尾时补的连接符（英文原文段传空格，
    否则 "hello," 与 "world" 会粘成 "hello,world"）；全角标点后一律不再补。
    空片段（含纯空白）直接跳过，全部为空时返回空串。
    """
    out = ""
    for raw in parts:
        piece = (raw or "").strip()
        if not piece:
            continue
        if out:
            last = out[-1]
            if last in SENTENCE_END_PUNCT:
                if punct_gap and last not in CJK_PUNCT:
                    out += punct_gap
            elif piece[0] not in SENTENCE_START_PUNCT:
                out += sep
        out += piece
    return out


def _format_original_paragraph(texts: list[str], langs: list[str]) -> str:
    """原文段：多句原文拼成一小段，语种标签只在段首出现一次。

    同一回合内混用多种语种时（少见）退回逐句前缀，免得张冠李戴；无语种则不加。
    """
    if not texts:
        return ""
    distinct = {lang for lang in langs if lang}
    if len(distinct) > 1:
        texts = [f"({lang}) {text}" if lang else text for text, lang in zip(texts, langs)]
    body = join_paragraph(texts, SUB_JOIN_SEP, punct_gap=SUB_JOIN_SEP)
    if len(distinct) == 1:
        return f"({next(iter(distinct))}) {body}"
    return body


class SubtitleEntry(QWidget):
    """一个「说话回合」块。

    头行（小号蓝）：回合首句时间 + 说话人；
    主标签（主字号白）：回合内各句译文连续拼成一整段，译文缺失的句子用原文顶上；
    副标签（小号灰）：回合内各句原文用空格拼成一小段（语种标签只在段首出现一次）
    ——某句译文与原文完全一致（或该句无译文）时原文已在主标签可见，不重复进副标签；
    副标签整体为空时整行隐藏。

    同回合追加语句走 :meth:`add_segment`，原地重建标签文本，控件数量不变。

    显示模式（``mode``）：``bilingual`` 主标签译文 + 副标签原文；
    ``zh_only`` 副标签整行隐藏（文本也清空 → 不占高），只留中文段落。
    ``name_resolver``：把声纹标签实时解析成显示名的函数（见 :meth:`_speaker_name`）。
    """

    def __init__(self, seg: Segment, display_name: str, parent: QWidget | None = None,
                 *, mode: str = MODE_BILINGUAL,
                 name_resolver: Callable[[str], str] | None = None):
        super().__init__(parent)
        # 整块不吃鼠标：点击穿透给容器，由窗口统一做移动/缩放
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.display_name = display_name
        # 名字解析器（默认 None → 头行沿用 display_name，与旧行为一致）；
        # 必须先于 add_segment 赋值：构造时就按它渲染头行
        self.name_resolver = name_resolver
        self._mode = normalize_mode(mode)
        self.segments: list[Segment] = []

        self.meta = QLabel("")
        self.main = QLabel("")
        self.sub = QLabel("")
        for w in (self.meta, self.main, self.sub):
            w.setWordWrap(True)
            w.setTextFormat(Qt.TextFormat.PlainText)
            w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.meta.setStyleSheet(f"color: {COLOR_META}; font-weight: bold; background: transparent;")
        self.main.setStyleSheet(f"color: {COLOR_MAIN}; background: transparent;")
        self.sub.setStyleSheet(f"color: {COLOR_SUB}; background: transparent;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.meta)
        lay.addWidget(self.main)
        lay.addWidget(self.sub)
        # 副标签常驻布局、按内容显隐：回合内句子增加后可能需要重新显示
        self.sub.hide()
        self.add_segment(seg)

    # ------------------------------------------------------------------ 内容
    @property
    def first(self) -> Segment:
        """回合首句（头行时间取它）。"""
        return self.segments[0]

    @property
    def speaker(self) -> str:
        """回合内首个非空声纹标签，没有则空串。"""
        for s in self.segments:
            if s.speaker:
                return s.speaker
        return ""

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """切换本块的显示模式并原地重渲染（高度随之变化，不新建/搬动控件）。"""
        mode = normalize_mode(mode)
        if mode == self._mode:
            return
        self._mode = mode
        self._rebuild()

    # -------------------------------------------------------------- 说话人名
    def _speaker_name(self) -> str:
        """头行显示的名字：有 resolver 就按当前声纹标签实时解析。

        resolver 为 None（旧行为 / 未接线）→ 直接用构造时传入的 ``display_name``。
        resolver 抛异常、返回空串时同样退回 ``display_name``：头行渲染绝不能因为
        「查名字」这一步把整块字幕弄崩（解析的是 UI 数据，不是控制流）。
        """
        resolver = self.name_resolver
        if resolver is None:
            return self.display_name
        try:
            name = resolver(self.speaker)
        except Exception:               # noqa: BLE001 —— 渲染路径必须兜住任何异常
            return self.display_name
        return name or self.display_name

    def _render_meta(self) -> None:
        """重绘头行（首句时间 + 说话人）；正文两块文本完全不碰。"""
        self.meta.setText(f"{fmt_ts(self.segments[0].start)}  {self._speaker_name()}")

    def refresh_name(self) -> None:
        """按当前 resolver 重渲染头行（改名回填用：不动文本、不新增块）。"""
        if self.segments:
            self._render_meta()

    def add_segment(self, seg: Segment) -> None:
        """把一条语句并入本回合（顺序追加），原地刷新三个标签。"""
        before = self.speaker
        self.segments.append(seg)
        # 首句入块 → 必须画头行；标签延迟出现（后面某句才带上声纹标签）也要跟上，
        # 否则头行会一直停在「发言人?」直到有人改名
        if len(self.segments) == 1 or self.speaker != before:
            self._render_meta()
        self._rebuild()

    def replace_segment(self, seg: Segment) -> bool:
        """把块内 seq 相同的那一句换成 ``seg`` 并原地重渲染（译文到达时用）。

        返回 False 表示本块不含该 seq（调用方据此判断是不是「新句子」）。
        位置不变：译文替换原文时块的排版、行数、历史顺序都不动。
        """
        for i, old in enumerate(self.segments):
            if old.seq == seg.seq:
                self.segments[i] = seg
                if i == 0:
                    # 头行时间取回合首句，替换首句时一并刷新（换人/改名同理）
                    self._render_meta()
                self._rebuild()
                return True
        return False

    def _rebuild(self) -> None:
        """按「段落」重建两块文本：译文段（主标签）+ 原文段（副标签），无换行。

        译文段：各句译文连续拼接（缺译文 → 原文占位，译文到达后原地换掉）；
        原文段：只收**真有译文**的句子（无译文/译文==原文的句子已在译文段可见，
        不重复），用空格拼成一小段。两段内都不插入换行或空行。
        ``zh_only`` 模式下原文段整段隐藏（清空文本 + hide），只渲染译文段。
        """
        mains: list[str] = []
        subs: list[str] = []
        sub_langs: list[str] = []
        for s in self.segments:
            tr = (s.translation or "").strip()
            txt = (s.text or "").strip()
            mains.append(tr or txt)             # 译文缺失 → 原文顶上主标签
            if tr and tr != txt:                # 译文==原文时原文已在主标签，不重复
                subs.append(txt)
                sub_langs.append((s.lang or "").strip())
        self.main.setText(join_paragraph(mains, MAIN_JOIN_SEP))
        if self._mode == MODE_ZH_ONLY:
            # 只显示中文：原文段整行隐藏，文本一并清空（空 QLabel 高度归零 → 块自适应）
            self.sub.setText("")
            self.sub.setVisible(False)
            return
        self.sub.setText(_format_original_paragraph(subs, sub_langs))
        self.sub.setVisible(bool(subs))

    def set_sizes(self, main_pt: int) -> None:
        """主字号 ``main_pt``；小号行取 0.72 倍（下限 9pt）。"""
        sub_pt = max(9, round(main_pt * 0.72))
        for w, pt in ((self.meta, sub_pt), (self.main, main_pt), (self.sub, sub_pt)):
            f = w.font()
            f.setPointSize(pt)
            w.setFont(f)


class SubtitleWindow(QWidget):
    def __init__(self, mode: str = MODE_BILINGUAL, *,
                 name_resolver: Callable[[str], str] | None = None):
        """``mode`` 默认 bilingual（既有 ``SubtitleWindow()`` 调用行为不变）。

        优先级：QSettings 里记住的值 > 构造参数 > 默认 bilingual。
        （用户可以右键切换过模式，下次启动要沿用他的选择，而不是 .env 的老值。）

        ``name_resolver``：声纹标签 → 显示名（默认 None，头行沿用 ``show_segment``
        传来的名字）。也可以构造后再赋值（``win.name_resolver = store.display_name``
        —— recorder 每场新建，app.py 就是这么做的），改名时调 :meth:`refresh_names`
        让已显示的块立刻跟手。
        """
        super().__init__()
        self._mode = normalize_mode(load_saved_mode() or mode)
        self.name_resolver = name_resolver      # 新块创建时继承（见 _add_block）
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # QWidget 子类默认不绘制样式表背景，需显式开启 WA_StyledBackground
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(BG_STYLE)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(MIN_W, MIN_H)     # 与拖拽缩放的最小尺寸保持一致

        self._entries: list[SubtitleEntry] = []      # 回合块（旧 -> 新）
        # seq → 所在回合块：译文原地更新按 seq 定位（只含保留窗口内的块，
        # 块被 MAX_TURNS 淘汰时同步摘除 → 不会指向已删除控件）
        self._seq_block: dict[int, SubtitleEntry] = {}
        self._current_turn: list[Segment] = []       # 当前未关闭回合的语句
        self._current_block: SubtitleEntry | None = None
        self._current_speaker = ""                   # 当前回合的声纹标签（可能为空）
        self._zoom = 1.0
        self._recording = False        # 录制中提示态（set_recording 维护）
        self._auto_scroll = True          # 停在底部 → True，用户向上翻看 → False
        self._op: str | None = None       # "move" / "resize"
        self._grab_edge: str | None = None
        self._press_global = QPoint()
        self._press_geom = self.geometry()
        self._move_offset = QPoint()

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(SCROLL_STYLE)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scroll.viewport().setAutoFillBackground(False)

        self._container = QWidget()
        self._container.setAutoFillBackground(False)
        self._container.setStyleSheet("background: transparent;")
        self._entries_layout = QVBoxLayout(self._container)
        self._entries_layout.setContentsMargins(0, 0, 0, 0)
        self._entries_layout.setSpacing(10)
        self._entries_layout.addStretch(1)      # 条目少时贴顶显示
        self.scroll.setWidget(self._container)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(0)
        lay.addWidget(self.scroll)

        self.zoom_label = QLabel("1.0x", self)
        self.zoom_label.setStyleSheet(
            f"color: {COLOR_ZOOM}; background: transparent; font-weight: bold;")
        self.zoom_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        f = self.zoom_label.font()
        f.setPointSize(9)
        self.zoom_label.setFont(f)
        self.zoom_label.hide()

        # 录制中提示：右上角半透明红底 pill（红点 + 「录音中」小字），默认隐藏
        self.recording_pill = QFrame(self)
        self.recording_pill.setObjectName("recording_pill")
        self.recording_pill.setStyleSheet(REC_PILL_STYLE)
        self.recording_pill.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        pill_lay = QHBoxLayout(self.recording_pill)
        pill_lay.setContentsMargins(8, 2, 10, 2)
        pill_lay.setSpacing(4)
        self.recording_dot = QLabel("●", self.recording_pill)
        self.recording_dot.setStyleSheet(
            f"color: {COLOR_REC_DOT}; background: transparent; font-weight: bold;")
        self.recording_label = QLabel("录音中", self.recording_pill)
        self.recording_label.setStyleSheet(
            "color: #FFFFFF; background: transparent; font-weight: bold;")
        for lab in (self.recording_dot, self.recording_label):
            f = lab.font()
            f.setPointSize(9)
            lab.setFont(f)
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            pill_lay.addWidget(lab)
        self.recording_pill.adjustSize()
        self.recording_pill.hide()

        bar = self.scroll_bar()
        bar.valueChanged.connect(self._on_scroll_changed)
        bar.rangeChanged.connect(self._on_range_changed)

        # 子控件（滚动区/视口/容器）自己会吃掉鼠标与滚轮，统一过滤回窗口处理
        for w in (self.scroll, self.scroll.viewport(), self._container,
                  self.zoom_label, self.recording_pill):
            w.installEventFilter(self)

        # 布局落定后再补一次滚到底（rangeChanged 已覆盖多数情况，这是兜底）
        self._stick_timer = QTimer(self)
        self._stick_timer.setSingleShot(True)
        self._stick_timer.setInterval(0)
        self._stick_timer.timeout.connect(self._stick_if_auto)

        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(ZOOM_BADGE_MS)
        self._zoom_timer.timeout.connect(self.zoom_label.hide)

        self.resize(720, 360)

    # ------------------------------------------------------------------ 绘制
    def paintEvent(self, e):
        """自绘半透明黑底圆角。

        WA_TranslucentBackground 的顶层窗口 Qt 不会绘制任何背景（样式表背景
        同样不画，实测 alpha 全 0），故必须在此手动绘制。用 CompositionMode_Source
        覆盖式写入，保证无论绘制几次，结果都是精确的 rgba(0,0,0,160)。
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(*BG_RGBA))
        p.drawRoundedRect(QRectF(self.rect()), BG_RADIUS, BG_RADIUS)
        p.end()

    # ------------------------------------------------------------ 对外接口
    def show_segment(self, seg: Segment, display_name: str):
        """把一条语句并入当前回合（必要时新开一个回合块）并滚到底部。

        幂等：同一 seq 再次到达（重发 / 两阶段显示下的竞态）按「原地更新」处理，
        不追加新块、不重复文本。
        """
        if not self.update_segment(seg, display_name):
            self._merge_into_turn(seg, display_name)
        if not self.isVisible():
            self.show()
        self.raise_()
        self._stick_if_auto()
        self._stick_timer.start()

    def update_segment(self, seg: Segment, display_name: str = "") -> bool:
        """译文到达后按 seq 找到原块并原地替换该句（不新增块、不搬动顺序）。

        返回 True 表示找到并已重渲染；False 表示该 seq 不在保留窗口内
        （块已被 MAX_TURNS 淘汰，或本就是新句子）。
        """
        block = self._seq_block.get(seg.seq)
        if block is None or not block.replace_segment(seg):
            return False
        if block is self._current_block:
            # current_turn 只用于分块判定（start/speaker），但保持同一批对象，
            # 免得「当前回合」里的语句还带着旧译文（调试/测试看到的视图失真）
            for i, s in enumerate(self._current_turn):
                if s.seq == seg.seq:
                    self._current_turn[i] = seg
                    break
        # 更新后内容变高/变短 → 仍按「停在底部才跟随」的规则重新贴底
        self._stick_if_auto()
        self._stick_timer.start()
        return True

    # ------------------------------------------------------------ 显示模式
    @property
    def mode(self) -> str:
        """当前显示模式：``bilingual``（中英对照）或 ``zh_only``（只中文）。"""
        return self._mode

    def set_mode(self, mode: str, *, persist: bool = True) -> None:
        """切换显示模式：已显示的块逐个重渲染，滚动位置尽量不跳。

        位置处理：先记下 scrollbar 的 value 与「是否停在底部」，重渲染后原位恢复
        （块变矮会让 range 收缩，超出部分由 QScrollBar 自行钳制）；原本停在底部、
        开着自动跟随的，继续贴底（不打断「跟随最新」的观感）。
        """
        mode = normalize_mode(mode)
        bar = self.scroll_bar()
        value = bar.value()
        was_following = self._auto_scroll
        changed = mode != self._mode
        self._mode = mode
        if changed:
            for entry in self._entries:
                entry.set_mode(mode)
            if was_following:
                self._stick_if_auto()
            else:
                bar.setValue(min(value, bar.maximum()))
        if persist:
            save_mode(mode)

    # ------------------------------------------------------------ 名字回填
    def refresh_names(self) -> None:
        """按当前 ``name_resolver`` 重渲染**所有已显示块**的头行（改名回填）。

        不动正文文本、不新增/搬动块、不重建控件：只把每个块的「时间 + 说话人」
        这一行重新算一遍。滚动位置与 ``set_mode`` 同一套处理 —— 原本停在底部且
        开着自动跟随的继续贴底，用户上翻看历史时保持原位不跳。
        """
        bar = self.scroll_bar()
        value = bar.value()
        was_following = self._auto_scroll
        for entry in self._entries:
            if entry.name_resolver is None:
                # 只给「还没接线」的块补上解析器。已经接过线的块保留自己那一个 ——
                # 面板跨会议不重建，若一律换成当前 store，上一场的块会被新一场的
                # 名字污染（同一标签在两场里常常是不同的人）。
                entry.name_resolver = self.name_resolver
            entry.refresh_name()
        if was_following:
            self._stick_if_auto()
        else:
            bar.setValue(min(value, bar.maximum()))

    # ---------------------------------------------------------- 录制状态提示
    def set_recording(self, active: bool) -> None:
        """录制中在右上角显示红点 + 「录音中」pill；停止后隐藏（幂等）。"""
        self._recording = bool(active)
        pill = getattr(self, "recording_pill", None)
        if pill is None:            # 构造期异常状态：不影响既有接口
            return
        if self._recording:
            self._position_recording_pill()
            pill.show()
            pill.raise_()
        else:
            pill.hide()

    @property
    def recording(self) -> bool:
        """当前是否处于「录制中」提示态。"""
        return bool(getattr(self, "_recording", False))

    def _position_recording_pill(self) -> None:
        pill = getattr(self, "recording_pill", None)
        if pill is None:
            return
        pill.adjustSize()
        pill.move(max(6, self.width() - pill.width() - 16), 10)

    # ------------------------------------------------------------ 测试辅助
    def entries(self) -> list[SubtitleEntry]:
        """按时间顺序（旧 → 新）返回当前保留的回合块。"""
        return list(self._entries)

    @property
    def current_turn(self) -> list[Segment]:
        """当前未关闭回合里的语句（副本）。"""
        return list(self._current_turn)

    def scroll_bar(self):
        return self.scroll.verticalScrollBar()

    def current_font_size(self) -> int:
        """当前主字号 = 宽度比例基础字号 × 倍率。"""
        base = int(self.width() * BASE_RATIO)
        base = max(FONT_MIN, min(FONT_MAX, base))
        return max(8, int(round(base * self._zoom)))

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def auto_scroll(self) -> bool:
        return self._auto_scroll

    def set_zoom(self, value: float) -> None:
        value = max(ZOOM_MIN, min(ZOOM_MAX, round(float(value), 2)))
        changed = abs(value - self._zoom) > 1e-9
        self._zoom = value
        if changed:
            self._apply_fonts()
        self._show_zoom_badge()

    # -------------------------------------------------------- 回合归并/分块
    def _merge_into_turn(self, seg: Segment, display_name: str) -> None:
        """同回合 → 原地追加；停顿或换人 → 关闭当前回合、开新块。"""
        if self._current_block is not None and self._same_turn(seg):
            self._current_block.add_segment(seg)
            self._seq_block[seg.seq] = self._current_block
            self._current_turn.append(seg)
            if not self._current_speaker and seg.speaker:
                # 说话人标签延迟出现（pyannote 滑窗打标后才有）→ 归入本回合并记住
                self._current_speaker = seg.speaker
            return
        self._start_turn(seg, display_name)

    def _same_turn(self, seg: Segment) -> bool:
        """新语句是否属于当前回合。"""
        if not self._current_turn:
            return False
        last = self._current_turn[-1]
        # 间隔按「起始时间差」判定：VAD/声纹切出的语句常首尾相接甚至重叠，
        # 用上一句 end 到本句 start 的空隙判定会永远合并，识别不出停顿。
        if abs(seg.start - last.start) > TURN_GAP_SECONDS:
            return False
        # 只有两边都有声纹标签且不同才算换人；标签为空（未配 HF token 的现状）
        # 时只按时间间隔分块。
        if seg.speaker and self._current_speaker and seg.speaker != self._current_speaker:
            return False
        return True

    def _start_turn(self, seg: Segment, display_name: str) -> None:
        self._current_turn = [seg]
        self._current_speaker = seg.speaker
        self._current_block = self._add_block(seg, display_name)

    # -------------------------------------------------------------- 块管理
    def _add_block(self, seg: Segment, display_name: str) -> SubtitleEntry:
        entry = SubtitleEntry(seg, display_name, self._container, mode=self._mode,
                              name_resolver=self.name_resolver)
        entry.set_sizes(self.current_font_size())
        for w in (entry, entry.meta, entry.main, entry.sub):
            w.installEventFilter(self)
        self._entries_layout.insertWidget(self._entries_layout.count() - 1, entry)
        self._entries.append(entry)
        self._seq_block[seg.seq] = entry
        # 当前回合块永远是最新的一块，裁剪只会淘汰更早的块
        while len(self._entries) > MAX_TURNS:
            old = self._entries.pop(0)
            # 先 hide 再 deleteLater：removeWidget 后它在下一个事件循环前仍是可见子控件
            # （会闪一下）；不 setParent(None)，避免瞬时冒出一个顶层窗口
            old.hide()
            self._entries_layout.removeWidget(old)
            old.deleteLater()
            # seq 映射同步摘除：否则译文更新会指向已删除的块（Qt 侧已 deleteLater）
            for s in old.segments:
                if self._seq_block.get(s.seq) is old:
                    self._seq_block.pop(s.seq, None)
        return entry

    # -------------------------------------------------------------- 字号/滚动
    def _apply_fonts(self) -> None:
        size = self.current_font_size()
        for entry in self._entries:
            entry.set_sizes(size)

    def _show_zoom_badge(self) -> None:
        self.zoom_label.setText(f"{self._zoom:.1f}x")
        self._position_zoom_label()
        self.zoom_label.show()
        self.zoom_label.raise_()
        self._zoom_timer.start()

    def _position_zoom_label(self) -> None:
        label = getattr(self, "zoom_label", None)
        if label is None:           # 构造期 resizeEvent 可能早于控件创建
            return
        label.adjustSize()
        label.move(
            max(6, self.width() - label.width() - 16),
            max(4, self.height() - label.height() - 12),
        )

    def _on_scroll_changed(self, value: int) -> None:
        # 停留底部 → 自动跟随；用户往上翻 → 暂停
        self._auto_scroll = value >= self.scroll_bar().maximum()

    def _on_range_changed(self, _min: int, maximum: int) -> None:
        if self._auto_scroll:
            self.scroll_bar().setValue(maximum)

    def _stick_if_auto(self) -> None:
        if self._auto_scroll:
            bar = self.scroll_bar()
            bar.setValue(bar.maximum())

    def _zoom_step(self, direction: int) -> None:
        self.set_zoom(self._zoom + direction * ZOOM_STEP)

    # ------------------------------------------------------------ 右键菜单
    def _build_mode_menu(self) -> QMenu:
        """模式菜单：中文 / 中英对照（单选，当前模式打勾）。"""
        menu = QMenu(self)
        group = QActionGroup(menu)
        group.setExclusive(True)                 # 打勾互斥，只亮当前一项
        for mode, label in MODE_MENU:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(mode == self._mode)
            act.setData(mode)
            group.addAction(act)
        menu.triggered.connect(self._on_mode_action)
        return menu

    def _on_mode_action(self, act) -> None:
        mode = act.data()
        if isinstance(mode, str) and mode:
            self.set_mode(mode)                  # 切换 + 持久化

    def contextMenuEvent(self, e):
        """右键弹菜单选显示模式（落点处弹出；子控件的事件由 eventFilter 转来）。"""
        menu = self._build_mode_menu()
        menu.exec(e.globalPos())
        menu.deleteLater()
        e.accept()

    # ------------------------------------------------------------ 窗口事件
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "_entries", None) is not None:
            self._apply_fonts()
        self._position_zoom_label()
        self._position_recording_pill()

    def showEvent(self, e):
        super().showEvent(e)
        self._apply_fonts()
        self._position_recording_pill()
        self.setFocus()

    def wheelEvent(self, e):
        if self._handle_wheel(e):
            return
        super().wheelEvent(e)

    def _handle_wheel(self, e) -> bool:
        """滚轮调倍率；Ctrl+滚轮交给滚动区翻历史。返回 True 表示已消费。"""
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        d = e.angleDelta().y() or e.pixelDelta().y()
        if not d:
            return False
        self._zoom_step(1 if d > 0 else -1)
        e.accept()
        return True

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_step(1)
        elif e.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._zoom_step(-1)
        else:
            super().keyPressEvent(e)
            return
        e.accept()

    # ------------------------------------------------------- 边缘缩放 / 拖动
    def _hit_test(self, pos: QPoint) -> str | None:
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        if not (0 <= x < w and 0 <= y < h):
            return None
        left, right = x < EDGE, x >= w - EDGE
        top, bottom = y < EDGE, y >= h - EDGE
        if top and left:
            return "tl"
        if top and right:
            return "tr"
        if bottom and left:
            return "bl"
        if bottom and right:
            return "br"
        if left:
            return "l"
        if right:
            return "r"
        if top:
            return "t"
        if bottom:
            return "b"
        return None

    def _update_cursor(self, pos: QPoint) -> None:
        edge = self._hit_test(pos)
        if edge:
            self.setCursor(_CURSORS[edge])
        else:
            self.unsetCursor()

    def _begin_interaction(self, local: QPoint, global_pos: QPoint) -> None:
        self._press_global = global_pos
        edge = self._hit_test(local)
        if edge:
            self._op = "resize"
            self._grab_edge = edge
            self._press_geom = self.geometry()
        else:
            self._op = "move"
            self._grab_edge = None
            self._move_offset = global_pos - self.frameGeometry().topLeft()

    def _continue_interaction(self, global_pos: QPoint) -> None:
        if self._op is None:
            return
        delta = global_pos - self._press_global
        if self._op == "move":
            self.move(global_pos - self._move_offset)
            return
        g = self._grab_edge or ""
        pg = self._press_geom
        l, t, r, b = pg.x(), pg.y(), pg.x() + pg.width(), pg.y() + pg.height()
        if "l" in g:
            l = min(l + delta.x(), r - MIN_W)
        if "r" in g:
            r = max(r + delta.x(), l + MIN_W)
        if "t" in g:
            t = min(t + delta.y(), b - MIN_H)
        if "b" in g:
            b = max(b + delta.y(), t + MIN_H)
        self.setGeometry(l, t, r - l, b - t)

    def _end_interaction(self) -> None:
        self._op = None
        self._grab_edge = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            g = e.globalPosition().toPoint()
            self._begin_interaction(self.mapFromGlobal(g), g)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # 位置一律用 globalPosition 映射：子控件冒泡上来的事件里 position() 已被
        # 按冒泡链翻译过（实测落在 (24,22) 这种错值），会让光标/命中判定错乱
        g = e.globalPosition().toPoint()
        if self._op is not None:
            self._continue_interaction(g)
            e.accept()
            return
        self._update_cursor(self.mapFromGlobal(g))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._op is not None:
            self._end_interaction()
            self._update_cursor(self.mapFromGlobal(e.globalPosition().toPoint()))
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # ---------------------------------------------------------------- 事件过滤
    def eventFilter(self, obj, ev):
        """子控件上的鼠标/滚轮/右键转发到窗口：整窗都能拖动、缩放、调字号、切模式。"""
        t = ev.type()
        if t == QEvent.Type.ContextMenu:
            # 子控件（滚动区/视口/容器/块）会自己吃掉右键 → 统一转到窗口弹模式菜单
            self.contextMenuEvent(ev)
            return True
        if t == QEvent.Type.Wheel:
            return True if self._handle_wheel(ev) else False
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove,
                 QEvent.Type.MouseButtonRelease):
            g = ev.globalPosition().toPoint()
            local = self.mapFromGlobal(g)
            if t == QEvent.Type.MouseButtonPress:
                if ev.button() != Qt.MouseButton.LeftButton:
                    return False
                self._begin_interaction(local, g)
                return True
            if t == QEvent.Type.MouseMove:
                if self._op is not None:
                    self._continue_interaction(g)
                    return True
                self._update_cursor(local)
                return False
            if self._op is not None:
                self._end_interaction()
                self._update_cursor(local)
                return True
        return super().eventFilter(obj, ev)
