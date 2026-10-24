# -*- coding: utf-8 -*-
"""主窗口：开始/结束录制、说话人改名、状态显示。"""
from __future__ import annotations

import dataclasses
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from string import Template

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout,
                             QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu,
                             QMessageBox,
                             QPushButton, QSizePolicy, QSystemTrayIcon,
                             QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from .capture import AudioCapture, CaptureError
from .config import (PROJECT_ROOT, ConfigError, Settings, load_settings,
                     secrets_source)
from .export import export_all, meeting_dir, write_transcript
from .subtitle import SubtitleWindow
from .speaker_dialog import SpeakerLabelDialog
from .transcript import (Journal, TranscriptStore, journal_finished, journal_meta,
                         load_journal, write_footer)
from .translator import Translator
# 声纹库只依赖 numpy（几十毫秒）：可以在顶层 import；真正的 pyannote 模型在
# voiceprints 内部惰性加载，不拖慢窗口出现。
from .voiceprints import VoiceprintLibrary, compute_speaker_embeddings


# 重型依赖延迟导入：顶层 import 会连带加载 pyannote/torch/faster-whisper，
# 实测让「双击快捷方式」到窗口出现的耗时从 ~1 秒涨到 13 秒以上（用户会以为没启动）。
# 这里留模块级占位名，首次真正需要时由 _ensure_heavy() 填充；测试 monkeypatch
# 这些名字（非 None）时不会被覆盖。
Recorder = None            # type: ignore[assignment]
Diarizer = None            # type: ignore[assignment]
assign_speakers = None     # type: ignore[assignment]
_heavy_lock = threading.Lock()


def _ensure_heavy(preload_diarize: bool = True) -> None:
    """填充重型依赖（Recorder → faster-whisper/silero；Diarizer → pyannote/torch）。

    实测 ``import recorder`` 要 6 秒左右（torch/faster-whisper/silero），所以窗口
    出现后会先调一次预热（见 MainWindow._preload_engine），点开始时不再等这 6 秒。
    """
    global Recorder, Diarizer, assign_speakers
    with _heavy_lock:
        if Recorder is None:
            from .recorder import Recorder as _Recorder
            Recorder = _Recorder
        if preload_diarize:
            if Diarizer is None:
                from .diarization import Diarizer as _Diarizer
                Diarizer = _Diarizer
            if assign_speakers is None:
                from .diarization import assign_speakers as _assign_speakers
                assign_speakers = _assign_speakers


def _resolve_diarize_tools():
    """取 (Diarizer 类, assign_speakers 函数)，必要时惰性加载 .diarization。

    只加载 diarization 模块（torch/pyannote），不连带 recorder/faster-whisper ——
    会后标注发生在录制结束之后，不该顺带把识别模型再拖进来。
    """
    global Diarizer, assign_speakers
    if Diarizer is None or assign_speakers is None:
        from .diarization import Diarizer as _Diarizer
        from .diarization import assign_speakers as _assign_speakers
        if Diarizer is None:
            Diarizer = _Diarizer
        if assign_speakers is None:
            assign_speakers = _assign_speakers
    return Diarizer, assign_speakers


def _read_audio(path: Path):
    """读整场录音 wav → (float32 单声道波形, 采样率)。

    soundfile 惰性导入：只在会后标注时加载，不拖慢窗口启动。
    """
    import soundfile as sf
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:      # 意外立体声：取左声道（录制侧只写单声道）
        audio = audio[:, 0]
    return audio, int(sample_rate)



def probe_groq(api_key: str, timeout: float = 6.0) -> tuple[str, str, bool]:
    """探测 Groq 是否可用：返回 (state, 文案, 是否可重试)。

    必须走**和真实识别完全相同的 SDK 链路**（openai + httpx）：早期用 urllib 探测，
    它发的 User-Agent 是 "Python-urllib/3.x"，会被 Groq 的 WAF 当成爬虫直接 403，
    而真正识别（openai SDK）完全正常 —— 于是自检误报「地区限制，请开代理」。

    - 成功 → ("ok", 可用)
    - 401  → key 无效（确定性错误，重试无意义）
    - 403  → 真的被拒绝（地区/权限）
    - 其它 → 网络/服务问题（可重试）
    """
    import openai

    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    try:
        client.models.list()          # 最轻量的鉴权+连通性检查
        return "ok", "云端识别（Groq）可用，字幕延迟约 3~5 秒。", False
    except openai.AuthenticationError:
        return ("error",
                "GROQ_API_KEY 无效或已过期：当前使用本地识别。"
                "到 console.groq.com 重新生成 key 并更新配置。", False)
    except openai.PermissionDeniedError:
        return ("warn",
                "云端识别被拒绝（HTTP 403）：如果开了代理，请改用全局/TUN 模式；"
                "否则当前回退本地识别（延迟 30~55 秒）。", True)
    except Exception as e:            # 网络/超时/5xx
        return ("warn",
                f"无法连接云端识别服务（{type(e).__name__}）："
                "请检查网络或代理；当前回退本地识别（延迟 30~55 秒）。", True)


MINING_PROMPT = """你是会议术语挖掘助手。从下面的会议逐字稿中，提取可能影响语音识别
准确率的术语：产品名、公司/客户名、人名、行业缩写、专有名词。
要求：1) 每个术语一行，只输出术语本身，不要编号/解释/空行；2) 2~16 个字符；
3) 中英文都行（英文缩写保留原样）；4) 宁缺毋滥，没有就输出空。
逐字稿：
{transcript}"""


def mine_terms(translator, transcript_md: str) -> list[str]:
    """LLM 读逐字稿 → 术语列表（已去重、清洗）。任何失败返回 []。"""
    from .translator import TranslationError
    try:
        raw = translator._chat([
            {"role": "system", "content": "你是会议术语挖掘助手，只输出术语列表。"},
            {"role": "user", "content": MINING_PROMPT.format(
                transcript=transcript_md[-12000:])},
        ], max_tokens=600)
    except (TranslationError, Exception):
        return []
    seen, out = set(), []
    for line in raw.splitlines():
        term = line.strip().strip("-·*1234567890. )(")
        if 2 <= len(term) <= 16 and term not in seen:
            seen.add(term)
            out.append(term)
    return out[:50]


def _fmt_hms(seconds: float) -> str:
    """秒 → 00:00:00（标记面板显示声纹首末时间）。"""
    s_int = max(0, int(seconds))
    return f"{s_int // 3600:02d}:{s_int % 3600 // 60:02d}:{s_int % 60:02d}"


def app_icon() -> QIcon:
    """应用图标：窗口与桌面快捷方式共用 assets/app.ico。"""
    return QIcon(str(PROJECT_ROOT / "assets" / "app.ico"))


# ---------------------------------------------------------------------------
# 暗色主题（与 assets/app.ico 的蓝青渐变呼应）
# ---------------------------------------------------------------------------
BG = "#1B1E2B"             # 主背景：深蓝黑
SURFACE = "#232839"        # 卡片 / 输入框 / 表格底
BORDER = "#343B4F"         # 常规描边
BORDER_STRONG = "#5A6478"  # 次按钮描边
TEXT = "#E8EAF2"           # 正文
MUTED = "#8B93A7"          # 次要文字
ACCENT_A = "#2A7DE1"       # 主色（渐变起点 / 选中色）
ACCENT_B = "#1FB6B8"       # 主色（渐变终点）
DISABLED = "#3A4052"       # 禁用灰化
PILL_BG = "#2A2F42"        # 配置胶囊底
PILL_FG = "#A8C7F0"        # 配置胶囊字
FONT_FAMILY = "Microsoft YaHei UI"

RECORDING_DIR = ".recording"     # <output_dir>/.recording/：边录边落盘的工作区（git 忽略）

# 会后离线说话人标注：初版产物已出，标注在后台补，用户可以继续用电脑
OFFLINE_DIARIZE_STATUS = ("正在会后标注说话人"
                          "（约 0.5~1.5 倍会议时长，可继续用电脑）")
OFFLINE_DIARIZED_DONE = "说话人标注完成，已更新逐字稿"
OFFLINE_NO_AUDIO = "[提示] 未保存录音（SAVE_AUDIO=false），跳过会后说话人标注"
# 声纹计算的硬上限：模型要下载 / 网络卡住 / CPU 太慢时，绝不能拖着不导出产物。
# 冷启动（import torch + 加载 wespeaker + 每人 ≤60 秒音频）实测 20 秒内，留足余量。
VOICEPRINT_TIMEOUT = 120.0

WINDOW_TITLE = "会议翻译助手"
WINDOW_TITLE_RECORDING = "会议翻译助手 · 录音中"
# 密钥仍是 .env 明文时启动给一条轻提示（不弹窗、不打断）
PLAINTEXT_SECRETS_HINT = ("提示：密钥存放在 .env 明文文件中，"
                          "可运行 setup_env.py 迁到 Windows 凭据管理器")

# 托盘常驻（CLOSE_TO_TRAY=true，默认）：关闭窗口只是把界面收起来 —— 录音、会后
# 整理、说话人标注继续在后台跑；真正的退出只在托盘菜单里（见 MainWindow._tray_quit）。
TRAY_TOOLTIP = "会议翻译助手"
TRAY_RECORDING_HINT = "录音继续中，点托盘图标可恢复窗口"
TRAY_QUIT_CONFIRM = ("退出将中止正在进行的后台任务（录音/标注/纪要生成）。"
                     "确定退出吗？")
# 托盘退出时等 recorder.stop() 收尾的上限（秒）：等不到就放弃（收尾线程是 daemon）
QUIT_STOP_TIMEOUT = 10.0

_AUDIO_PILL = {
    "both": "音频 · 自动双路",
    "speaker": "音频 · 仅系统声音",
    "mic": "音频 · 仅麦克风",
}

DARK_QSS = Template("""
QMainWindow, QWidget#central {
    background-color: $BG;
}
QWidget {
    color: $TEXT;
    font-family: "$FONT";
    font-size: 10.5pt;
}
QLabel {
    background: transparent;
    color: $TEXT;
}
QLabel#brand_title {
    font-size: 16pt;
    font-weight: bold;
}
QLabel#brand_sub {
    font-size: 9pt;
    color: $MUTED;
}
QLabel[role="pill"] {
    background-color: $PILL_BG;
    color: $PILL_FG;
    border-radius: 10px;
    padding: 4px 10px;
    font-size: 9pt;
}
QLabel#status {
    font-weight: bold;
    color: #FFFFFF;
}
QLabel#elapsed {
    font-size: 9pt;
    color: $MUTED;
}
QLabel#cloud_banner {
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 9.5pt;
}
QLabel#cloud_banner[state="warn"] {
    background-color: #3A2F1E;
    color: #FFCF7A;
    border: 1px solid #6B5628;
}
QLabel#cloud_banner[state="error"] {
    background-color: #3A2024;
    color: #FF9AA2;
    border: 1px solid #6B2F36;
}
QLabel#cloud_banner[state="ok"] {
    background-color: #1E3A2E;
    color: #8FE3B0;
    border: 1px solid #2C5A45;
}
QLabel#footer_hint {
    font-size: 8.5pt;
    color: $MUTED;
}
QFrame#card {
    background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 8px;
}
QLineEdit {
    background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 10px 12px;
    color: $TEXT;
    selection-background-color: $ACCENT_A;
    selection-color: #FFFFFF;
}
QLineEdit:focus {
    border: 1px solid $ACCENT_A;
}
QPushButton[role="primary"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 $ACCENT_A, stop:1 $ACCENT_B);
    border: none;
    border-radius: 8px;
    padding: 12px;
    color: #FFFFFF;
    font-weight: bold;
}
QPushButton[role="primary"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3B8DEE, stop:1 #2FC9CB);
}
QPushButton[role="primary"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2368BC, stop:1 #18999B);
}
QPushButton[role="primary"]:disabled {
    background: $DISABLED;
    color: #79819A;
}
QPushButton[role="secondary"] {
    background: transparent;
    border: 1px solid $BORDER_STRONG;
    border-radius: 8px;
    padding: 12px;
    color: $TEXT;
}
QPushButton[role="secondary"]:hover {
    border: 1px solid $ACCENT_B;
    color: #FFFFFF;
}
QPushButton[role="secondary"]:pressed {
    border: 1px solid $ACCENT_A;
    background-color: rgba(42, 125, 225, 0.14);
}
QPushButton[role="secondary"]:disabled {
    border: 1px solid $DISABLED;
    color: #6C7488;
}
QTableWidget {
    background-color: $SURFACE;
    alternate-background-color: #262C3D;
    border: 1px solid $BORDER;
    border-radius: 6px;
    gridline-color: $BORDER;
    color: $TEXT;
    selection-background-color: $ACCENT_A;
    selection-color: #FFFFFF;
}
QTableWidget::item {
    padding: 4px 8px;
}
QTableWidget::item:selected {
    background-color: $ACCENT_A;
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #1F2433;
    border: none;
    border-bottom: 1px solid $BORDER;
    color: $MUTED;
    padding: 6px 8px;
    font-weight: bold;
}
QTableCornerButton::section {
    background-color: #1F2433;
    border: none;
}
QMenu {
    background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 18px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: $ACCENT_A;
    color: #FFFFFF;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: $BORDER_STRONG;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
""").substitute(BG=BG, SURFACE=SURFACE, BORDER=BORDER, BORDER_STRONG=BORDER_STRONG,
                TEXT=TEXT, MUTED=MUTED, ACCENT_A=ACCENT_A, ACCENT_B=ACCENT_B,
                DISABLED=DISABLED, PILL_BG=PILL_BG, PILL_FG=PILL_FG, FONT=FONT_FAMILY)


# 术语分隔符：中英文逗号、顿号、分号都认（用户随手敲的，别要求记格式）
TERM_SEPARATORS = re.compile(r"[,，、;；]+")


def merge_asr_prompt(base: str, terms: str) -> str:
    """本场术语 + .env 的 ASR_PROMPT → 识别提示词（术语在前，逗号连接）。

    - 按中英文逗号/顿号/分号切分、去首尾空白、丢空项；
    - 大小写不敏感去重（用户重复写、或写了 .env 里已有的词时只留一次，术语在前优先）；
    - **术语为空时原样返回 base**（行为与线上完全一致，不多加一个空格）；
    - base 用逗号重新拼接只在有术语时才发生：既保证顺序稳定，也不会让空术语改变提示词。
    """
    user_terms = [t.strip() for t in TERM_SEPARATORS.split(terms or "") if t.strip()]
    if not user_terms:
        return base or ""
    merged: list[str] = []
    seen: set[str] = set()
    for term in user_terms + [t.strip() for t in (base or "").split(",") if t.strip()]:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(term)
    return ", ".join(merged)


def _secrets_source_is_env() -> bool:
    """密钥是否来自 .env 明文（读取失败一律当作不是，绝不因提示而崩溃）。"""
    try:
        return secrets_source() == ".env"
    except Exception:
        return False


def asr_pill_text(settings: Settings) -> str:
    """识别徽章：groq → 云端；其余（local）→ 本地 + whisper 模型名。"""
    provider = (getattr(settings, "asr_provider", "local") or "local").strip().lower()
    if provider == "groq":
        return "识别 · Groq 云端"
    model = (getattr(settings, "whisper_model", "") or "small").strip()
    return f"识别 · 本地 {model}"


def audio_pill_text(settings: Settings) -> str:
    """音频徽章：both / speaker / mic → 中文可读文案（未知值原样透出）。"""
    source = (getattr(settings, "audio_source", "both") or "both").strip().lower()
    return _AUDIO_PILL.get(source, f"音频 · {source}")


def translate_pill_text(settings: Settings) -> str:
    """翻译徽章：展示模型名，去掉 deepseek- 前缀（胶囊空间有限）。"""
    model = (getattr(settings, "translate_model", "") or "未配置").strip()
    if model.lower().startswith("deepseek-"):
        model = model[len("deepseek-"):]
    return f"翻译 · {model}"


def _notice_line(msg: str) -> str:
    """托盘气泡正文：取「第一行带路径的产物信息」，没有就取第一行。

    气泡只有几十个字宽，多行提示会被截断；产物路径（逐字稿/纪要）是用户最需要
    看到的（知道去哪儿找文件），所以优先挑带路径分隔符的那一行。
    """
    lines = [ln.strip() for ln in (msg or "").splitlines() if ln.strip()]
    for line in lines:
        if "/" in line or "\\" in line:
            return line
    return lines[0] if lines else ""


def _pill(text: str) -> QLabel:
    """配置胶囊：只读小徽章（样式见 DARK_QSS 的 QLabel[role="pill"]）。"""
    lab = QLabel(text)
    lab.setProperty("role", "pill")
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lab


def brand_pixmap(size: int = 48) -> QPixmap:
    """品牌行图标：assets/app.ico 的 48px 缩略。

    注意高 DPI：QIcon.pixmap(size) 内部已经按屏幕缩放返回 ``size * dpr`` 的物理像素
    并设好 devicePixelRatio，这里**不能再手动乘 dpr** —— 否则逻辑尺寸翻倍，
    48px 的 QLabel 只能显示左上 1/4（曾在 2x 缩放屏上实测到此问题）。
    """
    return app_icon().pixmap(size, size)


class RecorderBridge(QObject):
    """把工作线程的事件桥接到 Qt 主线程（线程安全）。"""
    segment = pyqtSignal(object)
    # 译文填充后的原地更新（两阶段显示第二阶段）：与 segment 分开，
    # 面板按 seg.seq 找到原块替换，不新增块。
    update = pyqtSignal(object)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    # 崩溃恢复结果（后台线程 → GUI 线程；恢复期间不能阻塞界面）
    recovered = pyqtSignal(bool, str)
    # 会后离线说话人标注结果（后台线程 → GUI 线程）：ok, 提示文案
    diarized = pyqtSignal(bool, str)
    # 声纹库认出的「熟人」建议（后台线程 → GUI 线程）：{label: (姓名, 分数)}
    # 单独一个信号而不是塞进 diarized：认人要弹模态框问用户，必须由主线程做
    speaker_suggestions = pyqtSignal(object)
    # 云端识别自检结果（state, 提示文案, 是否可重试）
    cloud_status = pyqtSignal(str, str, bool)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("会议翻译助手")
        self.recorder: Recorder | None = None
        # 显示模式默认取 .env 的 SUBTITLE_MODE；用户右键切换过的值由窗口内
        # QSettings 记住并优先使用（见 SubtitleWindow.__init__）
        self.subtitle = SubtitleWindow(
            getattr(self.settings, "subtitle_mode", "bilingual"))
        self.bridge = RecorderBridge()
        self.bridge.segment.connect(self._on_segment)
        self.bridge.update.connect(self._on_update)
        self.bridge.status.connect(self._on_status)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.recovered.connect(self._on_recovered)
        self.bridge.diarized.connect(self._on_diarized)
        self.bridge.speaker_suggestions.connect(self._on_speaker_suggestions)
        self.bridge.cloud_status.connect(self._on_cloud_status)
        # 声纹库（本地 声纹库.json）：会后认「熟人」，改名即学习。IO 异常由库自己吞掉。
        self._voiceprints = VoiceprintLibrary()
        self._reset_voiceprints()
        # 关窗触发收尾时置 True：导出完成后由 _on_finished 直接退出应用
        # （只有 CLOSE_TO_TRAY=false 的旧语义会走到这里）
        self._quit_after_stop = False
        # 托盘「退出」已确认：排队中的 finished/diarized 回槽不再弹窗、不再发气泡
        self._quitting = False
        # 「录制继续中」的托盘气泡只发一次（窗口恢复后重置，下次收起还能再提示）
        self._tray_hint_shown = False

        central = QWidget()
        central.setObjectName("central")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(12)

        # ---- 品牌行：48px 图标 + 「会议翻译助手」标题 / 副标题 ----
        brand = QHBoxLayout()
        brand.setSpacing(12)
        self.icon_label = QLabel()
        self.icon_label.setObjectName("brand_icon")
        self.icon_label.setFixedSize(48, 48)
        pix = brand_pixmap(48)
        if not pix.isNull():
            self.icon_label.setPixmap(pix)
        brand.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)

        head = QVBoxLayout()
        head.setSpacing(2)
        self.brand_title = QLabel("会议翻译助手")
        self.brand_title.setObjectName("brand_title")
        self.brand_sub = QLabel("实时翻译字幕 · 会后自动纪要")
        self.brand_sub.setObjectName("brand_sub")
        head.addWidget(self.brand_title)
        head.addWidget(self.brand_sub)
        brand.addLayout(head, 0)
        brand.addStretch(1)
        lay.addLayout(brand)

        # ---- 配置胶囊：三个只读徽章（识别 / 音频 / 翻译），随 .env 变化 ----
        pills = QHBoxLayout()
        pills.setSpacing(8)
        self.pill_asr = _pill(asr_pill_text(self.settings))
        self.pill_audio = _pill(audio_pill_text(self.settings))
        self.pill_translate = _pill(translate_pill_text(self.settings))
        for pill in (self.pill_asr, self.pill_audio, self.pill_translate):
            pills.addWidget(pill)
        pills.addStretch(1)
        lay.addLayout(pills)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("会议名称（用于文件命名，可留空）")
        # 默认用「年-月-日 时:分」预填；用户没手动改过时，点开始时刷新成实际开始时间
        self._title_edited = False
        self.title_input.textEdited.connect(self._mark_title_edited)
        self.title_input.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        lay.addWidget(self.title_input)

        # 本场术语/热词：只影响本次录制的识别提示词（不写回 .env，下次启动为空）
        self.terms_input = QLineEdit()
        self.terms_input.setPlaceholderText(
            "术语/热词（可选，逗号分隔，例：过敏门诊, 复合药房, Jebel Ali）")
        lay.addWidget(self.terms_input)

        # ---- 按钮行：主（渐变）/ 次（描边），等高 44px ----
        row = QHBoxLayout()
        row.setSpacing(10)
        self.start_btn = QPushButton("开始记录")
        self.start_btn.setProperty("role", "primary")
        self.start_btn.setFixedHeight(44)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn = QPushButton("结束并生成纪要")
        self.stop_btn.setProperty("role", "secondary")
        self.stop_btn.setFixedHeight(44)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        # 「标注说话人」独立动作：会后（或重启后对上一场）随时可重跑，可指定人数
        self.label_btn = QPushButton("标注说话人")
        self.label_btn.setProperty("role", "secondary")
        self.label_btn.setFixedHeight(44)
        self.label_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.label_btn.setToolTip("对本场录音重新跑说话人标注（可指定人数提高准确率）")
        self.label_btn.setEnabled(False)
        row.addWidget(self.start_btn, 1)
        row.addWidget(self.stop_btn, 1)
        row.addWidget(self.label_btn, 0)
        lay.addLayout(row)

        # ---- 状态卡片：状态文本（加粗白）+ 录制计时 ----
        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(14, 12, 14, 12)
        card_lay.setSpacing(6)
        # 密钥来源轻提示：凭据管理器优先；仍是 .env 明文时只在启动状态栏说一句
        self._secrets_hint = (PLAINTEXT_SECRETS_HINT
                              if _secrets_source_is_env() else "")
        status_text = "就绪"
        if self._secrets_hint:
            status_text = chr(10).join((status_text, self._secrets_hint))
        self.status = QLabel(status_text)
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.elapsed_label = QLabel("就绪")
        self.elapsed_label.setObjectName("elapsed")
        card_lay.addWidget(self.status)
        card_lay.addWidget(self.elapsed_label)
        lay.addWidget(card)

        # ---- 云端识别自检提示条：启动时后台探测 Groq 可用性，结果在这里常驻提示 ----
        self.cloud_banner = QLabel("")
        self.cloud_banner.setObjectName("cloud_banner")
        self.cloud_banner.setWordWrap(True)
        self.cloud_banner.setVisible(False)
        self.cloud_banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cloud_row = QHBoxLayout()
        cloud_row.setSpacing(8)
        cloud_row.addWidget(self.cloud_banner, 1)
        self.recheck_btn = QPushButton("重新检测")
        self.recheck_btn.setProperty("role", "secondary")
        # 不写死高度：次按钮 QSS 的 padding 是 12px，sizeHint 高 40px，
        # 写死 30px 会把「重新检测」四个字竖向裁掉（实测过）
        self.recheck_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.recheck_btn.setVisible(False)
        self.recheck_btn.clicked.connect(self._recheck_cloud)
        cloud_row.addWidget(self.recheck_btn, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(cloud_row)

        self.speaker_hint = QLabel(
            "说话人命名：会议结束后点「标注说话人」→ 这里会出现 SPEAKER_00/01 →\n"
            "双击「名字」列输入姓名即可（整场回填 + 存入声纹库，下次自动认人）")
        self.speaker_hint.setObjectName("footer_hint")
        self.speaker_hint.setWordWrap(True)
        lay.addWidget(self.speaker_hint)

        self.speaker_table = QTableWidget(0, 2)
        self.speaker_table.setHorizontalHeaderLabels(["声纹标签", "名字"])
        self.speaker_table.setAlternatingRowColors(True)
        self.speaker_table.setShowGrid(False)
        self.speaker_table.verticalHeader().setVisible(False)
        self.speaker_table.horizontalHeader().setStretchLastSection(True)
        self.speaker_table.setMinimumHeight(140)
        lay.addWidget(self.speaker_table, 1)

        self.footer_hint = QLabel(
            "关闭窗口会收进托盘，录音 / 会后整理继续在后台进行；"
            "托盘图标可恢复窗口或退出")
        self.footer_hint.setObjectName("footer_hint")
        self.footer_hint.setWordWrap(True)
        lay.addWidget(self.footer_hint)

        self.setCentralWidget(central)
        self.setStyleSheet(DARK_QSS)
        self.resize(560, 600)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.label_btn.clicked.connect(self._label_speakers)
        self.speaker_table.itemChanged.connect(self._rename_speaker)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_speakers)
        self._timer.start(5000)

        # 录制计时：每秒刷新「已录制 mm:ss · 语句 N 条」（仅录制中刷新，空闲显示「就绪」）
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed)
        self._elapsed_timer.start(1000)
        self._refresh_elapsed()

        self.setWindowIcon(app_icon())
        # 后台预热引擎：窗口已出现，用户还在看界面时就把 6 秒的 import 做完
        threading.Thread(target=self._preload_engine, daemon=True).start()
        # 崩溃恢复：只在「启动那一刻」快照待恢复日志，再推迟到下一轮事件循环询问。
        # 若把扫描放到定时器回调里，用户很快点了「开始记录」时，本场刚建的日志会被
        # 误判成「上次未完成会议」并弹模态框（实测会卡住自动化测试）。
        self._pending_journals = self._scan_pending_journals()
        QTimer.singleShot(0, self._check_recovery)
        # 云端自检：Groq 在国内直连常被 403 地区封禁 → 界面常驻提醒开代理
        QTimer.singleShot(300, self._start_cloud_check)
        # 上一场如果没标注（或上次提前关程序），启动即允许补标注
        QTimer.singleShot(400, self._enable_label_for_last_recording)
        # 托盘放在最后：菜单槽会读写 start_btn / stop_btn / subtitle，必须等它们就绪
        self._setup_tray()

    # ------------------------------------------------------------------
    # 系统托盘：关窗 ≠ 退出（后台继续录音 / 整理 / 标注）
    # ------------------------------------------------------------------

    def _setup_tray(self) -> None:
        """系统托盘：把「关窗」与「退出」解耦（关窗只收起界面，后台任务照旧跑）。"""
        self.tray_menu = QMenu(self)
        self.tray_show_action = self.tray_menu.addAction("打开主窗口")
        self.tray_show_action.triggered.connect(self._show_window)
        self.tray_stop_action = self.tray_menu.addAction("结束并生成纪要")
        self.tray_stop_action.triggered.connect(self._tray_stop)
        # 「标注说话人」由后续任务实现：实现好 _label_speakers_last_session 即自动接线，
        # 还没实现就置灰（点一个什么都不做的菜单项比置灰更糟）
        self.tray_label_action = self.tray_menu.addAction("标注说话人")
        label_handler = getattr(self, "_label_speakers_last_session", None)
        if callable(label_handler):
            self.tray_label_action.triggered.connect(label_handler)
        else:
            self.tray_label_action.setEnabled(False)
        self.tray_quit_action = self.tray_menu.addAction("退出")
        self.tray_quit_action.triggered.connect(self._tray_quit)

        # 图标用同一份 ico：窗口 / 托盘 / 快捷方式视觉一致
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip(TRAY_TOOLTIP)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason) -> None:
        """单击 / 双击托盘图标 → 还原主窗口（收进托盘后点得回来）。"""
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self._show_window()

    def _show_window(self) -> None:
        """把窗口从托盘里放回来（置前并激活）。"""
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def showEvent(self, e) -> None:
        """窗口重新出现（点托盘 / 点任务栏）：重新武装「录制继续中」的托盘提示。"""
        self._tray_hint_shown = False
        super().showEvent(e)

    def _tray_notify(self, title: str, msg: str) -> None:
        """托盘气泡：窗口收起来之后唯一「看得见」的提示通道（不可用则静默跳过）。"""
        tray = getattr(self, "tray", None)
        if tray is None or not msg or self._quitting:
            return
        try:
            if not tray.isVisible():      # 无系统托盘（远程会话 / 桌面 shell 重启）
                return
            tray.showMessage(title, msg)
        except Exception:
            pass          # 托盘异常也不能影响主流程：界面状态栏仍会更新

    def _hide_tray(self) -> None:
        """收掉托盘图标：避免进程退出后残留「幽灵图标」。"""
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.hide()

    def _is_recording(self) -> bool:
        """录制 / 收尾中：start 按钮禁用（_start 置位、_on_finished 复位）。"""
        return self.recorder is not None and not self.start_btn.isEnabled()

    def _refresh_elapsed(self) -> str:
        """录制计时行：录制中 → 「已录制 12:34 · 语句 23 条」，空闲 → 「就绪」。

        时间从 ``recorder.started_at`` 算（不是窗口创建时间），语句数取
        ``recorder.store.segments()`` 的长度。只由 QTimer 在 GUI 线程调用。
        """
        label = getattr(self, "elapsed_label", None)
        recorder = self.recorder
        if recorder is None or self.start_btn.isEnabled():
            text = "就绪"
        else:
            started = getattr(recorder, "started_at", None)
            seconds = 0 if started is None else max(
                0, int((datetime.now() - started).total_seconds()))
            store = getattr(recorder, "store", None)
            count = len(store.segments()) if store is not None else 0
            text = f"已录制 {seconds // 60:02d}:{seconds % 60:02d} · 语句 {count} 条"
        if label is not None:
            label.setText(text)
        return text

    def _tray_stop(self) -> None:
        """托盘「结束并生成纪要」：复用 _stop() 的后台收尾；空闲时无事可做。

        收尾中 stop_btn 已禁用 —— 再点一次不能再起一个 stop()（会重复补译 + 重复导出）。
        """
        if self._is_recording() and self.stop_btn.isEnabled():
            self._stop()

    def _tray_quit(self) -> None:
        """托盘「退出」：唯一真正的退出入口（先确认 → 收录音 → 收托盘 → quit）。

        退出会中止后台任务（录音 / 标注 / 纪要生成），所以必须显式确认；确认后先
        给录音一个收尾机会（最多 QUIT_STOP_TIMEOUT 秒），再收掉托盘图标并退出。
        """
        ret = QMessageBox.question(
            self, "退出确认", TRAY_QUIT_CONFIRM,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes:
            return
        # 置位后：排队中的 finished / diarized 回槽不再弹窗、不再发气泡（正在退出）
        self._quitting = True
        if self._is_recording():
            self._stop_before_quit()
        self._quit_app()

    def _stop_before_quit(self) -> None:
        """退出前的录音收尾：stop() 放后台线程跑，GUI 最多等 QUIT_STOP_TIMEOUT 秒。

        等不到就放弃 —— 收尾线程与标注 / 纪要线程都是 daemon，退出不再被它们拖住
        （初版产物在导出成功时就已落盘，放弃的只是增强项）。
        """
        recorder = self.recorder
        if recorder is None:
            return

        def _safe_stop() -> None:
            try:
                recorder.stop()
            except Exception:
                pass          # 收尾失败也不能阻断退出（退出路径不能把用户锁死）

        worker = threading.Thread(target=_safe_stop, daemon=True)
        worker.start()
        worker.join(QUIT_STOP_TIMEOUT)

    def _quit_app(self) -> None:
        """退出应用：先收托盘 / 浮窗，避免残留图标与隐藏窗口把进程留住。"""
        self._hide_tray()
        self.subtitle.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    # ------------------------------------------------------------------
    # 崩溃恢复：边录边落盘（JSONL）+ 下次启动询问是否恢复
    # ------------------------------------------------------------------

    def _recording_dir(self) -> Path:
        """录制工作区：``<output_dir>/.recording/``（隐藏目录，已加进 .gitignore）。"""
        return Path(self.settings.output_dir) / RECORDING_DIR

    def _new_journal_store(self, title: str, session_id: str) -> TranscriptStore:
        """为本场录制新建「带日志的 store」。

        日志建不出来（磁盘只读 / 权限 / 目录异常）不得阻断录制：降级成无日志的
        store 照常录，只是丧失崩溃恢复能力（状态栏告警一次）。
        """
        try:
            path = self._recording_dir() / f"rec_{session_id}.jsonl"
            return TranscriptStore(journal=Journal(path, title=title or "未命名会议"))
        except OSError as e:
            self.bridge.status.emit(f"[警告] 无法创建录制日志（本次不支持崩溃恢复）：{e}")
            return TranscriptStore()

    def _finish_journal(self, store) -> None:
        """给本场日志写 footer 并关闭（导出成功后才调用）。

        只在导出成功后写：导出失败时保持「未完成」状态，下次启动还能再恢复一次。
        """
        journal = getattr(store, "journal", None)
        if journal is None:
            return
        try:
            journal.footer()
        except OSError:
            pass
        finally:
            journal.close()

    # ------------------------------------------------------------------
    # 云端识别自检（Groq）：启动时探测，结果常驻界面提示条
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 术语自动学习：会后挖掘 + 会前注入
    # ------------------------------------------------------------------

    def _glossary(self):
        """惰性创建术语库（首次访问才读文件）。"""
        lib = getattr(self, "_glossary_lib", None)
        if lib is None:
            from .glossary import GlossaryLibrary
            lib = GlossaryLibrary()
            self._glossary_lib = lib
        return lib

    def _auto_terms_text(self) -> str:
        """术语库 top N 拼成提示词片段（自动学习那部分；空库返回空串）。"""
        try:
            top = self._glossary().top(30)
        except Exception:
            return ""
        return "，".join(top)

    def _mine_terms_after_meeting(self, store, title: str) -> None:
        """会后后台挖掘术语：LLM 读逐字稿 → 候选术语 → 计入术语库（自动学习）。"""
        if not getattr(self.settings, "auto_terms", True):
            return
        translator = None
        session = getattr(self, "_last_session", None)
        if session is not None:
            translator = session.get("translator")
        if translator is None:
            return
        md = store.to_markdown(title, datetime.now())

        def work():
            try:
                candidates = mine_terms(translator, md)
                lib = self._glossary()
                lib.add_many(candidates)
                if lib.save() and candidates:
                    self.bridge.status.emit(
                        f"术语库已自动更新（+{len(candidates)} 个候选术语，下次会议自动启用）")
            except Exception:
                pass          # 学习失败不影响任何主流程

        threading.Thread(target=work, daemon=True).start()

    def _enable_label_for_last_recording(self) -> None:
        """启动时若发现上一场录音还没标注，启用「标注说话人」按钮。"""
        try:
            found = self._find_last_recording()
        except Exception:
            found = None
        if found and not self._is_recording():
            self.label_btn.setEnabled(True)
            self.label_btn.setToolTip(f"检测到上一场录音：{found[0].name}，可补标注")

    def _start_cloud_check(self) -> None:
        """后台探测云端识别可用性（不阻塞界面，约 1~5 秒）。"""
        threading.Thread(target=self._cloud_check_worker, daemon=True).start()

    def _cloud_check_worker(self) -> None:
        state, text, can_retry = self._probe_cloud()
        self.bridge.cloud_status.emit(state, text, can_retry)

    def _probe_cloud(self) -> tuple[str, str, bool]:
        """探测 Groq 可达性：返回 (state, 提示文案, 是否允许重试)。

        只在 groq 模式下探测；local 模式没有云端依赖，直接返回隐藏提示。
        """
        settings = self.settings
        if (getattr(settings, "asr_provider", "local") or "local") != "groq":
            return "hidden", "", False
        key = (getattr(settings, "groq_api_key", "") or "").strip()
        if not key:
            return ("error",
                    "未配置 GROQ_API_KEY：当前使用本地识别（较慢）。"
                    "想用低延迟云端识别：到 console.groq.com 免费注册，"
                    "把 key 填入配置后重启程序。", False)
        return probe_groq(key)

    def _on_cloud_status(self, state: str, text: str, can_retry: bool) -> None:
        """自检结果回到 GUI 线程：更新提示条（可重试时补一句操作提示）。"""
        if state == "hidden" or not text:
            self.cloud_banner.setVisible(False)
            self.recheck_btn.setVisible(False)
            return
        self.cloud_banner.setProperty("state", state)
        self.cloud_banner.setText(text)
        self.cloud_banner.setVisible(True)
        self.recheck_btn.setVisible(bool(can_retry))
        # 改了 state 属性必须重新 polish，QSS 才会套用新配色
        self.cloud_banner.style().unpolish(self.cloud_banner)
        self.cloud_banner.style().polish(self.cloud_banner)

    def _recheck_cloud(self) -> None:
        """「重新检测」入口：清掉旧提示后重新探测。"""
        self.cloud_banner.setVisible(False)
        self.recheck_btn.setVisible(False)
        self._start_cloud_check()

    def _scan_pending_journals(self) -> list[Path]:
        """启动时扫描「未完成」的录制日志（无 footer）。

        只统计**有实际内容**的日志（至少一条语句）：没有语句的残缺日志（点开始
        记录后立刻退出、或整场一句没识别出来）恢复出来也是空文件，弹窗询问只会
        让人困惑 → 直接归档成 .abandoned.jsonl。
        """
        pending: list[Path] = []
        try:
            for path in sorted(self._recording_dir().glob("rec_*.jsonl")):
                if not path.is_file() or journal_finished(path):
                    continue
                try:
                    has_segment = any('"t": "add"' in line or '"t":"add"' in line
                                      for line in path.read_text(encoding="utf-8").splitlines())
                except OSError:
                    continue
                if has_segment:
                    pending.append(path)
                else:
                    self._abandon_journal(path)
        except OSError:
            return []
        return pending

    def _check_recovery(self) -> None:
        """询问是否恢复启动快照里的未完成会议（只问一次，且不含本场录制）。"""
        pending = list(getattr(self, "_pending_journals", []))
        self._pending_journals = []
        if not pending:
            return
        path = pending[0]
        title, started_at = journal_meta(path)
        when = f"{started_at:%Y-%m-%d %H:%M}" if started_at else "时间未知"
        ret = QMessageBox.question(
            self, "检测到未完成的会议",
            f"检测到上次会议未正常结束（{title or path.stem} · {when}）。\n是否恢复并生成纪要？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes:
            self._abandon_journal(path)
            return
        self.status.setText(f"正在恢复上次会议（{title or path.stem}）…")
        # 回放 + 纪要生成可能几秒到几十秒（要调翻译接口）：放后台线程，GUI 不冻结
        threading.Thread(target=self._recover_journal, args=(path,), daemon=True).start()

    def _abandon_journal(self, path: Path) -> None:
        """用户选择不恢复：改名成 .abandoned.jsonl，之后启动不再提示（原文件保留可找回）。"""
        try:
            path.rename(path.with_suffix(".abandoned.jsonl"))
        except OSError as e:
            self.status.setText(f"[警告] 无法归档未完成日志：{e}")

    def _recover_journal(self, path: Path) -> None:
        """后台恢复：回放日志 → 重建 store → 导出逐字稿/纪要 → 写 footer。

        翻译接口不可用（未配置 / 构造失败）降级为占位纪要，逐字稿照常产出 ——
        恢复的价值主要在会议内容本身，不能因为纪要模型挂了就整场丢掉。
        """
        try:
            title, started_at, store = load_journal(path)
            try:
                translator = Translator(self.settings)
            except Exception:
                translator = None          # 降级：export_all 会写占位纪要
            transcript_path, minutes_path = export_all(
                store, self.settings, title or path.stem, started_at,
                translator=translator)
            write_footer(path)             # 恢复完成：标记已落盘，下次启动不再询问
            self.bridge.recovered.emit(
                True, f"已恢复上次会议：\n逐字稿：{transcript_path}\n纪要：{minutes_path}")
        except Exception as e:
            self.bridge.recovered.emit(False, f"恢复失败：{e}")

    def _on_recovered(self, ok: bool, msg: str) -> None:
        """恢复结果回到 GUI 线程：更新状态栏 + 提示（弹框只能在主线程）。"""
        self.status.setText("已恢复上次会议" if ok else "恢复失败")
        if ok:
            QMessageBox.information(self, "恢复完成", msg)
        else:
            QMessageBox.critical(self, "恢复失败", msg)

    def _probe_capture(self) -> AudioCapture | None:
        """按 .env 的 AUDIO_SOURCE（默认 both）探测一次设备，成功返回可复用的实例。

        构造 ``AudioCapture`` 只解析设备、不打开音频流，因此探测不会占住设备；
        实例直接交给 ``Recorder`` 复用，避免「录到一半才发现设备不可用」。
        设备不可用（被占用 / 无麦克风 / 无 WASAPI）→ 弹框并返回 ``None``。
        """
        source = getattr(self.settings, "audio_source", "both") or "both"
        try:
            return AudioCapture(source=source)
        except CaptureError as e:
            QMessageBox.critical(self, "启动失败", f"音频源不可用：{e}")
            return None

    def _start(self):
        """开始录制：设备探测留在 GUI 线程（快），重型引擎初始化 + 录制放后台线程。

        Recorder 首次导入要加载 faster-whisper/av/torch（实测 3~13 秒），在 GUI
        线程里做会让窗口「点了没反应」；这里只切按钮状态并提示，初始化在后台完成。
        """
        capture = self._probe_capture()
        if capture is None:
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_recording_state(True)
        self._reset_voiceprints()
        self.status.setText("正在准备识别引擎…")
        if not self._title_edited:
            self.title_input.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        title = self.title_input.text().strip()
        # 术语在 GUI 线程读完（控件只能在主线程碰），再换成「本场配置」：其余字段原样，
        # 只把 asr_prompt 换成合并结果。术语为空 → 直接沿用 self.settings（零行为变化）。
        terms = self.terms_input.text()
        base_prompt = getattr(self.settings, "asr_prompt", "") or ""
        merged_prompt = merge_asr_prompt(base_prompt, terms)
        if getattr(self.settings, "auto_terms", True):
            merged_prompt = merge_asr_prompt(merged_prompt, self._auto_terms_text())
        # 纠错表挂到本场配置上（Recorder 的 _apply_learned_corrections 从这取）
        from .glossary import CorrectionTable
        try:
            setattr(run_settings, "_correction_table", CorrectionTable())
        except Exception:
            pass
        run_settings = self.settings
        if merged_prompt != base_prompt:
            run_settings = dataclasses.replace(self.settings, asr_prompt=merged_prompt)

        def _boot():
            try:
                _ensure_heavy(run_settings.diarize_enabled)
                translator = Translator(run_settings)
            except Exception as e:
                self.bridge.finished.emit(False, f"启动失败：{e}")
                return
            diarizer = None
            if run_settings.diarize_enabled:
                try:
                    diarizer = Diarizer(run_settings)
                except Exception as e:
                    # 预期内（未配 HF_TOKEN）与非预期（模型下载/网络异常）一律降级继续：
                    # Qt 槽内异常外冒会直接终止进程（qFatal），不能让它冒出去。
                    self._warn_diarizer_unavailable(e)
            try:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                store = self._new_journal_store(title, session_id)
                self.recorder = Recorder(
                    run_settings, session_id=session_id, capture=capture,
                    translator=translator,
                    diarizer=diarizer,
                    audio_source=getattr(run_settings, "audio_source", "both"),
                    store=store)
                self.recorder.on_segment(self.bridge.segment.emit)
                self.recorder.on_update(self.bridge.update.emit)
                self.recorder.on_status(self.bridge.status.emit)
            except Exception as e:
                self.bridge.finished.emit(False, f"启动失败：{e}")
                return
            # 只记引用，audio/started_at 在 start() 之后才存在 → 用的时候再取
            self._last_session = {"store": store, "title": title,
                                  "recorder": self.recorder, "translator": translator}
            self.bridge.status.emit("开始录制（等待语音…）")
            self.recorder.start(title)

        threading.Thread(target=_boot, daemon=True).start()

    def _preload_engine(self) -> None:
        """后台预热重型依赖：把「点开始记录」的等待从 ~6 秒压到 ~1 秒。"""
        try:
            _ensure_heavy(self.settings.diarize_enabled)
            # 预热 silero VAD（加载 ONNX，约 0.2 秒）：只放在预热阶段，
            # 不进点击路径，否则测试里（依赖被打桩时）会白等模型加载
            from .vad import UtteranceBuffer
            UtteranceBuffer()
        except Exception:
            pass          # 预热失败不影响使用：首次点击时还会再试一次

    def _mark_title_edited(self, _text: str = "") -> None:
        """用户手动改过会议名：之后不再用时间自动覆盖。"""
        self._title_edited = True

    def _warn_diarizer_unavailable(self, e: Exception) -> None:
        QMessageBox.warning(self, "说话人功能不可用", f"{e}\n\n将继续记录，但无法区分说话人。")

    def _sync_subtitle_names(self) -> None:
        """把「名字解析器」接到字幕浮窗上（幂等）：块头行实时走 store.display_name。

        recorder 每场录制新建（store 随之新建），所以不能只在构造窗口时设一次：
        在每条语句到达时懒同步，改名时再同步一次。两处都在 GUI 线程，改 Qt
        属性安全。
        """
        store = getattr(self.recorder, "store", None)
        if store is not None:
            self.subtitle.name_resolver = store.display_name

    def _on_segment(self, seg):
        self._sync_subtitle_names()        # 新块的头行也按最新名字渲染
        name = self.recorder.store.display_name(seg.speaker)
        self.subtitle.show_segment(seg, name)

    def _on_update(self, seg):
        """译文到达（第二阶段）：交给面板按 seq 原地替换，不新增块。"""
        self._sync_subtitle_names()
        name = self.recorder.store.display_name(seg.speaker)
        self.subtitle.update_segment(seg, name)

    def _on_status(self, msg):
        self.status.setText(msg)

    def _refresh_speakers(self):
        # 命名指引：有标签就隐藏，没有就显示（放在最前，不受后面提前返回影响）
        if getattr(self, "speaker_hint", None) is not None:
            self.speaker_hint.setVisible(
                not self.recorder or not self.recorder.store.speakers())
        if not self.recorder:
            return
        # 用户正在单元格里打字时跳过本轮：重设 item 会把编辑框文本清空（实测
        # 输入 "张三" → 变空串），下一次提交就成了「把名字清掉」，还会把改名
        # 事件广播成空名字。宁可晚 5 秒刷新，也不能吃掉用户正在输入的字。
        if self.speaker_table.state() == QAbstractItemView.State.EditingState:
            return
        self.speaker_table.blockSignals(True)
        speakers = self.recorder.store.speakers()
        self.speaker_table.setRowCount(len(speakers))
        for i, sp in enumerate(speakers):
            self.speaker_table.setItem(i, 0, QTableWidgetItem(sp.label))
            self.speaker_table.setItem(i, 1, QTableWidgetItem(sp.name))
        self.speaker_table.blockSignals(False)

    def _rename_speaker(self, item):
        """表格里改名字：写 store → 回填整场已显示的字幕 → 广播给 recorder。

        三件事的顺序有讲究：先落 store（后面两步都以它为准），再重渲染字幕
        （用户立刻看到「整场都改成新名字」），最后广播改名事件 —— 那是人工确认
        过的声纹标签归属，声纹学习（A2）要拿它当正样本。
        """
        if item.column() != 1 or not self.recorder:
            return
        label_item = self.speaker_table.item(item.row(), 0)
        if label_item is None:
            return
        label, name = label_item.text(), item.text()
        store = self.recorder.store
        before = store.display_name(label)
        store.set_speaker_name(label, name)
        if store.display_name(label) == before:
            # itemChanged 也会因「程序性重设 / 只改格式」触发：名字没变就别白刷新，
            # 更别给声纹学习喂重复样本
            return
        # ① 已显示的块立刻换名字（只重渲染头行：正文、块数、滚动位置都不动）
        self._sync_subtitle_names()
        refresh = getattr(self.subtitle, "refresh_names", None)
        if callable(refresh):
            refresh()
        # ② 改名 = 「这个声纹标签就是这个人」，交给 recorder 广播（只广播，不做逻辑）
        emit = getattr(self.recorder, "emit_speaker_renamed", None)
        if callable(emit):
            emit(label, name)
        # ③ 声纹学习：新名字入库；若与本次自动匹配的建议不同，还要剔除那条误样本
        self._learn_rename(label, name)

    def _stop(self):
        if not self.recorder:
            return
        self.stop_btn.setEnabled(False)
        self.status.setText("正在收尾（补译 + 生成纪要）…")
        # Qt 控件只能在 GUI 线程访问：标题在主线程取值、recorder 取快照，
        # finish() 线程内不再触碰任何控件，也不再动态查 self.recorder。
        title_hint = self.title_input.text().strip()
        recorder = self.recorder

        def finish():
            # recorder.stop() 也必须在 try 内（审查 New-6）：收尾失败（采集/标注线程
            # 异常等）若在 try 之外冒出，finished 信号永不发射 —— UI 卡在「正在收尾…」
            # 且两个按钮全禁用，用户既等不到结果也无法重试。与 export_all 共用兜底：
            # 任何异常都收敛成 finished(False, …) → 槽内恢复按钮 + 弹失败提示。
            try:
                recorder.stop()
                title = title_hint or recorder.title or "未命名会议"
                started_at = recorder.started_at or datetime.now()
                transcript_path, minutes_path = export_all(
                    recorder.store, self.settings, title, started_at,
                    translator=recorder.translator)
                # 导出成功 = 本场已完整落盘：写 footer 关账，下次启动不再提示恢复
                self._finish_journal(recorder.store)
                self.bridge.finished.emit(True,
                    f"已生成：\n逐字稿：{transcript_path}\n纪要：{minutes_path}")
            except Exception as e:
                self.bridge.finished.emit(False, f"导出失败：{e}")
                return
            # 会后自动学习：从逐字稿挖掘术语（后台线程，不影响任何流程）
            try:
                self._mine_terms_after_meeting(recorder.store, title)
            except Exception:
                pass
                return        # 初版产物都没出：不再做会后标注（改不了产物）
            # 初版产物已落盘 → 再起后台线程补说话人（失败只告警，产物保持原样）
            self._maybe_offline_diarize(recorder, title, started_at)

        threading.Thread(target=finish, daemon=True).start()

    # ------------------------------------------------------------------
    # 会后离线说话人标注：初版产物出完 → 整场音频跑 pyannote → 覆盖导出
    # ------------------------------------------------------------------

    def _offline_diarize_allowed(self) -> bool:
        """是否该跑会后标注：需要 HF token，且实时/离线开关至少一个开着。

        开会上实时标注通常是关掉的（.env 里 DIARIZE_ENABLED=false，太慢），所以
        真正的开关是 OFFLINE_DIARIZE（默认 true）；没有 token 就没有模型，直接跳过。
        """
        settings = self.settings
        if not (getattr(settings, "hf_token", "") or "").strip():
            return False
        return bool(getattr(settings, "diarize_enabled", False)
                    or getattr(settings, "offline_diarize", True))

    def _maybe_offline_diarize(self, recorder, title: str, started_at: datetime) -> None:
        """初版产物已导出 → 起 daemon 线程补标注（不阻塞 GUI、不阻塞用户）。

        只在 finish() 线程里调用：此时 recorder 已 stop、录音文件已封口。
        """
        if not self._offline_diarize_allowed():
            return
        audio = getattr(recorder, "audio_path", None)
        try:
            has_audio = audio is not None and Path(audio).is_file()
        except OSError:
            has_audio = False
        if not has_audio:
            self.bridge.status.emit(OFFLINE_NO_AUDIO)
            return
        # 先报状态再起线程：标注要跑几十分钟，用户得知道产物还会再更新一次
        self.bridge.status.emit(OFFLINE_DIARIZE_STATUS)
        # daemon：程序退出时直接放弃标注（产物是增强项，不值得拦住退出）
        threading.Thread(target=self._offline_diarize_worker,
                         args=(recorder, Path(audio), title, started_at),
                         daemon=True).start()

    # ------------------------------------------------------------------
    # 说话人标注（独立动作）：可随时重跑、可指定人数、重启后能补上一场
    # ------------------------------------------------------------------

    def _find_last_recording(self) -> tuple[Path, Path] | None:
        """找最近一场的 (wav, jsonl)：用于「重启后补标注」。没有则 None。"""
        try:
            wavs = sorted(self._recording_dir().glob("rec_*.wav"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return None
        for wav in wavs:
            journal = self._pair_journal(wav)
            if journal is None:
                continue
            try:
                if any('"t": "add"' in line or '"t":"add"' in line
                       for line in journal.read_text(encoding="utf-8").splitlines()):
                    return wav, journal
            except OSError:
                continue
        return None

    def _label_speakers(self) -> None:
        """按钮/托盘入口：本场会话优先；否则从「已生成的会议记录」里选一场补标记。

        历史场次分两种情况：没有声纹标签 → 先重跑标注；已有标签 → 直接打开
        标记面板改名字（不再重跑、不覆盖已有名字以外的内容）。
        """
        session = getattr(self, "_last_session", None)
        audio_guess = getattr(session.get("recorder"), "audio_path", None) if session else None
        if audio_guess and Path(audio_guess).is_file():
            recorder = session["recorder"]
            audio_path, store = Path(audio_guess), session["store"]
            title = session.get("title") or getattr(recorder, "title", "") or "未命名会议"
            started_at = getattr(recorder, "started_at", None) or datetime.now()
            num = self._ask_speaker_count()
            if num is None:
                return
            self._run_label_job(audio_path, store, title, started_at, num,
                                (session or {}).get("translator"))
            return
        # ---- 历史会议：从 .recording 里挑一场 ----
        past = self._list_past_recordings()
        if not past:
            self.status.setText("没有可标注的录音（需要先录一场会议）")
            return
        names = [f"{p['title']} · {p['when']} · {p['segments']} 句"
                 + (" · 已标声纹" if p["labels"] else " · 未标注") for p in past]
        idx, ok = QInputDialog.getItem(self, "选择会议",
                                       "选择要补标记发言人的会议：", names, 0, False)
        if not ok:
            return
        pick = past[idx]
        try:
            title, started_at, store = load_journal(pick["jsonl"], attach_journal=True)
        except Exception as e:
            self.status.setText(f"读取会议记录失败：{e}")
            return
        # 本场会话上下文替换成历史场次（面板/回填/导出都吃这一份）
        self._last_session = {"store": store, "title": title,
                              "started_at": started_at or datetime.now(),
                              "recorder": None, "translator": None}
        if store.speakers():
            # 已有标签：直接打开标记面板改名字（不重跑标注）
            QTimer.singleShot(200, self._open_label_dialog)
            return
        num = self._ask_speaker_count()
        if num is None:
            return
        self._run_label_job(pick["wav"], store, title, started_at or datetime.now(),
                            num, None)

    def _ask_speaker_count(self) -> int | None:
        """提问人数（0=自动）；取消返回 None。"""
        num, ok = QInputDialog.getInt(
            self, "说话人数量", "本场有几个人说话？（0 = 自动检测）", 0, 0, 12)
        return int(num) if ok else None

    def _run_label_job(self, audio_path, store, title, started_at, num,
                       translator) -> None:
        self.label_btn.setEnabled(False)
        self.label_btn.setText("标注中…")
        self.status.setText("正在标注说话人（约 0.5~1.5 倍会议时长，可关闭窗口后台继续）…")
        threading.Thread(target=self._label_worker,
                         args=(audio_path, store, title, started_at,
                               num or None, translator),
                         daemon=True).start()

    def _list_past_recordings(self) -> list[dict]:
        """历史会议清单（有逐字稿内容的录音，按时间倒序）。"""
        out: list[dict] = []
        try:
            entries = sorted(self._recording_dir().glob("rec_*.wav"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return out
        for wav in entries:
            journal = self._pair_journal(wav)
            if journal is None:
                continue
            try:
                j_title, j_started, _ = load_journal(journal)
                n_add = sum('"t": "add"' in line or '"t":"add"' in line
                            for line in journal.read_text(encoding="utf-8").splitlines())
            except (OSError, ValueError):
                continue
            if n_add == 0:
                continue
            labels = 0
            try:
                _, _, j_store = load_journal(journal)
                labels = len(j_store.speakers())
            except (OSError, ValueError):
                pass
            if any(e["jsonl"] == journal for e in out):
                continue                    # 同一场多段录音，只列一场
            out.append({"wav": wav, "jsonl": journal,
                        "title": j_title or wav.stem,
                        "when": (j_started.strftime("%m-%d %H:%M") if j_started else ""),
                        "segments": n_add, "labels": labels})
        return out

    def _pair_journal(self, wav: Path) -> Path | None:
        """找 wav 配对的日志：先按同名；找不到就取 mtime 最接近的日志（5 分钟内）。

        历史版本里日志与音频分别取 datetime.now()，同名会差 1 秒；而长会里 wav
        的 mtime 是「录音结束时刻」、日志是「最后事件时刻」，两者可能差 1 分钟
        （实测 85 分钟会议差了 60 秒）→ 取最接近者，并给 5 分钟上限防误配。
        """
        exact = wav.with_suffix(".jsonl")
        if exact.is_file():
            return exact
        try:
            mtime = wav.stat().st_mtime
        except OSError:
            return None
        best, best_gap = None, float("inf")
        try:
            for j in self._recording_dir().glob("rec_*.jsonl"):
                try:
                    gap = abs(j.stat().st_mtime - mtime)
                except OSError:
                    continue
                if gap < best_gap:
                    best, best_gap = j, gap
        except OSError:
            pass
        return best if best is not None and best_gap <= 300 else None

    def _label_worker(self, audio_path: Path, store, title: str,
                      started_at: datetime, num_speakers: int | None,
                      translator=None) -> None:
        """后台：整场 wav → label() → 回填 store（并写入日志）→ 覆盖产物。"""
        suggestions = None
        try:
            diarizer_cls, assign = _resolve_diarize_tools()
            diarizer = diarizer_cls(self.settings)
            audio, sample_rate = _read_audio(audio_path)
            spans = diarizer.label(audio, sample_rate, num_speakers=num_speakers)
            changed: list[int] = []
            for seg in store.segments():
                before = seg.speaker
                assign([seg], spans)
                if seg.speaker != before:
                    changed.append(seg.seq)
            # 回写日志：崩溃恢复/再次导出时标注不丢（store.update 会写 journal 事件）
            for seq in changed:
                for seg in store.segments():
                    if seg.seq == seq:
                        store.update(seq, speaker=seg.speaker)
                        break
            for seg in store.segments():
                if seg.speaker:
                    store.register_label(seg.speaker)
            suggestions = self._collect_voiceprints(store, audio, sample_rate, spans)
            export_all(store, self.settings, title, started_at,
                       translator=translator or self._journal_translator(store))
        except Exception as e:
            self.bridge.diarized.emit(False, f"[警告] 说话人标注失败（产物保持原样）：{e}")
            return
        if suggestions is not None:
            self.bridge.speaker_suggestions.emit(suggestions)
        self.bridge.diarized.emit(True, OFFLINE_DIARIZED_DONE)

    def _journal_translator(self, store):
        session = getattr(self, "_last_session", None)
        if session and session.get("store") is store:
            return session.get("translator")
        return None

    def _offline_diarize_worker(self, recorder, audio_path: Path,
                                title: str, started_at: datetime) -> None:
        """后台：整场 wav → label() → assign_speakers → 重新导出（覆盖同名产物）。

        任何失败（无 token / 模型下载失败 / 磁盘异常 / 导出异常）只告警：
        初版产物在导出成功时就已落盘，标注失败不改变它们。
        """
        suggestions = None
        try:
            diarizer_cls, assign = _resolve_diarize_tools()
            diarizer = diarizer_cls(self.settings)
            audio, sample_rate = _read_audio(audio_path)
            spans = diarizer.label(audio, sample_rate)
            store = recorder.store
            # assign_speakers 直接改 seg.speaker（store.segments() 返回同一批实例）
            assign(store.segments(), spans)
            for seg in store.segments():
                if seg.speaker:
                    store.register_label(seg.speaker)
            # 声纹认人：算出本场每人的声纹向量并与本地库比对（失败只跳过，不影响产物）
            suggestions = self._collect_voiceprints(store, audio, sample_rate, spans)
            # 覆盖同名产物：逐字稿/纪要现在带说话人
            export_all(store, self.settings, title, started_at,
                       translator=getattr(recorder, "translator", None))
        except Exception as e:
            self.bridge.diarized.emit(
                False, f"[警告] 会后说话人标注失败（已导出的产物保持原样）：{e}")
            return
        # 建议在产物更新之后发：模态框出现时，用户手上的文件已经是带说话人版本
        if suggestions is not None:
            self.bridge.speaker_suggestions.emit(suggestions)
        self.bridge.diarized.emit(True, OFFLINE_DIARIZED_DONE)

    def _on_diarized(self, ok: bool, msg: str) -> None:
        """标注结果回到 GUI 线程：只更新状态栏（不弹窗打断用户）。

        窗口若已收进托盘，再补一条托盘气泡：标注要跑几十分钟，用户多半已经忘了
        这回事，「逐字稿更新了」得让他看得见。
        """
        self.status.setText(msg)
        try:
            self.label_btn.setText("标注说话人")
            self.label_btn.setEnabled(True)
        except AttributeError:
            pass
        if not self.isVisible() and not getattr(self, "_quitting", False):
            self._tray_notify("说话人标注完成" if ok else "说话人标注失败",
                              _notice_line(msg))
        if ok and not getattr(self, "_label_dialog_scheduled", False):
            store = self._current_store()
            if store is not None and store.speakers():
                self._label_dialog_scheduled = True
                QTimer.singleShot(400, self._open_label_dialog)

    # ------------------------------------------------------------------
    # 说话人记忆（声纹库）：会后认「熟人」+ 改名即学习
    # ------------------------------------------------------------------

    def _voiceprint_on(self) -> bool:
        """声纹库开关（VOICEPRINT_ENABLED，默认开）。关掉 = 不认人也不学习。"""
        return bool(getattr(self.settings, "voiceprint_enabled", True))

    def _reset_voiceprints(self) -> None:
        """新一场会议：清掉上一场的待入库向量 / 已入库标记 / 自动匹配建议。

        不清的话，上一场残留的 SPEAKER_00 向量会在本场第一次改名时被当成
        「本场这个人的样本」写进库 —— 名字对、人却是另一个，等于往库里灌脏数据。
        """
        self._pending_vectors: dict[str, object] = {}
        self._adopted: dict[str, str] = {}
        self._suggestion_names: dict[str, str] = {}

    def _embed_with_deadline(self, audio, sample_rate, spans):
        """带硬超时地算声纹向量：模型下载卡住 / 网络不通时也要按时把产物交出去。

        daemon 线程是为了「超时就真的放弃」：线程池在解释器退出时会 join，
        卡住的下载会拖住整个程序退不出去；daemon 线程随进程一起消失。
        """
        box: dict[str, object] = {}
        token = getattr(self.settings, "hf_token", "")

        def work() -> None:
            try:
                box["value"] = compute_speaker_embeddings(
                    audio, sample_rate, spans, token=token)
            except Exception:        # 声纹是增强项：算不出来就当没有
                box["value"] = {}

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(VOICEPRINT_TIMEOUT)
        value = box.get("value")
        return value if isinstance(value, dict) else {}

    def _collect_voiceprints(self, store, audio, sample_rate, spans):
        """（标注线程）算本场声纹向量 → 比对本地库 → 返回 ``{label: (姓名, 分数)}``。

        完全跳过（开关关掉 / 模型不可用 / 没有人声段）返回 ``None``，没有熟人时返回
        ``{}``（空字典也要发回主线程：库里已有名字的标签还能靠这次学会样本）。
        跑在非 GUI 线程：只碰 numpy 与库实例，绝不碰 Qt 控件。
        """
        if not self._voiceprint_on():
            return None
        embeddings = self._embed_with_deadline(audio, sample_rate, spans)
        if not embeddings:
            return None
        threshold = float(getattr(self.settings, "voiceprint_threshold", 0.75) or 0.75)
        known = {sp.label for sp in store.speakers()}
        suggestions: dict[str, tuple[str, float]] = {}
        for label, vector in embeddings.items():
            label = str(label)
            if known and label not in known:      # 库里算出来的标签必须在本场存在
                continue
            # 向量先暂存：等用户确认 / 之后手动改名时入库（本场结束也不丢）
            self._pending_vectors[label] = vector
            try:
                hits = self._voiceprints.match(vector, threshold=threshold)
            except Exception:
                hits = []
            if hits:
                suggestions[label] = (str(hits[0][0]), float(hits[0][1]))
        return suggestions

    def _on_speaker_suggestions(self, suggestions) -> None:
        """（GUI 线程）会后声纹建议：已有名字的向量先入库，建议名留给标记面板预填。"""
        if not self._voiceprint_on():
            return
        store = getattr(getattr(self, "recorder", None), "store", None)
        if store is None:
            return
        known = {sp.label for sp in store.speakers()}
        wanted: dict[str, tuple[str, float]] = {}
        for label, value in dict(suggestions or {}).items():
            label = str(label)
            try:
                name, score = str(value[0]), float(value[1])
            except (TypeError, IndexError, ValueError):
                continue
            if label in known:
                wanted[label] = (name, score)
        changed = self._adopt_named(store)
        self._suggestion_names.update({l: n for l, (n, _s) in wanted.items()})
        if changed:
            self._voiceprints.save()
        # 面板由 _on_diarized 统一打开（含没有声纹建议的场次）；这里只做预填

    # ------------------------------------------------------------------
    # 标记面板：对照逐字稿内容给声纹起名（会议后人工标记）
    # ------------------------------------------------------------------

    def _current_store(self):
        """标记相关操作的 store：优先当前会话，其次历史场次。"""
        rec = getattr(self, "recorder", None)
        if rec is not None:
            return getattr(rec, "store", None)
        session = getattr(self, "_last_session", None)
        return session.get("store") if session else None

    def _open_label_dialog(self) -> None:
        """打开说话人标记面板（每声纹：示例发言 + 名字输入框 + 声纹建议预填）。"""
        self._label_dialog_scheduled = False
        store = self._current_store()
        if store is None:
            return
        labels = store.speakers()
        if not labels:
            return
        data: list[dict] = []
        for sp in labels:
            segs = [s for s in store.segments() if s.speaker == sp.label]
            samples = []
            for seg in segs[:3]:
                body_text = seg.translation or seg.text
                samples.append(f"{body_text.strip()}\n  〔原文〕 {seg.text.strip()}")
            first = segs[0].start if segs else 0.0
            last = segs[-1].end if segs else 0.0
            data.append({
                "label": sp.label,
                "samples": samples,
                "suggest": self._suggestion_names.get(sp.label, ""),
                "count": len(segs),
                "span": f"{_fmt_hms(first)} → {_fmt_hms(last)}",
            })
        dlg = SpeakerLabelDialog(data, parent=self)
        if dlg.exec():
            self._apply_label_names(dlg.names)

    def _apply_label_names(self, names: dict[str, str]) -> None:
        """面板确认：写 store → 声纹学习 → 回填字幕 → 异步重导产物。"""
        store = self._current_store()
        if not store or not names:
            return
        for label, name in names.items():
            before = store.display_name(label)
            store.set_speaker_name(label, name)
            if store.display_name(label) != before:
                self._learn_rename(label, name)
        self._sync_subtitle_names()
        refresh = getattr(self.subtitle, "refresh_names", None)
        if callable(refresh):
            refresh()
        self._refresh_speakers()
        self._voiceprints.save()
        self.status.setText(f"已命名 {len(names)} 位说话人，逐字稿/纪要已更新")
        if not self.isVisible():
            self._tray_notify("说话人已命名", f"已命名 {len(names)} 位，产物已更新")
        # 异步重导出（带姓名）：本场有 translator 就重生成纪要；历史场次无 translator
        # 时只覆盖逐字稿（不能把旧纪要冲成占位符）
        session = getattr(self, "_last_session", None)
        if session is not None:
            translator = session.get("translator")
            if translator is None:
                try:
                    if getattr(self.settings, "opencode_api_key", "").strip():
                        translator = Translator(self.settings)
                except Exception:
                    translator = None
            title = session.get("title") or "未命名会议"
            started_at = session.get("started_at") or datetime.now()

            def _reexport():
                try:
                    if translator is not None:
                        export_all(store, self.settings, title, started_at,
                                   translator=translator)
                    else:
                        d = meeting_dir(self.settings, title, started_at)
                        write_transcript(store, d, title, started_at)
                except Exception:
                    pass
            threading.Thread(target=_reexport, daemon=True).start()

    def _adopt_named(self, store) -> bool:
        """把「本场已有人名」的标签向量入库（自动应用 / 用户先前手动改名都算）。"""
        changed = False
        for speaker in store.speakers():
            if speaker.label and (speaker.name or "").strip():
                changed |= self._adopt_vector(speaker.label, speaker.name)
        return changed

    def _adopt_vector(self, label: str, name: str) -> bool:
        """本场这个 label 的向量 → 记到 name 名下（同名只记一次，改名则重记）。"""
        name = (name or "").strip()
        if not name or self._adopted.get(label) == name:
            return False
        vector = self._pending_vectors.get(label)
        if vector is None:                # 本场没算出这个 label 的向量（无音频 / 模型不可用）
            return False
        if not self._voiceprints.add(name, vector):
            return False
        self._adopted[label] = name
        return True

    def _learn_rename(self, label: str, name: str) -> None:
        """（GUI 线程）改名 = 人工确认「这个声纹标签就是这个人」→ 声纹库学习。

        两类信号：① 正样本——本场该 label 的向量记到新名字下（下次就能认出来）；
        ② 纠错——本次自动匹配建议的是**别人**，说明库里那条样本是误归属，剔除它
        （否则下次还会认错）。改名是用户动作、库操作只是内存 + 一个小 JSON（毫秒级），
        异常一律吞掉：不能因为「学不会」而让改名报错。
        """
        if not self._voiceprint_on():
            return
        name = (name or "").strip()
        suggested = self._suggestion_names.pop(label, "")
        adopted = self._adopted.get(label, "")
        # 该标签此前被记到过谁名下（自动匹配建议 / 本场已入库），而现在改成了别人：
        # 这些归属都是错的，得把那条样本从库里剔除（去重后各剔除一次）
        wrong = {other for other in (suggested, adopted) if other and other != name}
        try:
            changed = False
            vector = self._pending_vectors.get(label) if wrong else None
            if vector is not None:
                for other in sorted(wrong):
                    changed |= bool(self._voiceprints.penalize(other, vector))
                # 误样本已剔除：同一条向量重新记到新名字下（它是对的样本，只是记错了人）
                self._adopted.pop(label, None)
            if name:
                changed |= self._adopt_vector(label, name)
            if changed:
                self._voiceprints.save()
        except Exception as e:            # 改名本身已经生效，这里只提示一句
            self.status.setText(f"[警告] 声纹学习失败（改名已生效）：{e}")

    def _set_recording_state(self, active: bool) -> None:
        """录制状态同步到「主窗口标题」与「字幕浮窗右上角红点」。"""
        self.setWindowTitle(WINDOW_TITLE_RECORDING if active else WINDOW_TITLE)
        setter = getattr(self.subtitle, "set_recording", None)
        if callable(setter):
            setter(active)

    def _on_finished(self, ok: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_recording_state(False)
        self.status.setText("完成" if ok else "导出失败")
        self._refresh_elapsed()          # 计时行立刻回到「就绪」（不必等下一秒的定时器）
        if not self._quitting:
            if ok:
                QMessageBox.information(self, "完成", msg)
            else:
                QMessageBox.critical(self, "导出失败", msg)
            # 窗口可能已收进托盘：不强行 show()（会打断用户），弹窗本身仍会显示；
            # 再补一条托盘气泡（带产物路径），免得用户没看到弹窗以为还没结束。
            if not self.isVisible() and msg:
                self._tray_notify("纪要已生成" if ok else "导出失败",
                                  _notice_line(msg))
        session = getattr(self, "_last_session", None)
        audio_guess = getattr(session.get("recorder"), "audio_path", None) if session else None
        if audio_guess and Path(audio_guess).is_file():
            self.label_btn.setEnabled(True)
        if self._quit_after_stop:
            # 关窗时选了「结束并保存后退出」（CLOSE_TO_TRAY=false）：导出完成即彻底退出
            self._quit_after_stop = False
            self._quit_app()

    def closeEvent(self, e):
        """关闭窗口：**总是** 收进托盘（录音 / 整理 / 标注继续在后台跑）。

        真正退出只有一条路：托盘菜单「退出」（见 _tray_quit）。开会时「关窗」的真实
        意图几乎都是「窗口挡住屏幕了」，而不是「这次会议不要了」。
        ``CLOSE_TO_TRAY=false`` 时恢复旧语义（关闭即退出，见 _close_with_exit）。
        """
        if getattr(self.settings, "close_to_tray", True):
            e.ignore()
            self.hide()
            if (self._is_recording() and not self._tray_hint_shown
                    and not self._quitting):
                self._tray_hint_shown = True
                self._tray_notify("录制继续中", TRAY_RECORDING_HINT)
            return
        self._close_with_exit(e)

    def _close_with_exit(self, e) -> None:
        """旧语义（CLOSE_TO_TRAY=false）：关闭主窗口 = 退出整个程序。

        录制中关窗：先确认；选「是」→ 后台结束录制、导出产物，完成后再退出，
        避免静默丢数据（导出在后台线程，GUI 不会冻结）。
        """
        if self._is_recording():
            finishing = not self.stop_btn.isEnabled()
            ret = QMessageBox.question(
                self, "退出确认",
                "正在生成纪要，退出会丢失本次结果。确定退出吗？" if finishing
                else "正在录制，退出会先结束录制并保存本次记录。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            if finishing:
                self._quit_app()
            else:
                self._quit_after_stop = True
                self._stop()          # 后台收尾 + 导出，完成后 _on_finished 退出
            e.ignore()
            return
        # 字幕浮窗是独立顶层窗口：hide 掉才会「最后一个窗口关完 → 进程退出」
        self.subtitle.hide()
        e.accept()


def main():
    """应用入口：创建窗口并进入事件循环（重型依赖由 _start 在后台惰性加载）。"""
    app = QApplication(sys.argv)
    try:
        settings = load_settings()
    except ConfigError as e:
        QMessageBox.critical(None, "配置错误", f"{e}\n\n请复制 .env.example 为 .env 并填写配置。")
        sys.exit(1)
    win = MainWindow(settings)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
