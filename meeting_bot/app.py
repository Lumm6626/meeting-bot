# -*- coding: utf-8 -*-
"""主窗口：开始/结束录制、说话人改名、状态显示。"""
from __future__ import annotations

import sys
import threading
from datetime import datetime
from string import Template

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMenu, QMessageBox, QPushButton, QSizePolicy,
                             QSystemTrayIcon, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from .capture import AudioCapture, CaptureError
from .config import PROJECT_ROOT, ConfigError, Settings, load_settings
from .export import export_all
from .subtitle import SubtitleWindow
from .translator import Translator


# 重型依赖延迟导入：顶层 import 会连带加载 pyannote/torch/faster-whisper，
# 实测让「双击快捷方式」到窗口出现的耗时从 ~1 秒涨到 13 秒以上（用户会以为没启动）。
# 这里留模块级占位名，首次真正需要时由 _ensure_heavy() 填充；测试 monkeypatch
# 这些名字（非 None）时不会被覆盖。
Recorder = None        # type: ignore[assignment]
Diarizer = None        # type: ignore[assignment]
_heavy_lock = threading.Lock()
_heavy_warm = False


def _ensure_heavy(preload_diarize: bool = True) -> None:
    """填充重型依赖（Recorder → faster-whisper/silero；Diarizer → pyannote/torch）。

    实测 ``import recorder`` 要 6 秒左右（torch/faster-whisper/silero），所以窗口
    出现后会先调一次预热（见 MainWindow._preload_engine），点开始时不再等这 6 秒。
    """
    global Recorder, Diarizer, _heavy_warm
    with _heavy_lock:
        if Recorder is None:
            from .recorder import Recorder as _Recorder
            Recorder = _Recorder
        if preload_diarize and Diarizer is None:
            from .diarization import Diarizer as _Diarizer
            Diarizer = _Diarizer
        if not _heavy_warm:
            # 预热 silero VAD（约 0.2s），让首次切句不额外等待
            try:
                from .vad import UtteranceBuffer
                UtteranceBuffer()
            except Exception:
                pass
            _heavy_warm = True


def app_icon() -> QIcon:
    """应用图标：窗口 / 托盘 / 桌面快捷方式共用 assets/app.ico。"""
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


def _pill(text: str) -> QLabel:
    """配置胶囊：只读小徽章（样式见 DARK_QSS 的 QLabel[role="pill"]）。"""
    lab = QLabel(text)
    lab.setProperty("role", "pill")
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lab


def brand_pixmap(size: int = 48) -> QPixmap:
    """品牌行图标：assets/app.ico 的 48px 缩略（按屏幕 DPR 取图，HiDPI 下不糊）。"""
    dpr = 1.0
    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is not None:
        dpr = float(screen.devicePixelRatio() or 1.0)
    pm = app_icon().pixmap(int(round(size * dpr)), int(round(size * dpr)))
    if not pm.isNull():
        pm.setDevicePixelRatio(dpr)
    return pm


class RecorderBridge(QObject):
    """把工作线程的事件桥接到 Qt 主线程（线程安全）。"""
    segment = pyqtSignal(object)
    # 译文填充后的原地更新（两阶段显示第二阶段）：与 segment 分开，
    # 面板按 seg.seq 找到原块替换，不新增块。
    update = pyqtSignal(object)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("会议翻译助手")
        self.recorder: Recorder | None = None
        self.subtitle = SubtitleWindow()
        self.bridge = RecorderBridge()
        self.bridge.segment.connect(self._on_segment)
        self.bridge.update.connect(self._on_update)
        self.bridge.status.connect(self._on_status)
        self.bridge.finished.connect(self._on_finished)
        # 关窗触发收尾时置 True：导出完成后由 _on_finished 直接退出应用
        self._quit_after_stop = False

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
        row.addWidget(self.start_btn, 1)
        row.addWidget(self.stop_btn, 1)
        lay.addLayout(row)

        # ---- 状态卡片：状态文本（加粗白）+ 录制计时 ----
        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(14, 12, 14, 12)
        card_lay.setSpacing(6)
        self.status = QLabel("就绪")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.elapsed_label = QLabel("就绪")
        self.elapsed_label.setObjectName("elapsed")
        card_lay.addWidget(self.status)
        card_lay.addWidget(self.elapsed_label)
        lay.addWidget(card)

        self.speaker_table = QTableWidget(0, 2)
        self.speaker_table.setHorizontalHeaderLabels(["声纹标签", "名字"])
        self.speaker_table.setAlternatingRowColors(True)
        self.speaker_table.setShowGrid(False)
        self.speaker_table.verticalHeader().setVisible(False)
        self.speaker_table.horizontalHeader().setStretchLastSection(True)
        self.speaker_table.setMinimumHeight(140)
        lay.addWidget(self.speaker_table, 1)

        self.footer_hint = QLabel("录制中关闭窗口将收进托盘继续工作 · 托盘图标可随时结束")
        self.footer_hint.setObjectName("footer_hint")
        self.footer_hint.setWordWrap(True)
        lay.addWidget(self.footer_hint)

        self.setCentralWidget(central)
        self.setStyleSheet(DARK_QSS)
        self.resize(560, 600)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
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

    def _quit_app(self) -> None:
        """退出应用：先收浮窗，避免隐藏窗口把进程留住。"""
        self.subtitle.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()

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
        self.status.setText("正在准备识别引擎…")
        if not self._title_edited:
            self.title_input.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        title = self.title_input.text().strip()

        def _boot():
            try:
                _ensure_heavy(self.settings.diarize_enabled)
                translator = Translator(self.settings)
            except Exception as e:
                self.bridge.finished.emit(False, f"启动失败：{e}")
                return
            diarizer = None
            if self.settings.diarize_enabled:
                try:
                    diarizer = Diarizer(self.settings)
                except Exception as e:
                    # 预期内（未配 HF_TOKEN）与非预期（模型下载/网络异常）一律降级继续：
                    # Qt 槽内异常外冒会直接终止进程（qFatal），不能让它冒出去。
                    self._warn_diarizer_unavailable(e)
            try:
                self.recorder = Recorder(
                    self.settings, capture=capture, translator=translator,
                    diarizer=diarizer,
                    audio_source=getattr(self.settings, "audio_source", "both"))
                self.recorder.on_segment(self.bridge.segment.emit)
                self.recorder.on_update(self.bridge.update.emit)
                self.recorder.on_status(self.bridge.status.emit)
            except Exception as e:
                self.bridge.finished.emit(False, f"启动失败：{e}")
                return
            self.recorder.start(title)

        threading.Thread(target=_boot, daemon=True).start()

    def _preload_engine(self) -> None:
        """后台预热重型依赖：把「点开始记录」的等待从 ~6 秒压到 ~1 秒。"""
        try:
            _ensure_heavy(self.settings.diarize_enabled)
        except Exception:
            pass          # 预热失败不影响使用：首次点击时还会再试一次

    def _mark_title_edited(self, _text: str = "") -> None:
        """用户手动改过会议名：之后不再用时间自动覆盖。"""
        self._title_edited = True

    def _warn_diarizer_unavailable(self, e: Exception) -> None:
        QMessageBox.warning(self, "说话人功能不可用", f"{e}\n\n将继续记录，但无法区分说话人。")

    def _on_segment(self, seg):
        name = self.recorder.store.display_name(seg.speaker)
        self.subtitle.show_segment(seg, name)

    def _on_update(self, seg):
        """译文到达（第二阶段）：交给面板按 seq 原地替换，不新增块。"""
        name = self.recorder.store.display_name(seg.speaker)
        self.subtitle.update_segment(seg, name)

    def _on_status(self, msg):
        self.status.setText(msg)

    def _refresh_speakers(self):
        if not self.recorder:
            return
        self.speaker_table.blockSignals(True)
        speakers = self.recorder.store.speakers()
        self.speaker_table.setRowCount(len(speakers))
        for i, sp in enumerate(speakers):
            self.speaker_table.setItem(i, 0, QTableWidgetItem(sp.label))
            self.speaker_table.setItem(i, 1, QTableWidgetItem(sp.name))
        self.speaker_table.blockSignals(False)

    def _rename_speaker(self, item):
        if item.column() != 1 or not self.recorder:
            return
        label_item = self.speaker_table.item(item.row(), 0)
        if label_item:
            self.recorder.store.set_speaker_name(label_item.text(), item.text())

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
                transcript_path, minutes_path = export_all(
                    recorder.store, self.settings, title,
                    recorder.started_at or datetime.now(),
                    translator=recorder.translator)
                self.bridge.finished.emit(True,
                    f"已生成：\n逐字稿：{transcript_path}\n纪要：{minutes_path}")
            except Exception as e:
                self.bridge.finished.emit(False, f"导出失败：{e}")

        threading.Thread(target=finish, daemon=True).start()

    def _on_finished(self, ok: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("完成" if ok else "导出失败")
        self._refresh_elapsed()          # 计时行立刻回到「就绪」（不必等下一秒的定时器）
        if ok:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)
        if self._quit_after_stop:
            # 关窗时选了「结束并保存后退出」：导出完成即彻底退出
            self._quit_after_stop = False
            self._quit_app()

    def closeEvent(self, e):
        """关闭主窗口 = 退出整个程序（不留后台托盘/常驻进程）。

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
