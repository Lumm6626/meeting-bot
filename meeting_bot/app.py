# -*- coding: utf-8 -*-
"""主窗口：开始/结束录制、说话人改名、状态显示。"""
from __future__ import annotations

import sys
import threading
from datetime import datetime

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from .capture import SOURCE_MIC, SOURCE_SPEAKER, AudioCapture, CaptureError
from .config import ConfigError, Settings, load_settings
from .diarization import Diarizer, DiarizationError
from .export import export_all
from .recorder import Recorder
from .subtitle import SubtitleWindow
from .translator import Translator


class RecorderBridge(QObject):
    """把工作线程的事件桥接到 Qt 主线程（线程安全）。"""
    segment = pyqtSignal(object)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    # 退出收尾完成（在后台线程里 emit）→ 主线程 _finish_close。
    # 不能用 QTimer.singleShot(0, ...)：该静态方法的计时器建在「调用线程」的事件循环上，
    # Python threading.Thread 没有 Qt 事件分发器 → 定时器永不触发，窗口会永久隐藏
    # 且进程不退出（已用离屏 Qt 实测复现）。信号 + 队列投递才是跨线程回主线程的正路。
    shutdown_done = pyqtSignal()


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("会议翻译助手")
        self.recorder: Recorder | None = None
        self.subtitle = SubtitleWindow()
        self.bridge = RecorderBridge()
        self.bridge.segment.connect(self._on_segment)
        self.bridge.status.connect(self._on_status)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.shutdown_done.connect(self._finish_close)
        # 退出收尾状态位：True 时二次 close() 直接放行，避免弹窗递归 / 重复后台收尾
        self._closing = False

        central = QWidget()
        lay = QVBoxLayout(central)

        # 音频源：线上会议截扬声器（loopback），面对面用麦克风。
        # itemData 存 AudioCapture 的 source 取值，UI 文案与协议值解耦。
        self.source_combo = QComboBox()
        self.source_combo.addItem("扬声器（线上会议）", SOURCE_SPEAKER)
        self.source_combo.addItem("麦克风（现场对话）", SOURCE_MIC)
        self.source_combo.setToolTip(
            "扬声器：截取系统播放的声音（Teams 等线上会议）\n"
            "麦克风：采集现场对话（会录到你自己的声音）")
        lay.addWidget(self.source_combo)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("会议名称（用于文件命名，可留空）")
        lay.addWidget(self.title_input)

        row = QHBoxLayout()
        self.start_btn = QPushButton("开始记录")
        self.stop_btn = QPushButton("结束并生成纪要")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)

        self.status = QLabel("就绪")
        lay.addWidget(self.status)

        self.speaker_table = QTableWidget(0, 2)
        self.speaker_table.setHorizontalHeaderLabels(["声纹标签", "名字"])
        lay.addWidget(self.speaker_table)

        self.setCentralWidget(central)
        self.resize(520, 420)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.speaker_table.itemChanged.connect(self._rename_speaker)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_speakers)
        self._timer.start(5000)

    def _probe_capture(self) -> AudioCapture | None:
        """按当前下拉选中的音频源探测一次设备，成功返回可复用的捕获实例。

        构造 ``AudioCapture`` 只解析设备、不打开音频流，因此探测不会占住设备；
        实例直接交给 ``Recorder`` 复用，避免「录到一半才发现设备不可用」。
        设备不可用（被占用 / 无麦克风 / 无 WASAPI）→ 弹框并返回 ``None``。
        """
        source = self.source_combo.currentData() or SOURCE_SPEAKER
        try:
            return AudioCapture(source=source)
        except CaptureError as e:
            QMessageBox.critical(self, "启动失败",
                                 f"无法打开麦克风/扬声器捕获设备：{e}")
            return None

    def _start(self):
        # 先探测设备再加载翻译/说话人：设备不可用时立刻失败，不必等重型初始化
        capture = self._probe_capture()
        if capture is None:
            return
        try:
            translator = Translator(self.settings)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"翻译接口初始化失败：{e}")
            return
        try:
            diarizer = Diarizer(self.settings)
        except DiarizationError as e:
            # 预期内：未配置 HF_TOKEN / 模型不可用等
            self._warn_diarizer_unavailable(e)
            diarizer = None
        except Exception as e:
            # 非预期：Pipeline.from_pretrained 的网络异常（HTTPError/ConnectionError）等。
            # Qt 槽内异常若外冒会直接终止进程（qFatal），故与预期内一样降级为「无说话人」继续记录。
            self._warn_diarizer_unavailable(e)
            diarizer = None
        self.recorder = Recorder(self.settings, capture=capture, translator=translator,
                                 diarizer=diarizer,
                                 audio_source=self.source_combo.currentData())
        self.recorder.on_segment(self.bridge.segment.emit)
        self.recorder.on_status(self.bridge.status.emit)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # 录制中锁住音频源：中途换设备没有意义（捕获已在跑），却会误导用户以为生效
        self.source_combo.setEnabled(False)
        threading.Thread(
            target=self.recorder.start,
            args=(self.title_input.text().strip(),),
            daemon=True,
        ).start()

    def _warn_diarizer_unavailable(self, e: Exception) -> None:
        QMessageBox.warning(self, "说话人功能不可用", f"{e}\n\n将继续记录，但无法区分说话人。")

    def _on_segment(self, seg):
        name = self.recorder.store.display_name(seg.speaker)
        self.subtitle.show_segment(seg, name)

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
        self.source_combo.setEnabled(True)
        self.status.setText("完成" if ok else "导出失败")
        if ok:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def closeEvent(self, e):
        """关闭窗口：录制 / 收尾中先确认，避免静默丢弃本次记录。

        recorder.stop() 最坏要等 10s（producer）+ 20s（翻译排空）+ 补译 + 180s（标注线程），
        在 GUI 线程同步调用会把窗口冻结数分钟（最终审查 Critical-2）。改为：确认后隐藏
        窗口，后台线程做收尾，完成信号回到主线程再真正关闭。
        """
        if self._closing:
            # 已完成一次收尾：直接放行（_finish_close 触发的二次 close）
            e.accept()
            return
        if self.recorder is not None and not self.start_btn.isEnabled():
            ret = QMessageBox.question(
                self, "退出确认",
                "正在录制/生成纪要，退出将丢弃本次记录。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self._closing = True
            self.hide()          # 先让窗口消失：收尾期间用户看到的是「已关闭」而非冻结的窗口
            e.ignore()           # 真正的关闭交给 _finish_close（收尾完成后）

            def _shutdown():
                try:
                    self.recorder.stop()
                except Exception:
                    pass             # 收尾失败不阻断退出（退出路径不能把用户锁死在窗口里）
                self.bridge.shutdown_done.emit()
            threading.Thread(target=_shutdown, daemon=True).start()
            return
        e.accept()

    def _finish_close(self):
        """后台收尾完成（GUI 线程槽）：复位状态后真正关闭窗口。

        必须先 setEnabled(True)：否则二次 closeEvent 看到「start 按钮仍禁用」会再次弹确认框、
        再起一个收尾线程 —— 递归弹窗且窗口关不掉。
        """
        self._closing = False
        self.start_btn.setEnabled(True)
        self.source_combo.setEnabled(True)
        self.close()


def main():
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
