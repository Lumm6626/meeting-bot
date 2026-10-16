# -*- coding: utf-8 -*-
"""主窗口：开始/结束录制、说话人改名、状态显示。"""
from __future__ import annotations

import sys
import threading
from datetime import datetime

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

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

        central = QWidget()
        lay = QVBoxLayout(central)

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

    def _start(self):
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
        self.recorder = Recorder(self.settings, translator=translator, diarizer=diarizer)
        self.recorder.on_segment(self.bridge.segment.emit)
        self.recorder.on_status(self.bridge.status.emit)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
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
            recorder.stop()
            title = title_hint or recorder.title or "未命名会议"
            try:
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
        if ok:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def closeEvent(self, e):
        """关闭窗口：录制 / 收尾中先确认，避免静默丢弃本次记录。"""
        if self.recorder is not None and not self.start_btn.isEnabled():
            ret = QMessageBox.question(
                self, "退出确认",
                "正在录制/生成纪要，退出将丢弃本次记录。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            try:
                self.recorder.stop()
            except Exception:
                pass
        e.accept()


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
