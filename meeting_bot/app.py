# -*- coding: utf-8 -*-
"""主窗口：开始/结束录制、说话人改名、状态显示。"""
from __future__ import annotations

import sys
import threading
from datetime import datetime

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMenu, QMessageBox, QPushButton, QSystemTrayIcon,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .capture import AudioCapture, CaptureError
from .config import PROJECT_ROOT, ConfigError, Settings, load_settings
from .diarization import Diarizer, DiarizationError
from .export import export_all
from .recorder import Recorder
from .subtitle import SubtitleWindow
from .translator import Translator


def app_icon() -> QIcon:
    """应用图标：窗口 / 托盘 / 桌面快捷方式共用 assets/app.ico。"""
    return QIcon(str(PROJECT_ROOT / "assets" / "app.ico"))


class RecorderBridge(QObject):
    """把工作线程的事件桥接到 Qt 主线程（线程安全）。"""
    segment = pyqtSignal(object)
    # 译文填充后的原地更新（两阶段显示第二阶段）：与 segment 分开，
    # 面板按 seg.seq 找到原块替换，不新增块。
    update = pyqtSignal(object)
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
        self.bridge.update.connect(self._on_update)
        self.bridge.status.connect(self._on_status)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.shutdown_done.connect(self._finish_close)
        # 退出收尾状态位：True 时二次 close() 直接放行，避免弹窗递归 / 重复后台收尾
        self._closing = False
        # 收尾完成后是否直接退出应用（托盘「退出」路径 True；关窗路径 False）
        self._quit_after_stop = False

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

        # 托盘放在最后：菜单槽会读写 start_btn / stop_btn / subtitle，必须等它们就绪
        self.setWindowIcon(app_icon())
        self._setup_tray()

    def _setup_tray(self) -> None:
        """系统托盘：把「关窗」与「退出」解耦（关窗只收起界面，录制照旧）。"""
        self.tray_menu = QMenu(self)
        self.tray_show_action = self.tray_menu.addAction("打开主窗口")
        self.tray_show_action.triggered.connect(self._show_window)
        self.tray_stop_action = self.tray_menu.addAction("结束并生成纪要")
        self.tray_stop_action.triggered.connect(self._tray_stop)
        self.tray_quit_action = self.tray_menu.addAction("退出")
        self.tray_quit_action.triggered.connect(self._tray_quit)

        # 图标用同一份 ico：窗口 / 托盘 / 快捷方式视觉一致
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("会议翻译助手")
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason) -> None:
        """单击 / 双击托盘图标 → 还原主窗口（已收进托盘也能回来）。"""
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self._show_window()

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_notify(self, title: str, msg: str) -> None:
        """托盘气泡：窗口隐藏时唯一「看得见」的提示通道。"""
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.showMessage(title, msg)

    def _hide_tray(self) -> None:
        """收掉托盘图标：避免进程退出后残留「幽灵图标」。"""
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.hide()

    def _is_recording(self) -> bool:
        """录制 / 收尾中：start 按钮禁用（_start 置位、_on_finished 复位）。"""
        return self.recorder is not None and not self.start_btn.isEnabled()

    def _tray_stop(self) -> None:
        """托盘「结束并生成纪要」：录制中复用 _stop() 的后台收尾；空闲时无事可做。

        收尾中 stop_btn 已禁用 —— 再点一次不能再起一个 stop()（会重复补译 + 重复导出）。
        """
        if self._is_recording() and self.stop_btn.isEnabled():
            self._stop()

    def _tray_quit(self) -> None:
        """托盘「退出」：录制中先确认（丢弃本次记录），确认后后台收尾再退出。

        收尾（生成纪要）中再点退出：不能再调一次 stop()（会二次补译 + 重复导出），
        确认后直接退出；收尾线程是 daemon，随进程结束放弃。
        """
        if not self._is_recording():
            self._quit_app()
            return
        finishing = not self.stop_btn.isEnabled()
        ret = QMessageBox.question(
            self, "退出确认",
            "正在生成纪要，退出将丢失本次结果？确定退出吗？" if finishing
            else "正在录制，退出将丢弃本次记录？确定退出吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes:
            return
        if finishing:
            self._quit_app()
            return
        self._shutdown_in_background(quit_after=True)

    def _quit_app(self) -> None:
        """退出应用：先收托盘 / 浮窗，避免残留图标与隐藏窗口把进程留住。"""
        self._hide_tray()
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
        # 先探测设备再加载翻译/说话人：设备不可用时立刻失败，不必等重型初始化
        capture = self._probe_capture()
        if capture is None:
            return
        try:
            translator = Translator(self.settings)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"翻译接口初始化失败：{e}")
            return
        diarizer = None
        if self.settings.diarize_enabled:
            try:
                diarizer = Diarizer(self.settings)
            except DiarizationError as e:
                # 预期内：未配置 HF_TOKEN / 模型不可用等
                self._warn_diarizer_unavailable(e)
            except Exception as e:
                # 非预期：Pipeline.from_pretrained 的网络异常（HTTPError/ConnectionError）等。
                # Qt 槽内异常若外冒会直接终止进程（qFatal），故与预期内一样降级为「无说话人」继续记录。
                self._warn_diarizer_unavailable(e)
        self.recorder = Recorder(self.settings, capture=capture, translator=translator,
                                 diarizer=diarizer,
                                 audio_source=getattr(self.settings, "audio_source", "both"))
        self.recorder.on_segment(self.bridge.segment.emit)
        self.recorder.on_update(self.bridge.update.emit)
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
        if ok:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)
        # 窗口可能已收进托盘：不强行 show()（会打断用户），弹窗本身仍会显示；
        # 再补一条托盘气泡，免得用户没看到弹窗以为还没结束。
        if not self.isVisible() and msg:
            self._tray_notify("纪要已生成" if ok else "导出失败", msg.splitlines()[0])

    def closeEvent(self, e):
        """关闭窗口：录制中不退出，只收进托盘（录制继续、字幕浮窗照旧）。

        旧语义是「弹框确认 → 丢弃本次记录 → 退出」，但开会时关窗的真实意图多半只是
        「窗口挡住屏幕了」。现在关窗 = 隐藏主窗口；退出入口集中到托盘菜单：
        「结束并生成纪要」复用 _stop() 的后台收尾，「退出」在录制中先明确确认
        （丢弃本次记录）才后台收尾并退出 —— 既不静默丢数据，也不把 GUI 线程
        卡在数分钟的 recorder.stop() 上。
        """
        if self._closing:
            # 已完成一次收尾：直接放行（_finish_close 触发的二次 close）
            e.accept()
            return
        if self._is_recording():
            if self.tray.isVisible():
                e.ignore()
                self.hide()
                self._tray_notify("录制继续中", "字幕仍在更新。点托盘图标可重新打开窗口。")
                return
            # 系统托盘不可用（少数 Windows 会话 / 远程桌面）：收进托盘会变成
            # 「窗口消失、录制还在跑、无处点回来」的死角 → 退回确认框并说明后果
            ret = QMessageBox.question(
                self, "退出确认",
                "系统托盘不可用，关闭窗口将丢弃本次记录。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self._shutdown_in_background(quit_after=False)   # 后台收尾后再关窗
            e.ignore()
            return
        self._hide_tray()        # 非录制中：正常关闭，顺手收掉托盘图标
        e.accept()

    def _shutdown_in_background(self, *, quit_after: bool) -> None:
        """隐藏界面后在后台线程收尾，完成信号回主线程再关闭 / 退出。

        recorder.stop() 最坏要等 10s（producer）+ 20s（翻译排空）+ 补译 + 180s（标注线程），
        在 GUI 线程同步调用会把窗口冻结数分钟（最终审查 Critical-2），故一律异步。
        """
        self._closing = True
        self._quit_after_stop = quit_after
        self.hide()
        recorder = self.recorder

        def _shutdown():
            # 收尾失败不阻断退出（退出路径不能把用户锁死在窗口里）
            try:
                if recorder is not None:
                    recorder.stop()
            except Exception:
                pass
            self.bridge.shutdown_done.emit()

        threading.Thread(target=_shutdown, daemon=True).start()

    def _finish_close(self):
        """后台收尾完成（GUI 线程槽）：复位状态后真正关闭窗口。

        必须先 setEnabled(True)：否则二次 closeEvent 看到「仍在录制」会再次收进托盘、
        窗口永远关不掉。
        """
        self._closing = False
        self.start_btn.setEnabled(True)
        if self._quit_after_stop:
            self._quit_after_stop = False
            self._quit_app()     # 托盘「退出」：收尾完成后真正退出
        else:
            self.close()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())   # 任务栏 / 弹窗统一图标（托盘见 MainWindow）
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
