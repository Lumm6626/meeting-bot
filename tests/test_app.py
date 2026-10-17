# -*- coding: utf-8 -*-
"""Task 12 修复轮 2：app.py 的线程红线与退出路径固化测试（离屏，无网络 / 无录音）。

两条不变量来自修复轮 1 的临时验证脚本（.superpowers/sdd/.../task-12-fix1-verify.py，
该目录不入版本控制），此处固化成常驻回归，防止后续改动悄悄回退：

  a. ``_stop()`` 的 finish() 线程内零 Qt 控件访问（Major-1 的线程红线）；
  b. ``closeEvent``：录制中关窗 = 收进托盘（不停止录制），空闲时才真正退出；
  b2. 托盘菜单：打开主窗口 / 结束并生成纪要 / 退出（录制中先确认）；
  c. finish() 内 recorder.stop() 抛异常也必须走到 finished(False)（New-6 兜底）。

全程离屏 + 假对象：不构造真 Recorder、不启动采集/识别/翻译、不发任何网络请求。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 必须在 import PyQt6 之前

import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QSystemTrayIcon as _RealTray

import meeting_bot.app as appmod
from meeting_bot.app import MainWindow
from meeting_bot.config import Settings
from meeting_bot.transcript import Segment, TranscriptStore

WIDGETS = ("title_input", "status", "start_btn", "stop_btn", "speaker_table")



class _FakeSignal:
    """最小信号替身：只支持 connect / emit（替代 QSystemTrayIcon.activated）。"""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class FakeTray:
    """QSystemTrayIcon 替身：离屏环境没有系统托盘，用本类记录可见性与气泡。

    ``ActivationReason`` / ``MessageIcon`` 转发真枚举，app.py 里的枚举比较照常成立。
    """

    ActivationReason = _RealTray.ActivationReason
    MessageIcon = _RealTray.MessageIcon
    instances: list["FakeTray"] = []

    def __init__(self, icon=None, parent=None):
        self.icon = icon
        self.parent = parent
        self.tooltip = ""
        self.menu = None
        self.messages: list[tuple[str, str]] = []
        self.visible = False
        self.activated = _FakeSignal()
        FakeTray.instances.append(self)

    def setToolTip(self, text):
        self.tooltip = text

    def setContextMenu(self, menu):
        self.menu = menu

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def isVisible(self):
        return self.visible

    def showMessage(self, title, msg, *args, **kwargs):
        self.messages.append((title, msg))

# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------

def _settings(tmp_path, audio_source: str = "both") -> Settings:
    return Settings("https://example.invalid/v1", "sk-fake", "m1", "m2", "medium",
                    "", tmp_path / "会议记录", tmp_path, audio_source=audio_source)


@pytest.fixture(scope="module")
def qapp():
    """离屏 QApplication（整模块复用；无事件循环，需要时手动 processEvents）。"""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def fake_tray(monkeypatch):
    """全模块统一用 FakeTray 顶替 QSystemTrayIcon（真实托盘在离屏下不可用）。

    autouse：本文件所有 MainWindow 构造点（含 _make_win）都会经过它，避免个别用例
    意外创建真实托盘对象（会打印 QSystemTrayIcon 警告并污染全局状态）。
    """
    FakeTray.instances.clear()
    monkeypatch.setattr(appmod, "QSystemTrayIcon", FakeTray)
    yield FakeTray


@pytest.fixture
def win(qapp, tmp_path):
    w = MainWindow(_settings(tmp_path))
    w._timer.stop()      # 关掉 5 秒定时刷新：避免其读写控件干扰「零控件访问」断言
    try:
        yield w
    finally:
        w.recorder = None
        w.start_btn.setEnabled(True)
        w.hide()


@pytest.fixture
def dialogs(monkeypatch):
    """QMessageBox 四个静态方法 → 记录器，避免模态阻塞（返回 (recorded, answer)）。"""
    recorded: list[tuple[str, str, str]] = []
    answer = {"ret": appmod.QMessageBox.StandardButton.Yes}

    def _rec(kind):
        return staticmethod(lambda *a, **k: recorded.append((kind, a[1], a[2])))

    monkeypatch.setattr(appmod.QMessageBox, "information", _rec("info"))
    monkeypatch.setattr(appmod.QMessageBox, "critical", _rec("crit"))
    monkeypatch.setattr(appmod.QMessageBox, "warning", _rec("warn"))

    def _question(*a, **k):
        recorded.append(("question", a[1], a[2]))
        return answer["ret"]

    monkeypatch.setattr(appmod.QMessageBox, "question", staticmethod(_question))
    return recorded, answer


class ThreadGuard:
    """Qt 控件代理：记录每次方法调用所在线程，非 GUI 线程调用 → violation。"""

    def __init__(self, obj, label: str, gui_thread: threading.Thread):
        self._obj = obj
        self.label = label
        self.gui_thread = gui_thread
        self.calls: list[tuple[str, bool]] = []
        self.violations: list[str] = []

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **k):
            in_gui = threading.current_thread() is self.gui_thread
            self.calls.append((name, in_gui))
            if not in_gui:
                self.violations.append(f"{self.label}.{name}")
            return attr(*a, **k)

        return wrapped


class FakeRecorder:
    """_stop()/closeEvent() 所需的最小 Recorder 替身（stop_boom 模拟收尾抛异常）。"""

    def __init__(self, title: str = "假会议", stop_boom: bool = False):
        self.store = TranscriptStore()
        self.title = title
        self.started_at = datetime(2026, 10, 16, 10, 0, 0)
        self.translator = None
        self.stop_boom = stop_boom
        self.stop_calls = 0
        self.stop_thread = None

    def stop(self):
        self.stop_calls += 1
        self.stop_thread = threading.current_thread()
        if self.stop_boom:
            raise RuntimeError("模拟收尾异常")


def _install_guards(win, monkeypatch, gui_thread):
    guards = {}
    for name in WIDGETS:
        guard = ThreadGuard(getattr(win, name), name, gui_thread)
        monkeypatch.setattr(win, name, guard)
        guards[name] = guard
    return guards


def _pump_until(app, pred, timeout: float = 10.0) -> bool:
    """跑 Qt 事件循环直到 pred() 为真。

    bridge.finished 是跨线程 emit：连到主线程对象的槽走队列投递，
    不 pump 事件循环就永远不会到达 _on_finished。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    app.processEvents()
    return pred()


def _export_stub(exports, threads):
    """export_all 替身：记录收到的标题与调用线程，不落盘、不调模型。"""
    def fake_export_all(store, settings, title, started_at, translator=None):
        threads.append(threading.current_thread())
        exports.append({"store": store, "title": title, "started_at": started_at,
                        "translator": translator})
        return Path("逐字稿.md"), Path("纪要.docx")
    return fake_export_all


def _prepare_recording(win, recorder):
    """把窗口置成「录制中」并注入 fake recorder（不点按钮、不起线程）。"""
    win.recorder = recorder
    win.start_btn.setEnabled(False)
    win.stop_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# a. finish() 线程内零 Qt 控件访问（Major-1 线程红线固化）
# ---------------------------------------------------------------------------

def test_finish_thread_touches_no_qt_widget(win, qapp, dialogs, monkeypatch):
    """固化不变式 a：收尾全链路（含信号回槽）只有 GUI 线程碰控件。

    旧实现把 ``title_input.text()`` 放在 finish() 内：该函数跑在后台线程，跨线程读
    Qt 控件在真机上会随机崩溃 / 触发 qFatal。此处用 ThreadGuard 代理 5 个控件记录
    调用线程，跑完「成功收尾 → 弹框 → 恢复按钮」整条路径后断言零 off-GUI 访问。
    """
    recorded, _ = dialogs
    gui_thread = threading.current_thread()
    exports, export_threads, finished = [], [], []

    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))

    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.title_input.setText("季度复盘")     # 装守卫前写入

    guards = _install_guards(win, monkeypatch, gui_thread)
    for g in guards.values():
        g.calls.clear()
        g.violations.clear()

    win.bridge.finished.connect(lambda ok_, msg: finished.append((ok_, msg)))
    win._stop()

    assert _pump_until(qapp, lambda: bool(finished)), "finished 信号未到达槽"
    assert _pump_until(qapp, lambda: guards["start_btn"]._obj.isEnabled())

    # 1) 正向对照：finish 确实跑在非 GUI 线程，否则下面「零 off-GUI 访问」毫无意义
    assert recorder.stop_calls == 1
    assert recorder.stop_thread is not gui_thread
    assert export_threads and export_threads[0] is not gui_thread
    # 2) 标题在主线程取好并传下去（输入框优先）
    assert [e["title"] for e in exports] == ["季度复盘"]
    assert exports[0]["store"] is recorder.store
    assert [c for c in guards["title_input"].calls if c[0] == "text"] == [("text", True)]
    # 3) 关键断言：finish() 线程零 Qt 控件访问
    bad = [(name, g.violations) for name, g in guards.items() if g.violations]
    assert not bad, f"收尾线程访问了 Qt 控件：{bad}"
    # 4) 收尾正常收场：状态/按钮回位、成败提示走槽（GUI 线程）
    assert guards["status"]._obj.text() == "完成"
    assert guards["start_btn"]._obj.isEnabled() and not guards["stop_btn"]._obj.isEnabled()
    assert [d[:2] for d in recorded] == [("info", "完成")]
    assert "逐字稿.md" in recorded[0][2] and finished[0][0] is True


# ---------------------------------------------------------------------------
# b. New-6：finish() 内 recorder.stop() 抛异常也必须收敛成 finished(False)
# ---------------------------------------------------------------------------

def test_finish_stop_exception_still_emits_and_restores_buttons(win, qapp, dialogs, monkeypatch):
    """New-6 回归：收尾异常不得让 finished 永不发射（UI 卡死在「正在收尾…」）。

    旧实现 recorder.stop() 在 try 之外：抛异常则 finished 信号永不发射，
    两个按钮全禁用 —— 用户既等不到结果也无法重试。修复后与 export_all 共用兜底。
    """
    recorded, _ = dialogs
    exports, export_threads, finished = [], [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))

    recorder = FakeRecorder(stop_boom=True)
    _prepare_recording(win, recorder)

    win.bridge.finished.connect(lambda ok_, msg: finished.append((ok_, msg)))
    win._stop()

    assert _pump_until(qapp, lambda: bool(finished)), "stop() 抛异常后 finished 未发射"
    assert _pump_until(qapp, lambda: win.start_btn.isEnabled())

    ok, msg = finished[0]
    assert ok is False and "导出失败：模拟收尾异常" in msg
    assert exports == []                 # stop() 失败后不再继续导出
    assert win.status.text() == "导出失败"
    assert win.start_btn.isEnabled() and not win.stop_btn.isEnabled()   # 可重试
    assert [d[:2] for d in recorded] == [("crit", "导出失败")]


# ---------------------------------------------------------------------------
# c. closeEvent：录制中收进托盘（录制继续），空闲时才真正退出
# ---------------------------------------------------------------------------

def test_close_event_idle_accepts_and_hides_tray(win, dialogs):
    """未录制：直接退出，不弹确认框；托盘图标一并收掉（不留幽灵图标）。"""
    recorded, _ = dialogs
    win.recorder = None
    win.start_btn.setEnabled(True)
    win.show()
    tray = win.tray
    assert tray.visible

    assert win.close() is True          # e.accept()：真关闭
    assert not win.isVisible()
    assert not tray.visible             # 退出前 hide() 托盘
    assert [d for d in recorded if d[0] == "question"] == []


def test_close_event_while_recording_hides_to_tray_without_stopping(win, dialogs):
    """录制中关窗：收进托盘继续录 —— 不弹框、不停止录制、给一条托盘气泡。"""
    recorded, _ = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.show()
    tray = win.tray

    assert win.close() is False         # e.ignore()：窗口没关，只是隐藏
    assert not win.isVisible()
    assert win._closing is False
    assert recorder.stop_calls == 0     # 关键：录制继续（音频 / 字幕不中断）
    assert win._quit_after_stop is False
    assert not win.start_btn.isEnabled()          # 录制状态保持
    assert [d for d in recorded if d[0] == "question"] == []   # 不再弹「丢弃记录」确认
    assert tray.messages and tray.messages[-1][0] == "录制继续中"
    assert "字幕仍在更新" in tray.messages[-1][1]

    # 托盘单击 / 双击 → 重新打开主窗口
    tray.activated.emit(FakeTray.ActivationReason.Trigger)
    assert win.isVisible()


def test_on_finished_while_hidden_notifies_via_tray(win, dialogs):
    """收尾完成时窗口若已收进托盘：不强行 show()，弹窗照旧 + 托盘气泡补一条。"""
    recorded, _ = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.hide()
    win._on_finished(True, "已生成：" + chr(10) + "逐字稿：a.md")   # 多行 msg：取首行做气泡

    assert not win.isVisible()                       # 不打断用户
    assert recorded[-1][:2] == ("info", "完成")      # 弹窗仍然出现
    assert win.tray.messages[-1] == ("纪要已生成", "已生成：")


def test_close_event_without_usable_tray_falls_back_to_confirm(win, qapp, dialogs):
    """系统托盘不可用时的死角兜底：不能「窗口消失、录制还在跑、无处点回来」。

    此时退回确认框（明确告知会丢记录）：No 留在窗口里；Yes 后台收尾后再关窗。
    """
    recorded, answer = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.show()
    win.tray.visible = False            # 模拟托盘不可用（isVisible() 为 False）

    answer["ret"] = appmod.QMessageBox.StandardButton.No
    assert win.close() is False
    assert win.isVisible() and recorder.stop_calls == 0
    assert "系统托盘不可用" in recorded[-1][2]

    answer["ret"] = appmod.QMessageBox.StandardButton.Yes
    assert win.close() is False         # 后台收尾，GUI 不阻塞
    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and not win._closing)
    assert win.start_btn.isEnabled()
    assert win.close() is True


# ---------------------------------------------------------------------------
# f. 托盘菜单：结束并生成纪要 / 退出（录制中先确认）
# ---------------------------------------------------------------------------

def test_app_icon_resource_and_tray_wiring(win, qapp):
    """图标资源可被 Qt 加载，窗口 / 托盘共用同一份，托盘菜单三项齐备。"""
    ico = appmod.PROJECT_ROOT / "assets" / "app.ico"
    assert ico.is_file(), "缺少 assets/app.ico（scripts/make_icon.py 生成）"
    assert ico.stat().st_size > 0
    assert not appmod.app_icon().isNull(), "assets/app.ico 无法被 Qt 加载"
    assert not win.windowIcon().isNull()
    assert isinstance(win.tray, FakeTray)
    assert win.tray.tooltip == "会议翻译助手"
    assert [a.text() for a in win.tray_menu.actions()] ==         ["打开主窗口", "结束并生成纪要", "退出"]


def test_tray_menu_stop_triggers_finish(win, qapp, dialogs, monkeypatch):
    """托盘「结束并生成纪要」→ 复用 _stop() 的后台收尾，停止并导出。"""
    recorded, _ = dialogs
    exports, export_threads, finished = [], [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))

    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.title_input.setText("周会")

    win.bridge.finished.connect(lambda ok_, msg: finished.append((ok_, msg)))
    win.tray_stop_action.trigger()          # QAction.trigger() → _tray_stop → _stop

    assert _pump_until(qapp, lambda: bool(finished)), "托盘结束未走 _stop 收尾"
    assert _pump_until(qapp, lambda: win.start_btn.isEnabled())
    assert recorder.stop_calls == 1         # 走的是现有收尾线程
    assert recorder.stop_thread is not threading.current_thread()
    assert [e["title"] for e in exports] == ["周会"]
    assert recorded[-1][0] == "info" and "逐字稿.md" in recorded[-1][2]


def test_tray_menu_stop_idle_is_noop(win, dialogs, monkeypatch):
    """空闲时点「结束并生成纪要」：不得重复收尾 / 重复导出。"""
    recorded, _ = dialogs
    calls = []
    monkeypatch.setattr(appmod, "export_all",
                        lambda *a, **k: calls.append(a) or (Path("a"), Path("b")))
    win.recorder = None
    win.start_btn.setEnabled(True)

    win.tray_stop_action.trigger()

    assert calls == []
    assert [d for d in recorded if d[0] in ("info", "crit")] == []


def test_tray_stop_double_trigger_does_not_double_export(win, qapp, dialogs, monkeypatch):
    """收尾中再点「结束并生成纪要」：不得二次 stop()/导出（会生成重复纪要）。"""
    recorded, _ = dialogs
    exports, export_threads, finished = [], [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))

    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.bridge.finished.connect(lambda ok_, msg: finished.append((ok_, msg)))

    win.tray_stop_action.trigger()
    assert not win.stop_btn.isEnabled()      # 已进入收尾（stop 按钮禁用）
    win.tray_stop_action.trigger()           # 第二次点击必须被挡掉

    assert _pump_until(qapp, lambda: bool(finished))
    assert _pump_until(qapp, lambda: win.start_btn.isEnabled())
    qapp.processEvents()
    assert recorder.stop_calls == 1
    assert len(exports) == 1
    assert len(finished) == 1


class _FakeApp:
    """QApplication 替身：只记录 quit()（离屏无事件循环，真 quit 无从观测）。"""

    def __init__(self):
        self.quits = 0

    def quit(self):
        self.quits += 1


@pytest.fixture
def fake_app(monkeypatch):
    class _AppClass:
        _inst = _FakeApp()

        @staticmethod
        def instance():
            return _AppClass._inst

    monkeypatch.setattr(appmod, "QApplication", _AppClass)
    return _AppClass._inst


def test_tray_quit_with_recording_confirms(win, qapp, dialogs, fake_app):
    """录制中托盘「退出」：先弹确认；No 不动，Yes → 隐藏 + 后台收尾 + quit。"""
    recorded, answer = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.show()
    tray = win.tray

    # --- No：留在窗口里，不收尾、不退出 ---
    answer["ret"] = appmod.QMessageBox.StandardButton.No
    win.tray_quit_action.trigger()
    assert recorded[-1][0] == "question"
    assert recorded[-1][1] == "退出确认"
    assert "正在录制，退出将丢弃本次记录" in recorded[-1][2]
    assert recorder.stop_calls == 0
    assert fake_app.quits == 0
    assert win.isVisible()

    # --- Yes：隐藏 → 后台收尾 → 收尾完成后 quit ---
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes
    win.tray_quit_action.trigger()
    assert not win.isVisible()
    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and fake_app.quits == 1)
    assert recorder.stop_thread is not threading.current_thread()
    assert win.start_btn.isEnabled()            # 状态已复位
    assert win._quit_after_stop is False
    assert not tray.visible                     # 退出前收掉托盘图标


def test_tray_quit_without_recording_quits_directly(win, dialogs, fake_app):
    """空闲时托盘「退出」：不弹确认，直接退出（托盘图标收掉）。"""
    recorded, _ = dialogs
    win.recorder = None
    win.start_btn.setEnabled(True)

    win.tray_quit_action.trigger()

    assert [d for d in recorded if d[0] == "question"] == []
    assert fake_app.quits == 1
    assert not win.tray.visible


def test_tray_quit_quits_even_if_stop_raises(win, qapp, dialogs, fake_app):
    """收尾抛异常仍要退出：退出路径不能把用户锁死在托盘里。"""
    _, answer = dialogs
    recorder = FakeRecorder(stop_boom=True)
    _prepare_recording(win, recorder)
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes

    win.tray_quit_action.trigger()

    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and fake_app.quits == 1)
    assert not win.isVisible()


def test_tray_quit_while_finishing_does_not_stop_twice(win, qapp, dialogs, fake_app, monkeypatch):
    """收尾中托盘退出：确认文案改为「正在生成纪要」，且不再二次 stop()。"""
    recorded, answer = dialogs
    exports, export_threads, finished = [], [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))

    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.bridge.finished.connect(lambda ok_, msg: finished.append((ok_, msg)))
    win.tray_stop_action.trigger()           # 进入收尾
    assert not win.stop_btn.isEnabled()

    answer["ret"] = appmod.QMessageBox.StandardButton.Yes
    win.tray_quit_action.trigger()

    assert recorded[-1][0] == "question" and "正在生成纪要" in recorded[-1][2]
    assert fake_app.quits == 1
    assert recorder.stop_calls == 1          # 关键：没有二次 stop()
    assert not win.tray.visible
    # 被放弃的收尾线程若已跑完，回槽也不该崩（进程退出前最后一条消息）
    assert _pump_until(qapp, lambda: bool(finished))


def test_tray_quit_does_not_block_gui_thread(win, qapp, dialogs, fake_app):
    """Critical-2 回归：托盘退出也不得在 GUI 线程同步等 stop()（否则窗口冻结数分钟）。

    用「stop() 阻塞到测试放行」的假 recorder 量化：trigger() 必须远小于阻塞时长内返回
    （同步实现至少阻塞 0.5s），且此时还没 quit —— 退出必须等收尾真正结束。
    """
    _, answer = dialogs
    released = threading.Event()

    class _SlowStop(FakeRecorder):
        def stop(self):
            released.wait(timeout=5)      # 模拟慢收尾（真实场景可达数分钟）
            super().stop()

    recorder = _SlowStop()
    _prepare_recording(win, recorder)
    win.show()
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes

    t0 = time.monotonic()
    win.tray_quit_action.trigger()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"托盘退出在 GUI 线程阻塞了 {elapsed:.2f}s（窗口冻结）"
    assert not win.isVisible()
    assert fake_app.quits == 0            # 收尾还被卡着 —— 但界面已返回

    released.set()
    assert _pump_until(qapp, lambda: fake_app.quits == 1)
    assert recorder.stop_calls == 1


# ---------------------------------------------------------------------------
# d. 音频源自动模式：无 UI 选择，_probe_capture/_start 统一取 settings.audio_source
# ---------------------------------------------------------------------------

class _FakeProbeCapture:
    """AudioCapture 替身：只记录 source，不碰声卡。"""

    def __init__(self, source="speaker", sample_rate=16000):
        self.source = source
        self.sample_rate = sample_rate
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def read(self, timeout=1.0):
        return None


class _FakeStartedRecorder:
    """_start() 用的 Recorder 替身：记录构造参数，start() 立即返回（不起线程）。"""

    def __init__(self, settings, capture=None, audio_source=None):
        self.settings = settings
        self.capture = capture
        self.audio_source = audio_source
        self.store = TranscriptStore()
        self.title = ""
        self.started_title = None
        self.handlers = []
        self.update_handlers = []

    def on_segment(self, fn):
        self.handlers.append(fn)

    def on_update(self, fn):
        self.update_handlers.append(fn)

    def on_status(self, fn):
        self.handlers.append(fn)

    def start(self, title=""):
        self.started_title = title


def _install_fake_stack(monkeypatch):
    """把 _start 的重型依赖全部换成离屏替身，返回记录容器。"""
    made = {"capture": [], "recorder": []}

    def _capture_factory(source="both", sample_rate=16000):
        cap = _FakeProbeCapture(source=source, sample_rate=sample_rate)
        made["capture"].append(cap)
        return cap

    def _recorder_factory(settings, capture=None, asr=None, translator=None,
                          diarizer=None, audio_source=None):
        rec = _FakeStartedRecorder(settings, capture=capture, audio_source=audio_source)
        made["recorder"].append(rec)
        return rec

    monkeypatch.setattr(appmod, "AudioCapture", _capture_factory)
    monkeypatch.setattr(appmod, "Recorder", _recorder_factory)
    monkeypatch.setattr(appmod, "Translator", lambda settings: object())
    monkeypatch.setattr(appmod, "Diarizer", lambda settings: None)   # 无说话人，不影响录制
    return made


def test_no_audio_source_combo(win):
    """开箱即用：窗口里不再有音频源下拉（自动模式，线上/线下不用手动切换）。"""
    assert not hasattr(win, "source_combo"), "音频源下拉应已删除"
    assert win.title_input is not None


def _make_win(qapp, tmp_path, audio_source):
    w = MainWindow(_settings(tmp_path, audio_source=audio_source))
    w._timer.stop()
    return w


def test_probe_capture_uses_settings_audio_source(qapp, tmp_path, monkeypatch):
    """_probe_capture 按 settings.audio_source（.env 的 AUDIO_SOURCE）探测设备。"""
    made = _install_fake_stack(monkeypatch)
    w = _make_win(qapp, tmp_path, "mic")
    try:
        cap = w._probe_capture()
        assert cap is not None
        assert made["capture"][0].source == "mic", "探测用的不是 .env 配置的音频源"
    finally:
        w.recorder = None
        w.hide()


def test_probe_capture_defaults_to_both(qapp, tmp_path, monkeypatch):
    """默认值 both：老 .env 没写 AUDIO_SOURCE 时走自动双路。"""
    made = _install_fake_stack(monkeypatch)
    w = _make_win(qapp, tmp_path, "both")
    try:
        w._probe_capture()
        assert made["capture"][0].source == "both"
    finally:
        w.recorder = None
        w.hide()


def test_start_forwards_audio_source_and_reuses_probed_capture(qapp, tmp_path, monkeypatch):
    """_start 把同一 audio_source 传给 Recorder，并复用探测实例（不再开第二次设备）。"""
    made = _install_fake_stack(monkeypatch)
    # _on_finished(True) 会弹模态框，离屏环境必须静音弹窗
    monkeypatch.setattr(appmod.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.QMessageBox, "critical", lambda *a, **k: None)
    w = _make_win(qapp, tmp_path, "speaker")
    try:
        w._start()

        assert made["capture"], "未做设备探测（构造 AudioCapture）"
        assert made["capture"][0].source == "speaker"
        rec = made["recorder"][0]
        assert rec.audio_source == "speaker"
        assert rec.capture is made["capture"][0], "探测实例未被复用（会在录制时再开一次设备）"
        assert rec.handlers, "片段/状态回调未挂到 bridge"
        assert rec.update_handlers, "译文更新回调（on_update）未挂到 bridge"
        assert not w.start_btn.isEnabled()

        w._on_finished(True, "已生成")
        assert w.start_btn.isEnabled()
    finally:
        w.recorder = None
        w.hide()


def test_probe_capture_reports_failure_without_recording(win, dialogs, monkeypatch):
    """设备不可用：critical 弹框说明、返回 None、_start 直接放弃（recorder 保持 None）。"""
    recorded, _ = dialogs

    def _boom(source="speaker", sample_rate=16000):
        raise appmod.CaptureError("no such device")

    monkeypatch.setattr(appmod, "AudioCapture", _boom)

    assert win._probe_capture() is None
    assert recorded[-1][0] == "crit"
    assert "音频源不可用" in recorded[-1][2]

    # _start 走同一探测：失败即返回，不留半启动状态
    win._start()
    assert win.recorder is None
    assert win.start_btn.isEnabled()


# ---------------------------------------------------------------------------
# e. 两阶段显示：bridge.update 信号 → _on_update → subtitle.update_segment
# ---------------------------------------------------------------------------

class _FakeSubtitle:
    """字幕面板替身：记录 show_segment / update_segment 的调用（不建窗口）。"""

    def __init__(self):
        self.calls: list[tuple[str, int, str, str]] = []

    def show_segment(self, seg, name):
        self.calls.append(("show", seg.seq, name, seg.translation))

    def update_segment(self, seg, name):
        self.calls.append(("update", seg.seq, name, seg.translation))


def test_bridge_update_signal_forwarded_to_subtitle(win, qapp):
    """译文到达走独立通道：bridge.update → _on_update → subtitle.update_segment。

    两条通道必须分开：原文事件只能 show_segment（否则重复开块），译文只能
    update_segment（否则同一句话在面板里出现两行）。这里替换成记录型假面板，
    用真实信号投递（emit → 槽）验证接管与分工。
    """
    fake = _FakeSubtitle()
    win.subtitle = fake
    win.recorder = FakeRecorder()          # store 提供 display_name（空标签 → 发言人?）

    seg = Segment(start=0.0, end=1.0, text="hello", lang="en",
                  translation="", seq=1)
    assert hasattr(win.bridge, "update"), "RecorderBridge 缺少 update 信号"
    assert win.bridge.receivers(win.bridge.update) >= 1, "_on_update 未连接到 update 信号"

    # 第一阶段：原文事件
    win.bridge.segment.emit(seg)
    assert _pump_until(qapp, lambda: len(fake.calls) == 1)
    assert fake.calls[-1][:2] == ("show", 1)
    assert fake.calls[-1][3] == ""          # 原文阶段没有译文

    # 第二阶段：译文更新（同一个 Segment 被填充译文）
    seg.translation = "你好"
    win.bridge.update.emit(seg)
    assert _pump_until(qapp, lambda: len(fake.calls) == 2)
    assert fake.calls[-1][:2] == ("update", 1)
    assert fake.calls[-1][2] == "发言人?"     # display_name 走 recorder.store
    assert fake.calls[-1][3] == "你好"
    # 译文更新不得触发新增块路径
    assert [c[0] for c in fake.calls] == ["show", "update"]

    # 直接调槽同样只走 update（两个槽逻辑同源、通道不同）
    win._on_update(seg)
    assert [c[0] for c in fake.calls] == ["show", "update", "update"]
