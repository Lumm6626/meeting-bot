# -*- coding: utf-8 -*-
"""Task 12 修复轮 2：app.py 的线程红线与退出路径固化测试（离屏，无网络 / 无录音）。

两条不变量来自修复轮 1 的临时验证脚本（.superpowers/sdd/.../task-12-fix1-verify.py，
该目录不入版本控制），此处固化成常驻回归，防止后续改动悄悄回退：

  a. ``_stop()`` 的 finish() 线程内零 Qt 控件访问（Major-1 的线程红线）；
  b. ``closeEvent`` 的三个分支：未录制 / 确认框选 No / 选 Yes；
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

import meeting_bot.app as appmod
from meeting_bot.app import MainWindow
from meeting_bot.config import Settings
from meeting_bot.transcript import TranscriptStore

WIDGETS = ("title_input", "status", "start_btn", "stop_btn", "speaker_table")


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------

def _settings(tmp_path) -> Settings:
    return Settings("https://example.invalid/v1", "sk-fake", "m1", "m2", "medium",
                    "", tmp_path / "会议记录", tmp_path)


@pytest.fixture(scope="module")
def qapp():
    """离屏 QApplication（整模块复用；无事件循环，需要时手动 processEvents）。"""
    app = QApplication.instance() or QApplication([])
    yield app


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
# c. closeEvent 三个分支（Major-4 防静默丢数据）
# ---------------------------------------------------------------------------

def test_close_event_without_recorder_accepts_without_dialog(win, dialogs):
    """未录制：直接退出，不弹确认框（既有行为不变）。"""
    recorded, _ = dialogs
    win.recorder = None
    win.start_btn.setEnabled(True)
    win.show()
    assert win.isVisible()

    assert win.close() is True
    assert [d for d in recorded if d[0] == "question"] == []


def test_close_event_confirm_branches(win, qapp, dialogs):
    """录制中：选 No → 不退出、不收尾；选 Yes → 隐藏窗口 + 后台收尾 + 完成后再关闭。"""
    recorded, answer = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.show()
    gui_thread = threading.current_thread()

    # --- No：留在窗口里，不触发收尾 ---
    answer["ret"] = appmod.QMessageBox.StandardButton.No
    assert win.close() is False
    assert win.isVisible()
    assert recorder.stop_calls == 0
    assert recorded[-1][0] == "question" and recorded[-1][1] == "退出确认"
    assert "退出将丢弃本次记录" in recorded[-1][2]

    # --- Yes：第一次 close 立即返回（收尾在后台，不冻结 GUI 线程）---
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes
    assert win.close() is False            # e.ignore()：真正的关闭等收尾完成
    assert not win.isVisible()             # 窗口已隐藏，而非冻结在屏幕上

    # 收尾完成（bridge.shutdown_done 队列投递）→ _finish_close 复位并 close()
    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and not win._closing)
    assert recorder.stop_thread is not gui_thread   # 关键：stop() 不在 GUI 线程跑
    assert win.start_btn.isEnabled()                # 复位：二次 close 不会再弹框
    assert win.close() is True                      # 二次 close 走 e.accept()


def test_close_event_accepts_when_recorder_stop_raises(win, qapp, dialogs):
    """收尾抛异常仍要放行退出：退出路径不能把用户锁死在窗口里（异步路径同样成立）。"""
    _, answer = dialogs
    recorder = FakeRecorder(stop_boom=True)
    _prepare_recording(win, recorder)
    win.show()
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes

    assert win.close() is False
    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and not win._closing)
    assert win.start_btn.isEnabled()
    assert win.close() is True


def test_close_event_does_not_block_gui_thread(win, qapp, dialogs):
    """Critical-2 回归：closeEvent 不得在 GUI 线程同步等 stop()。

    recorder.stop() 最坏要等 10s（producer 收尾）+ 20s（翻译排空）+ 补译 + 180s（标注
    线程）同步调用会把窗口冻结数分钟。用「stop() 阻塞到测试放行」的假 recorder 量化：
    close() 必须在远小于阻塞时长内返回（修复前为同步 → 至少阻塞 0.5s）。
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
    assert win.close() is False
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"closeEvent 在 GUI 线程阻塞了 {elapsed:.2f}s（窗口冻结）"
    assert not win.isVisible()
    assert recorder.stop_calls == 0       # 收尾还被卡着 —— 但界面已经返回了

    released.set()
    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and not win._closing)
    assert win.close() is True
