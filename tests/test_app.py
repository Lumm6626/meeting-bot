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
import re
import time
from datetime import datetime, timedelta
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
def _no_preload(monkeypatch):
    """整个模块禁用后台预热：它在后台 import torch（持 GIL 数秒），会让依赖时序的
    用例（后台初始化线程）随机变慢/超时。真实启动行为由 test_preload_runs 覆盖。"""
    monkeypatch.setattr(appmod.MainWindow, "_preload_engine", lambda self: None)


@pytest.fixture
def win(qapp, tmp_path):
    w = MainWindow(_settings(tmp_path))
    # 关掉定时刷新（说话人 5s / 录制计时 1s）：避免其读写控件干扰「零控件访问」断言
    for timer_name in ("_timer", "_elapsed_timer"):
        timer = getattr(w, timer_name, None)
        if timer is not None:
            timer.stop()
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
# c. closeEvent：关闭主窗口 = 退出整个程序（录制中先确认并保存）
# ---------------------------------------------------------------------------

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


def test_close_event_idle_quits(win, dialogs):
    """未录制：直接关闭，不弹确认；字幕浮窗一并收起（否则进程退不掉）。"""
    recorded, _ = dialogs
    win.recorder = None
    win.start_btn.setEnabled(True)
    win.show()
    win.subtitle.show()

    assert win.close() is True
    assert not win.isVisible()
    assert not win.subtitle.isVisible()
    assert [d for d in recorded if d[0] == "question"] == []


def test_close_event_recording_cancel_keeps_running(win, dialogs):
    """录制中关窗 → 确认框；选「否」留在窗口里，录制不受影响。"""
    recorded, answer = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.show()
    answer["ret"] = appmod.QMessageBox.StandardButton.No

    assert win.close() is False
    assert win.isVisible()
    assert recorder.stop_calls == 0
    assert any(d[0] == "question" and "保存本次记录" in d[2] for d in recorded)


def test_close_event_recording_yes_exports_then_quits(win, qapp, dialogs, fake_app, monkeypatch):
    """录制中关窗选「是」：后台结束录制并导出，完成后再彻底退出（GUI 不冻结）。"""
    exports, export_threads, finished = [], [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _, answer = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.show()
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes

    t0 = time.monotonic()
    assert win.close() is False            # 后台收尾，GUI 立即返回
    assert time.monotonic() - t0 < 0.2
    assert win._quit_after_stop is True
    assert _pump_until(qapp, lambda: recorder.stop_calls == 1 and len(exports) == 1)
    assert _pump_until(qapp, lambda: fake_app.quits == 1)


def test_close_event_finishing_quits_immediately(win, dialogs, fake_app):
    """正在生成纪要时关窗：确认后立即退出（不再二次 stop）。"""
    recorded, answer = dialogs
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win.stop_btn.setEnabled(False)         # 收尾中：stop 已禁用
    win.show()
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes

    assert win.close() is False
    assert fake_app.quits == 1
    assert recorder.stop_calls == 0
    assert any("丢失本次结果" in d[2] for d in recorded)


def test_on_finished_quits_when_close_requested(win, dialogs, fake_app):
    """关窗触发的收尾完成 → 弹窗提示后彻底退出。"""
    recorder = FakeRecorder()
    _prepare_recording(win, recorder)
    win._quit_after_stop = True
    win._on_finished(True, "已生成：x.md")
    assert fake_app.quits == 1
    assert win._quit_after_stop is False


def test_app_icon_loads(win):
    """图标资源可被 Qt 加载，窗口标题栏使用它。"""
    ico = appmod.PROJECT_ROOT / "assets" / "app.ico"
    assert ico.is_file()
    assert not appmod.app_icon().isNull()
    assert not win.windowIcon().isNull()


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
    for timer_name in ("_timer", "_elapsed_timer"):
        timer = getattr(w, timer_name, None)
        if timer is not None:
            timer.stop()
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
        # _start 的重型初始化在后台线程（避免点击卡 6 秒）：等 Recorder 就绪
        assert _pump_until(qapp, lambda: bool(made["recorder"]) and w.recorder is not None,
                           timeout=5.0), "后台初始化未在 5 秒内完成"

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


# ---------------------------------------------------------------------------
# g. 暗色主题重设计：品牌行 / 配置胶囊 / 状态卡片 / 录制计时
# ---------------------------------------------------------------------------

def _themed_settings(tmp_path, **kwargs) -> Settings:
    """带主题相关字段的 Settings（默认取 .env.example 的值）。"""
    kw = dict(asr_provider="local", audio_source="both",
              translate_model="deepseek-v4-flash", whisper_model="small")
    kw.update(kwargs)
    return Settings("https://example.invalid/v1", "sk-fake", kw["translate_model"],
                    "deepseek-v4-pro", kw["whisper_model"], "",
                    tmp_path / "会议记录", tmp_path,
                    asr_provider=kw["asr_provider"], audio_source=kw["audio_source"])


def _stop_timers(w):
    for timer_name in ("_timer", "_elapsed_timer"):
        timer = getattr(w, timer_name, None)
        if timer is not None:
            timer.stop()


def test_config_badges_reflect_settings(qapp, tmp_path):
    """三个只读胶囊徽章反映 .env 配置：识别源 / 音频源 / 翻译模型（去 deepseek- 前缀）。"""
    cases = [
        (dict(asr_provider="groq", audio_source="both",
              translate_model="deepseek-v4-flash"),
         ("识别 · Groq 云端", "音频 · 自动双路", "翻译 · v4-flash")),
        (dict(asr_provider="local", audio_source="speaker",
              translate_model="deepseek-v4-pro"),
         ("识别 · 本地 small", "音频 · 仅系统声音", "翻译 · v4-pro")),
        (dict(asr_provider="local", audio_source="mic",
              translate_model="gpt-4o-mini"),
         ("识别 · 本地 small", "音频 · 仅麦克风", "翻译 · gpt-4o-mini")),
    ]
    for kwargs, expected in cases:
        w = MainWindow(_themed_settings(tmp_path, **kwargs))
        try:
            got = (w.pill_asr.text(), w.pill_audio.text(), w.pill_translate.text())
            assert got == expected, f"{kwargs} → {got}"
            for pill in (w.pill_asr, w.pill_audio, w.pill_translate):
                assert pill.property("role") == "pill"      # 走 QSS 的胶囊样式
        finally:
            _stop_timers(w)
            w.recorder = None
            w.start_btn.setEnabled(True)
            w.hide()


def test_brand_row_status_card_and_equal_height_buttons(win):
    """品牌行（48px 图标 + 标题/副标题）、状态卡片、底部提示在位，主次按钮等高 44px。"""
    for name in WIDGETS:
        assert hasattr(win, name), f"控件 {name} 缺失（控件名必须保持不变）"
    assert win.brand_title.text() == "会议翻译助手"
    assert win.brand_sub.text() == "实时翻译字幕 · 会后自动纪要"
    assert not win.icon_label.pixmap().isNull(), "品牌行未加载 assets/app.ico"
    assert win.icon_label.width() == win.icon_label.height() == 48
    for btn in (win.start_btn, win.stop_btn):
        assert btn.minimumHeight() == btn.maximumHeight() == 44      # 固定等高
    assert win.start_btn.property("role") == "primary"               # 主按钮（渐变）
    assert win.stop_btn.property("role") == "secondary"              # 次按钮（描边）
    assert win.footer_hint.text().startswith("录制中关闭窗口将收进托盘")
    assert win.elapsed_label.text() == "就绪"


def _elapsed_recorder(seconds_ago: int = 65, segments: int = 2) -> FakeRecorder:
    """录制中的 fake recorder：started_at 回溯 N 秒、store 里预置 M 条语句。"""
    rec = FakeRecorder(title="计时会议")
    rec.started_at = datetime.now() - timedelta(seconds=seconds_ago)
    for i in range(segments):
        rec.store.add(start=float(i), end=float(i) + 1.0, text=f"句子{i}")
    return rec


def test_elapsed_label_updates_while_recording(win):
    """录制中 elapsed_label 显示「已录制 mm:ss · 语句 N 条」；空闲 / 未开始显示「就绪」。"""
    assert win.elapsed_label.text() == "就绪"
    assert win._elapsed_timer.interval() == 1000          # 每秒刷新一次

    rec = _elapsed_recorder(seconds_ago=65, segments=2)
    _prepare_recording(win, rec)

    text = win._refresh_elapsed()
    assert "01:05" in text, text
    assert "语句 2 条" in text, text
    assert win.elapsed_label.text() == text

    # 语句继续增长 → 计数跟着走（计时起点不变）
    rec.store.add(start=9.0, end=10.0, text="句子3")
    assert "语句 3 条" in win._refresh_elapsed()
    assert "01:05" in win.elapsed_label.text()

    # 收尾完成 / 空闲（start 按钮恢复）→ 回到「就绪」
    win.start_btn.setEnabled(True)
    win._refresh_elapsed()
    assert win.elapsed_label.text() == "就绪"

    # 从未开始录制（recorder 为 None）也不能崩
    win.recorder = None
    assert win._refresh_elapsed() == "就绪"


def test_entry_point_and_main_exist():
    """回归：app.main 必须存在且可导入（曾被正则手术误删 → 双击快捷方式毫无反应）。"""
    import meeting_bot.__main__  # noqa: F401  （导入即验证入口模块完整）
    assert callable(appmod.main)


def test_title_defaults_to_timestamp_and_refreshes(win):
    """会议名默认「年-月-日 时:分」；用户没手改时，点开始时刷新成实际开始时间。"""
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", win.title_input.text())
    first = win.title_input.text()
    win._mark_title_edited("手改的名字")     # 模拟用户手改（textEdited 信号）
    win.title_input.setText("手改的名字")
    win._title_edited = True
    assert win.title_input.text() == "手改的名字"    # 手改后不被覆盖


def test_brand_pixmap_logical_size_matches_label(win, qapp):
    """回归：品牌图标逻辑尺寸必须等于控件尺寸（高 DPI 下二次乘 DPR 会只显示左上 1/4）。"""
    from meeting_bot.app import brand_pixmap
    pm = brand_pixmap(48)
    assert not pm.isNull()
    logical_w = pm.width() / pm.devicePixelRatio()
    logical_h = pm.height() / pm.devicePixelRatio()
    assert abs(logical_w - 48) <= 0.5, f"图标逻辑宽 {logical_w} ≠ 48（会被裁切）"
    assert abs(logical_h - 48) <= 0.5
    assert win.icon_label.width() == 48 and win.icon_label.height() == 48
