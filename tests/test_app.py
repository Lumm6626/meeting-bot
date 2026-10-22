# -*- coding: utf-8 -*-
"""Task 12 修复轮 2：app.py 的线程红线与退出路径固化测试（离屏，无网络 / 无录音）。

两条不变量来自修复轮 1 的临时验证脚本（.superpowers/sdd/.../task-12-fix1-verify.py，
该目录不入版本控制），此处固化成常驻回归，防止后续改动悄悄回退：

  a. ``_stop()`` 的 finish() 线程内零 Qt 控件访问（Major-1 的线程红线）；
  b. ``closeEvent``：关闭主窗口 = 退出程序（录制中先确认并保存产物）；
  c. finish() 内 recorder.stop() 抛异常也必须走到 finished(False)（New-6 兜底）。

全程离屏 + 假对象：不构造真 Recorder、不启动采集/识别/翻译、不发任何网络请求。
"""
import dataclasses
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 必须在 import PyQt6 之前

import threading
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QTableWidgetItem
from PyQt6.QtWidgets import QSystemTrayIcon as _RealTray

import meeting_bot.app as appmod
from meeting_bot.app import MainWindow
from meeting_bot.config import Settings
from meeting_bot.transcript import Segment, TranscriptStore
from meeting_bot.voiceprints import VoiceprintLibrary

WIDGETS = ("title_input", "terms_input", "status", "start_btn", "stop_btn", "speaker_table")



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
def _no_secrets_probe(monkeypatch):
    """默认不改状态栏提示：避免读开发者本机凭据库/环境变量，断言可复现。

    需要提示的用例在测试体内再 monkeypatch 成 ".env"（后设的覆盖本 fixture）。
    """
    monkeypatch.setattr(appmod, "secrets_source", lambda: "none")


@pytest.fixture(autouse=True)
def _no_embedding_model(monkeypatch):
    """默认不加载真声纹模型：它在后台 import torch 并去 HF 拉权重（无缓存时会重试
    到超时），本模块其余用例并不需要它。需要声纹行为的用例在测试体内再 monkeypatch
    成自己的 fake（后设的覆盖本 fixture）。"""
    monkeypatch.setattr(appmod, "compute_speaker_embeddings", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def _no_preload(monkeypatch):
    """整个模块禁用后台预热：它在后台 import torch（持 GIL 数秒），会让依赖时序的
    用例（后台初始化线程）随机变慢/超时。真实启动行为由 test_preload_runs 覆盖。"""
    monkeypatch.setattr(appmod.MainWindow, "_preload_engine", lambda self: None)


@pytest.fixture
def win(qapp, tmp_path):
    w = MainWindow(_settings(tmp_path))
    # 声纹库指到 tmp_path：测试绝不读写真仓库里的 声纹库.json（生物特征数据）
    w._voiceprints = VoiceprintLibrary(tmp_path / "声纹库.json")
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
# b2. 会后离线说话人标注：初版产物先出 → 后台补标注 → 覆盖同名产物
# ---------------------------------------------------------------------------

class _DiarizableRecorder(FakeRecorder):
    """带「整场录音」的 recorder 替身：会后标注读它跑 pyannote。"""

    def __init__(self, audio_path, title: str = "会后标注会议"):
        super().__init__(title=title)
        self.audio_path = audio_path
        self.store.add(0.0, 4.0, "hello world", lang="en")


def _write_wav(tmp_path) -> Path:
    """写一秒 16kHz 单声道 wav（真实文件：会后标注线程会去读它）。"""
    import numpy as np
    import soundfile as sf
    path = tmp_path / "rec_test.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    return path


def _fake_diarize_tools(monkeypatch, calls):
    """monkeypatch Diarizer / assign_speakers，记录构造与调用（不加载 torch）。"""
    class _FakeDiarizer:
        def __init__(self, settings):
            calls["init"] = settings

        def label(self, audio, sample_rate=16000):
            calls["label"] = (len(audio), sample_rate)
            return [(0.0, 4.0, "SPEAKER_00")]

    def _fake_assign(segs, spans):
        for seg in segs:
            seg.speaker = spans[0][2]
        calls["assigned"] = True
        return segs

    monkeypatch.setattr(appmod, "Diarizer", _FakeDiarizer)
    monkeypatch.setattr(appmod, "assign_speakers", _fake_assign)


def test_finish_runs_offline_diarize_and_re_exports(win, qapp, dialogs, monkeypatch, tmp_path):
    """配了 HF_TOKEN：finish 流程先出初版，再后台标注并重新导出（共两次 export_all）。

    断言四条不变量：① 初版 + 标注版各导一次；② 标注线程读到的是整场 wav（16kHz）；
    ③ 期间状态栏出现「标注」提示；④ bridge.diarized 报告成功。
    """
    exports, export_threads, calls, diarized, statuses = [], [], {}, [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)

    win.bridge.diarized.connect(lambda ok, msg: diarized.append((ok, msg)))
    win.bridge.status.connect(statuses.append)

    win._stop()

    assert _pump_until(qapp, lambda: len(exports) == 2 and bool(diarized), timeout=15.0), (
        f"会后标注未完成：exports={len(exports)} diarized={diarized} statuses={statuses}")
    # ① 初版（finish 线程）+ 标注版（会后线程）
    assert "init" in calls and "assigned" in calls
    # ② 整场 wav：16000 采样、16000 样本（1 秒）
    assert calls["label"] == (16000, 16000)
    # 语句被标上说话人（重新导出时才带得出来）
    assert recorder.store.segments()[0].speaker == "SPEAKER_00"
    assert calls["assigned"] is True
    # ③ 期间提示 + ④ 完成信号
    assert any("标注" in m for m in statuses), f"状态栏没提示标注中：{statuses}"
    assert any("0.5~1.5 倍" in m for m in statuses)
    assert diarized[0][0] is True and "已更新逐字稿" in diarized[0][1]
    assert "标注" in win.status.text()
    # 标注是「覆盖同名产物」：两次导出必须拿到同一个 store / title / 时间
    assert exports[0]["store"] is exports[1]["store"] is recorder.store
    assert exports[0]["title"] == exports[1]["title"] == win.title_input.text().strip()
    assert exports[0]["started_at"] == exports[1]["started_at"] == recorder.started_at


def test_offline_diarize_skipped_without_hf_token(win, qapp, dialogs, monkeypatch, tmp_path):
    """没有 HF_TOKEN：只出初版产物，绝不起标注线程（无模型可用）。"""
    exports, export_threads, calls, diarized = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    win.settings = dataclasses.replace(win.settings, hf_token="")

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)

    win.bridge.diarized.connect(lambda ok, msg: diarized.append((ok, msg)))
    win._stop()

    assert _pump_until(qapp, lambda: bool(exports) and win.start_btn.isEnabled())
    # 再给标注线程一段机会（若被误触发，第二次 export 会在这期间出现）
    deadline = time.time() + 0.5
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(exports) == 1, f"无 token 却触发了会后标注：{len(exports)} 次导出"
    assert "init" not in calls and not diarized


def test_offline_diarize_skipped_without_audio(win, qapp, dialogs, monkeypatch):
    """配了 token 但没落盘音频（SAVE_AUDIO=false）：跳过并提示一句，不报错。"""
    exports, export_threads, calls, statuses = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")

    recorder = _DiarizableRecorder(None)
    _prepare_recording(win, recorder)

    win.bridge.status.connect(statuses.append)
    win._stop()

    assert _pump_until(qapp, lambda: bool(exports) and win.start_btn.isEnabled())
    deadline = time.time() + 0.5
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(exports) == 1
    assert "init" not in calls
    assert any("SAVE_AUDIO" in m for m in statuses), f"未提示缺音频：{statuses}"


def test_offline_diarize_failure_only_warns(win, qapp, dialogs, monkeypatch, tmp_path):
    """标注失败（模型起不来等）：只发 diarized(False) 告警，不抛穿线程、不改产物。"""
    exports, export_threads, diarized = [], [], []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))

    class _BoomDiarizer:
        def __init__(self, settings):
            raise RuntimeError("模型下载失败")

    monkeypatch.setattr(appmod, "Diarizer", _BoomDiarizer)
    monkeypatch.setattr(appmod, "assign_speakers", lambda segs, spans: segs)
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)

    win.bridge.diarized.connect(lambda ok, msg: diarized.append((ok, msg)))
    win._stop()

    assert _pump_until(qapp, lambda: bool(diarized), timeout=15.0)
    ok, msg = diarized[0]
    assert ok is False and "模型下载失败" in msg
    assert len(exports) == 1, "标注失败后不应重新导出（产物保持初版）"
    assert len(recorder.store.segments()) == 1


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
    """把 _start 的重型依赖全部换成离屏替身，返回记录容器（translator 也记 settings）。"""
    made = {"capture": [], "recorder": [], "translator": []}

    def _capture_factory(source="both", sample_rate=16000):
        cap = _FakeProbeCapture(source=source, sample_rate=sample_rate)
        made["capture"].append(cap)
        return cap

    def _recorder_factory(settings, capture=None, asr=None, translator=None,
                          diarizer=None, audio_source=None, store=None):
        rec = _FakeStartedRecorder(settings, capture=capture, audio_source=audio_source)
        made["recorder"].append(rec)
        return rec

    def _translator_factory(settings):
        made["translator"].append(settings)
        return object()

    monkeypatch.setattr(appmod, "AudioCapture", _capture_factory)
    monkeypatch.setattr(appmod, "Recorder", _recorder_factory)
    monkeypatch.setattr(appmod, "Translator", _translator_factory)
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
                           timeout=20.0), "后台初始化超时"

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
    assert win.footer_hint.text().startswith("录制中关闭窗口会先结束录制并保存")
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


@pytest.fixture(autouse=True)
def _silence_messageboxes(monkeypatch):
    """兜底：任何用例都不允许弹真实模态框（离屏下会直接卡死整个测试进程）。"""
    for name, ret in (("information", None), ("critical", None), ("warning", None)):
        monkeypatch.setattr(appmod.QMessageBox, name, staticmethod(lambda *a, **k: ret))
    monkeypatch.setattr(appmod.QMessageBox, "question",
                        staticmethod(lambda *a, **k: appmod.QMessageBox.StandardButton.No))


def test_recovery_ignores_journal_created_by_current_session(win, qapp, tmp_path):
    """回归：本场刚开始录制时，不能把自己刚建的日志误判成「上次未完成会议」弹窗。"""
    import json
    w = win
    # 模拟启动快照为空 → 之后本场录制创建了日志
    w._pending_journals = []
    w._check_recovery()          # 不应弹窗、不应恢复
    assert w._pending_journals == []


# ---------------------------------------------------------------------------
# e. 本场术语/热词：合并进 asr_prompt，只对本次录制生效
# ---------------------------------------------------------------------------

def _make_terms_win(qapp, tmp_path, **overrides):
    """带 asr_prompt 的窗口（不经 load_settings，直接 dataclasses.replace）。"""
    from dataclasses import replace

    w = MainWindow(replace(_settings(tmp_path), **overrides))
    for timer_name in ("_timer", "_elapsed_timer"):
        timer = getattr(w, timer_name, None)
        if timer is not None:
            timer.stop()
    return w


def test_merge_asr_prompt_unit():
    """纯函数：分隔符/去空白/去重（大小写不敏感、术语在前）、空术语原样返回。"""
    assert appmod.merge_asr_prompt("", "过敏门诊，复合药房、Jebel Ali, a;b；c") ==         "过敏门诊, 复合药房, Jebel Ali, a, b, c"
    # 术语为空：连空格都不动（零行为变化的硬要求）
    assert appmod.merge_asr_prompt("allergy,pricing", "") == "allergy,pricing"
    assert appmod.merge_asr_prompt("allergy, pricing", "  、 ，;") == "allergy, pricing"
    # 用户术语优先保留，.env 里重复的词只留一次
    assert appmod.merge_asr_prompt("allergy, pricing", "Allergy、过敏门诊") ==         "Allergy, 过敏门诊, pricing"


def test_terms_input_widget_present(win):
    """会议名下方有「术语/热词」输入框，placeholder 说明格式，样式沿用 QLineEdit。"""
    assert win.terms_input.placeholderText().startswith("术语/热词")
    assert "逗号分隔" in win.terms_input.placeholderText()
    assert win.terms_input.text() == ""
    # 布局顺序：会议名在上、术语在其下方（同一列）
    layout = win.centralWidget().layout()
    items = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert items.index(win.terms_input) == items.index(win.title_input) + 1


def test_start_merges_terms_into_asr_prompt(qapp, tmp_path, monkeypatch):
    """填了术语：Recorder 与 Translator 收到合并后的 asr_prompt（术语在前）。"""
    made = _install_fake_stack(monkeypatch)
    monkeypatch.setattr(appmod.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.QMessageBox, "critical", lambda *a, **k: None)
    base = "allergy clinic, immunotherapy"
    w = _make_terms_win(qapp, tmp_path, asr_prompt=base)
    try:
        w.terms_input.setText("过敏门诊，复合药房、Jebel Ali")
        w._start()
        assert _pump_until(qapp, lambda: bool(made["recorder"]) and w.recorder is not None,
                           timeout=20.0), "后台初始化超时"

        expected = "过敏门诊, 复合药房, Jebel Ali, allergy clinic, immunotherapy"
        rec_settings = made["recorder"][0].settings
        assert rec_settings.asr_prompt == expected, rec_settings.asr_prompt
        assert made["translator"][0].asr_prompt == expected, "Translator 没用合并后的配置"
        assert made["translator"][0] is rec_settings, "Recorder/Translator 必须共用同一个 run_settings"
        # 其余配置原样（只换 asr_prompt）
        assert rec_settings.output_dir == w.settings.output_dir
        assert rec_settings.asr_provider == w.settings.asr_provider
        # 本场术语不写回全局配置（下一场输入框为空 → 回到 .env 原值）
        assert w.settings.asr_prompt == base
    finally:
        w.recorder = None
        w.start_btn.setEnabled(True)
        w.hide()


def test_start_without_terms_keeps_settings_object(qapp, tmp_path, monkeypatch):
    """术语为空：直接沿用 self.settings（不 replace），行为与加功能前完全一致。"""
    made = _install_fake_stack(monkeypatch)
    monkeypatch.setattr(appmod.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.QMessageBox, "critical", lambda *a, **k: None)
    w = _make_terms_win(qapp, tmp_path, asr_prompt="allergy clinic")
    try:
        w._start()
        assert _pump_until(qapp, lambda: bool(made["recorder"]), timeout=5.0)
        assert made["recorder"][0].settings is w.settings
        assert made["translator"][0] is w.settings
        assert w.settings.asr_prompt == "allergy clinic"
    finally:
        w.recorder = None
        w.start_btn.setEnabled(True)
        w.hide()


def test_start_terms_without_env_prompt(qapp, tmp_path, monkeypatch):
    """.env 没配 ASR_PROMPT 时，术语本身就该成为识别提示词（不是空串）。"""
    made = _install_fake_stack(monkeypatch)
    monkeypatch.setattr(appmod.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.QMessageBox, "critical", lambda *a, **k: None)
    w = _make_terms_win(qapp, tmp_path)
    try:
        w.terms_input.setText("复合药房, Jebel Ali")
        w._start()
        assert _pump_until(qapp, lambda: bool(made["recorder"]), timeout=5.0)
        assert made["recorder"][0].settings.asr_prompt == "复合药房, Jebel Ali"
    finally:
        w.recorder = None
        w.start_btn.setEnabled(True)
        w.hide()


# ---------------------------------------------------------------------------
# f. 录制中醒目提示：窗口标题 + 字幕浮窗红点（keyring 之外的用户感知）
# ---------------------------------------------------------------------------

class _RecordingSubtitle:
    """字幕面板替身：记录 set_recording 调用（其余接口空实现）。"""

    def __init__(self):
        self.recording_calls: list[bool] = []

    def set_recording(self, active: bool):
        self.recording_calls.append(bool(active))

    def show_segment(self, *_a, **_k):
        pass

    def update_segment(self, *_a, **_k):
        return False

    def hide(self):
        pass


def test_start_and_finish_toggle_recording_ui(qapp, tmp_path, monkeypatch):
    """_start → 标题含「录音中」+ set_recording(True)；_on_finished → 全部复位。"""
    _install_fake_stack(monkeypatch)
    monkeypatch.setattr(appmod.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.QMessageBox, "critical", lambda *a, **k: None)
    w = _make_win(qapp, tmp_path, "both")
    fake = _RecordingSubtitle()
    w.subtitle = fake
    try:
        assert w.windowTitle() == appmod.WINDOW_TITLE

        w._start()
        assert "录音中" in w.windowTitle(), "开始录制后主窗口标题没有提示录音中"
        assert fake.recording_calls == [True], "未通知字幕浮窗显示录制提示"
        assert _pump_until(qapp, lambda: w.recorder is not None, timeout=5.0)

        w._on_finished(True, "已生成")
        assert w.windowTitle() == appmod.WINDOW_TITLE, "收尾完成后标题未复位"
        assert fake.recording_calls[-1] is False
        assert w.start_btn.isEnabled()
    finally:
        w.recorder = None
        w.hide()


def test_plaintext_secrets_hint_in_status(qapp, tmp_path, monkeypatch):
    """密钥还在 .env 明文 → 启动状态栏追加一条轻提示（不弹窗）。"""
    monkeypatch.setattr(appmod, "secrets_source", lambda: ".env")
    w = MainWindow(_settings(tmp_path))
    for timer_name in ("_timer", "_elapsed_timer"):
        timer = getattr(w, timer_name, None)
        if timer is not None:
            timer.stop()
    try:
        text = w.status.text()
        assert ".env" in text and "凭据管理器" in text
        assert "setup_env.py" in text
    finally:
        w.recorder = None
        w.hide()


def test_no_hint_when_secrets_in_keyring(qapp, tmp_path, monkeypatch):
    """已在凭据管理器：状态栏保持干净的「就绪」，不打扰用户。"""
    monkeypatch.setattr(appmod, "secrets_source", lambda: "keyring")
    w = MainWindow(_settings(tmp_path))
    for timer_name in ("_timer", "_elapsed_timer"):
        timer = getattr(w, timer_name, None)
        if timer is not None:
            timer.stop()
    try:
        assert w.status.text() == "就绪"
    finally:
        w.recorder = None
        w.hide()


# ---------------------------------------------------------------------------
# h. 说话人改名 → 回填整场字幕 + 广播改名事件（声纹学习的数据来源）
# ---------------------------------------------------------------------------

class _RenameRecorder(FakeRecorder):
    """FakeRecorder + 改名事件记录（模拟 A2 声纹学习的消费端）。"""

    def __init__(self):
        super().__init__()
        self.renames: list[tuple[str, str]] = []
        self.rename_listeners: list = []

    def on_speaker_renamed(self, fn):
        self.rename_listeners.append(fn)

    def emit_speaker_renamed(self, label, name):
        self.renames.append((label, name))
        for fn in list(self.rename_listeners):
            fn(label, name)


class _RenameSubtitle:
    """记录 refresh_names 调用的假字幕面板（name_resolver 只是个可写属性）。"""

    def __init__(self):
        self.name_resolver = None
        self.refresh_calls = 0
        self.refreshed_names: list = []

    def show_segment(self, *_a, **_k):
        pass

    def update_segment(self, *_a, **_k):
        return False

    def refresh_names(self):
        self.refresh_calls += 1
        # 记录「那一刻 resolver 会显示什么」= 改名是否已经落进 store
        if callable(self.name_resolver):
            self.refreshed_names.append(self.name_resolver("SPEAKER_00"))

    def hide(self):
        pass


def _rename_in_table(win, label: str, name: str):
    """在说话人表格里改名字并触发真实的 _rename_speaker 链路。"""
    win.speaker_table.blockSignals(True)          # 先静音：只为摆数据，不触发回调
    win.speaker_table.setRowCount(1)
    win.speaker_table.setItem(0, 0, QTableWidgetItem(label))
    item = QTableWidgetItem(name)
    win.speaker_table.setItem(0, 1, item)
    win.speaker_table.blockSignals(False)
    win._rename_speaker(item)                     # 等价于用户按下回车
    return item


def test_rename_speaker_refreshes_subtitle_and_emits_event(win, qapp):
    """改名一次要触发三件事：落 store、字幕面板重渲染、recorder 广播 (label, name)。"""
    fake_sub = _RenameSubtitle()
    win.subtitle = fake_sub
    rec = _RenameRecorder()
    rec.store.register_label("SPEAKER_00")
    win.recorder = rec

    _rename_in_table(win, "SPEAKER_00", "张三")
    _pump_until(qapp, lambda: True, timeout=0.05)

    assert rec.store.display_name("SPEAKER_00") == "张三"   # ① 落 store
    assert fake_sub.refresh_calls == 1, "改名后未让字幕面板重渲染已显示的块"  # ②
    assert fake_sub.refreshed_names == ["张三"], "重渲染时 store 里还是旧名字（顺序错了）"
    assert fake_sub.name_resolver is not None, "没把 store.display_name 接成解析器"
    assert rec.renames == [("SPEAKER_00", "张三")]          # ③ 广播给声纹学习


def test_rename_speaker_ignores_label_column_and_empty_recorder(win, qapp):
    """改「声纹标签」列 / 无 recorder 时不做事（不炸、不广播）。"""
    fake_sub = _RenameSubtitle()
    win.subtitle = fake_sub
    win.recorder = None
    win.speaker_table.blockSignals(True)
    win.speaker_table.setRowCount(1)
    win.speaker_table.setItem(0, 0, QTableWidgetItem("SPEAKER_00"))
    label_item = win.speaker_table.item(0, 0)
    win.speaker_table.setItem(0, 1, QTableWidgetItem("张三"))
    win.speaker_table.blockSignals(False)

    win._rename_speaker(label_item)          # 第 0 列：不是名字列
    assert fake_sub.refresh_calls == 0

    rec = _RenameRecorder()
    win.recorder = rec
    win._rename_speaker(win.speaker_table.item(0, 1))   # 有 recorder，正常走一遍
    assert rec.renames == [("SPEAKER_00", "张三")]

    win.recorder = None                      # 活动结束（recorder 置空）后再改：安全返回
    win._rename_speaker(win.speaker_table.item(0, 1))
    assert fake_sub.refresh_calls == 1


def test_rename_callback_receives_store_bound_speaker_event(win, qapp):
    """注册在 recorder 上的回调（A2 的入口）收到改名事件，且带最终名字。"""
    rec = _RenameRecorder()
    rec.store.register_label("SPEAKER_01")
    win.recorder = rec
    win.subtitle = _RenameSubtitle()
    seen: list[tuple[str, str]] = []
    rec.on_speaker_renamed(lambda label, name: seen.append((label, name)))

    _rename_in_table(win, "SPEAKER_01", "李四")
    assert seen == [("SPEAKER_01", "李四")]


def test_rename_updates_real_subtitle_block_header(win, qapp):
    """端到端：真面板 + 真 store —— 改名前头行是 SPEAKER_00，改名后立刻变「张三」。"""
    rec = _RenameRecorder()
    win.recorder = rec
    seg = rec.store.add(0.0, 2.0, "hello", lang="en")
    rec.store.update(seg.seq, speaker="SPEAKER_00", translation="你好")
    seg = rec.store.segments()[0]
    win.subtitle.name_resolver = None        # 先清干净：下面由 _on_segment 懒接线

    win._on_segment(seg)
    _pump_until(qapp, lambda: True, timeout=0.05)
    assert win.subtitle.entries()[0].meta.text() == "[00:00:00]  SPEAKER_00"
    assert callable(win.subtitle.name_resolver), "_on_segment 未接上 store.display_name"

    _rename_in_table(win, "SPEAKER_00", "张三")
    _pump_until(qapp, lambda: True, timeout=0.05)
    # 已显示的块立刻换名字，正文不动
    assert win.subtitle.entries()[0].meta.text() == "[00:00:00]  张三"
    assert win.subtitle.entries()[0].main.text() == "你好"
    assert len(win.subtitle.entries()) == 1


def test_rename_to_same_name_does_not_rebroadcast(win, qapp):
    """同名重复触发（程序性重设 / 只改格式）不白刷新、不给声纹学习喂重复样本。"""
    fake_sub = _RenameSubtitle()
    win.subtitle = fake_sub
    rec = _RenameRecorder()
    rec.store.register_label("SPEAKER_00")
    win.recorder = rec

    _rename_in_table(win, "SPEAKER_00", "张三")
    assert rec.renames == [("SPEAKER_00", "张三")]
    assert fake_sub.refresh_calls == 1

    _rename_in_table(win, "SPEAKER_00", "张三")      # 名字没变
    assert rec.renames == [("SPEAKER_00", "张三")], "同名不该再广播一次"
    assert fake_sub.refresh_calls == 1, "同名不该再重渲染整场字幕"


def test_refresh_speakers_skips_while_user_is_editing(win, qapp):
    """5 秒定时刷新不能重设正在编辑的单元格（重设会把输入框里的字清成空串）。"""
    rec = _RenameRecorder()
    rec.store.register_label("SPEAKER_00")
    win.recorder = rec
    win.show()
    win.speaker_table.blockSignals(True)
    win.speaker_table.setRowCount(1)
    win.speaker_table.setItem(0, 0, QTableWidgetItem("SPEAKER_00"))
    name_item = QTableWidgetItem("张三")
    win.speaker_table.setItem(0, 1, name_item)
    win.speaker_table.blockSignals(False)
    _pump_until(qapp, lambda: True, timeout=0.05)

    win.speaker_table.editItem(name_item)            # 用户点进单元格开始改名
    _pump_until(qapp, lambda: True, timeout=0.05)
    assert win.speaker_table.state() == QAbstractItemView.State.EditingState

    win._refresh_speakers()                          # 模拟 5 秒定时器
    assert win.speaker_table.item(0, 1) is name_item, "正在编辑时被重设：用户输入会被清空"


def test_refresh_speakers_rebuilds_rows_when_not_editing(win, qapp):
    """非编辑态照常刷新（跳过逻辑只针对「用户正在打字」那一刻）。"""
    rec = _RenameRecorder()
    rec.store.register_label("SPEAKER_00")
    win.recorder = rec
    win.speaker_table.blockSignals(True)
    win.speaker_table.setRowCount(1)
    win.speaker_table.setItem(0, 0, QTableWidgetItem("旧标签"))
    old_item = QTableWidgetItem("旧名字")
    win.speaker_table.setItem(0, 1, old_item)
    win.speaker_table.blockSignals(False)
    assert win.speaker_table.state() != QAbstractItemView.State.EditingState

    win._refresh_speakers()                          # 用 store 里的真相重填
    assert win.speaker_table.item(0, 0).text() == "SPEAKER_00"
    assert win.speaker_table.item(0, 1).text() == ""             # store 里还没名字
    assert win.speaker_table.item(0, 1) is not old_item


# ---------------------------------------------------------------------------
# d. 声纹库：会后认「熟人」+ 改名即学习（全部 fake，不加载模型、不碰磁盘）
# ---------------------------------------------------------------------------

VEC = np.array([1.0, 0.0], dtype=np.float32)


class _FakeVoiceprints:
    """VoiceprintLibrary 替身：记录调用与参数，绝不读写磁盘。"""

    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.thresholds: list[float] = []
        self.added: list[tuple[str, object]] = []
        self.penalized: list[tuple[str, object]] = []
        self.saves = 0

    def match(self, vector, threshold=0.75):
        self.thresholds.append(threshold)
        return list(self.hits)

    def add(self, name, vector):
        self.added.append((name, vector))
        return True

    def penalize(self, name, vector, tol=0.15):
        self.penalized.append((name, vector))
        return True

    def save(self):
        self.saves += 1
        return True


def _fake_embeddings(monkeypatch, embeddings, calls=None):
    """把真模型整段换掉：compute_speaker_embeddings 直接返回给定向量。"""
    calls = calls if calls is not None else []

    def fake(audio, sample_rate, spans, embed_fn=None, token=None):
        calls.append((len(audio), sample_rate, list(spans)))
        return dict(embeddings)

    monkeypatch.setattr(appmod, "compute_speaker_embeddings", fake)
    return calls


def _questions(recorded):
    return [d for d in recorded if d[0] == "question"]


def test_offline_diarize_suggests_known_speaker_and_applies(
        win, qapp, dialogs, monkeypatch, tmp_path, ):
    """会后认出熟人 → 弹框问 → 选「是」：store 里改名、向量入库、产物照旧重导出。"""
    recorded, answer = dialogs
    exports, export_threads, calls, seen = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    _fake_embeddings(monkeypatch, {"SPEAKER_00": VEC})
    library = _FakeVoiceprints(hits=[("张总", 0.83)])
    win._voiceprints = library
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)
    win.bridge.speaker_suggestions.connect(seen.append)

    win._stop()
    assert _pump_until(
        qapp, lambda: recorder.store.display_name("SPEAKER_00") == "张总", timeout=15.0), \
        f"没有自动命名：seen={seen} 表格={win.speaker_table.rowCount()}"

    # ① 信号契约：{label: (姓名, 分数)}，只发给主线程弹框用
    assert seen == [{"SPEAKER_00": ("张总", 0.83)}]
    # ② 弹框问过用户，且写明「谁 ≈ 谁（多少分）」
    queries = _questions(recorded)
    assert len(queries) == 1
    assert "≈" in queries[0][2] and "张总" in queries[0][2] and "0.83" in queries[0][2]
    # ③ store 落名字（导出产物与字幕都以它为准）+ 表格立刻可见
    assert recorder.store.display_name("SPEAKER_00") == "张总"
    assert win.speaker_table.item(0, 1).text() == "张总"
    # ④ 本场向量入库 + 落盘；阈值来自 settings
    assert library.added == [("张总", VEC)]
    assert library.saves >= 1
    assert library.thresholds == [0.75]
    # ⑤ 认人不影响产物：初版 + 标注版各导出一次
    assert len(exports) == 2


def test_offline_diarize_suggestion_declined_keeps_labels(
        win, qapp, dialogs, monkeypatch, tmp_path):
    """选「否」：标签保持原样、不入库；但向量留着（下次改名时学，匹配也不丢）。"""
    recorded, answer = dialogs
    exports, export_threads, calls, seen = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    _fake_embeddings(monkeypatch, {"SPEAKER_00": VEC})
    library = _FakeVoiceprints(hits=[("张总", 0.83)])
    win._voiceprints = library
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")
    answer["ret"] = appmod.QMessageBox.StandardButton.No

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)
    win.bridge.speaker_suggestions.connect(seen.append)

    win._stop()
    assert _pump_until(qapp, lambda: bool(_questions(recorded)) and bool(seen), timeout=15.0)
    assert _pump_until(qapp, lambda: win.start_btn.isEnabled())

    assert recorder.store.display_name("SPEAKER_00") == "SPEAKER_00"   # 名字没被改
    assert library.added == []                                         # 没名字就不入库
    # 向量留着：之后用户在表格里改名时 `_learn_rename` 会拿它当正样本
    assert "SPEAKER_00" in win._pending_vectors
    assert win._suggestion_names == {"SPEAKER_00": "张总"}             # 供「改错即纠正」用


def test_offline_diarize_without_embeddings_skips_recognition(
        win, qapp, dialogs, monkeypatch, tmp_path):
    """声纹算不出来（没 token / 没音频 / 模型挂）：不弹框、不发建议，产物照常更新。"""
    recorded, _answer = dialogs
    exports, export_threads, calls, seen = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    _fake_embeddings(monkeypatch, {})
    library = _FakeVoiceprints(hits=[("张总", 0.83)])
    win._voiceprints = library
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)
    win.bridge.speaker_suggestions.connect(seen.append)

    win._stop()
    assert _pump_until(qapp, lambda: len(exports) == 2 and win.start_btn.isEnabled())
    deadline = time.time() + 0.3
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert seen == [] and _questions(recorded) == []
    assert library.added == [] and library.thresholds == []


def test_voiceprint_disabled_skips_whole_path(win, qapp, dialogs, monkeypatch, tmp_path):
    """.env 里 VOICEPRINT_ENABLED=false：连向量都不算（省时间），一切照旧。"""
    exports, export_threads, calls, seen = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    emb = _fake_embeddings(monkeypatch, {"SPEAKER_00": VEC})
    library = _FakeVoiceprints(hits=[("张总", 0.83)])
    win._voiceprints = library
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken",
                                       voiceprint_enabled=False)

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)
    win.bridge.speaker_suggestions.connect(seen.append)

    win._stop()
    assert _pump_until(qapp, lambda: len(exports) == 2 and win.start_btn.isEnabled())
    assert emb == []            # 开关关掉就别白算声纹
    assert seen == [] and library.thresholds == []


def _rename_setup(win, library, label="SPEAKER_00", suggested="", vector=VEC):
    """搭一个「本场已有声纹向量」的改名场景（recorder + 表格数据）。"""
    win._voiceprints = library
    win.subtitle = _RenameSubtitle()
    rec = _RenameRecorder()
    rec.store.register_label(label)
    win.recorder = rec
    win._pending_vectors = {label: vector} if vector is not None else {}
    win._suggestion_names = {label: suggested} if suggested else {}
    return rec


def test_rename_learns_vector_and_penalizes_wrong_auto_match(win, qapp):
    """改名 = 正样本入库；与自动匹配建议不同 → 剔除那条误归属样本（仅一次）。"""
    library = _FakeVoiceprints()
    rec = _rename_setup(win, library, suggested="张总")

    _rename_in_table(win, "SPEAKER_00", "李四")

    assert rec.renames == [("SPEAKER_00", "李四")]      # A3 广播照旧
    assert library.penalized == [("张总", VEC)], "认错人了却没剔除误样本"
    assert library.added == [("李四", VEC)], "新名字的样本没入库"
    assert library.saves == 1, "学完没落盘（重启就白学）"
    assert win._suggestion_names == {}, "建议已处理完，不该留着再纠正一次"


def test_rename_matching_suggestion_does_not_penalize(win, qapp):
    """用户把标签改成「就是建议的那个名字」：只入库，不惩罚（匹配是对的）。"""
    library = _FakeVoiceprints()
    _rename_setup(win, library, suggested="张总")

    _rename_in_table(win, "SPEAKER_00", "张总")

    assert library.penalized == []
    assert library.added == [("张总", VEC)]
    assert library.saves == 1


def test_rename_without_pending_vector_touches_no_library(win, qapp):
    """本场没算出这个标签的向量（没音频 / 无 token）：改名只是改名，库不动。"""
    library = _FakeVoiceprints()
    _rename_setup(win, library, suggested="张总", vector=None)

    _rename_in_table(win, "SPEAKER_00", "李四")

    assert library.added == [] and library.penalized == []
    assert library.saves == 0


def test_rename_twice_relearns_and_drops_previous_name(win, qapp):
    """标签改两次（先李四、后王五 = 用户自己写错了）：新名字学一份，旧名字那条剔掉。"""
    library = _FakeVoiceprints()
    _rename_setup(win, library, suggested="")

    _rename_in_table(win, "SPEAKER_00", "李四")
    assert library.added == [("李四", VEC)]
    _rename_in_table(win, "SPEAKER_00", "王五")

    assert library.added == [("李四", VEC), ("王五", VEC)]
    assert library.penalized == [("李四", VEC)], "旧名字的误样本没剔除，下次还会认成李四"


def test_adopt_vector_does_not_duplicate_same_name(win, qapp):
    """同场同一标签 → 同一个名字只入库一次（弹框自动命名与手动改名都会走这条）。"""
    library = _FakeVoiceprints()
    _rename_setup(win, library)

    assert win._adopt_vector("SPEAKER_00", "李四") is True
    assert win._adopt_vector("SPEAKER_00", "李四") is False
    assert library.added == [("李四", VEC)]
    assert library.penalized == []


def test_correcting_auto_applied_name_relearns(win, qapp, dialogs):
    """自动命名后用户发现认错了：剔除「张总」那条，并把同一条向量记到「李四」名下。"""
    recorded, answer = dialogs
    answer["ret"] = appmod.QMessageBox.StandardButton.Yes
    library = _FakeVoiceprints(hits=[("张总", 0.83)])
    rec = _rename_setup(win, library, vector=VEC)
    win._pending_vectors = {"SPEAKER_00": VEC}

    win._on_speaker_suggestions({"SPEAKER_00": ("张总", 0.83)})
    assert rec.store.display_name("SPEAKER_00") == "张总"
    assert library.added == [("张总", VEC)]

    _rename_in_table(win, "SPEAKER_00", "李四")         # 用户发现认错了

    assert library.penalized == [("张总", VEC)], "误归属样本没被剔除"
    assert library.added[-1] == ("李四", VEC), "同一条向量没记到正确的人名下"
    assert rec.store.display_name("SPEAKER_00") == "李四"
    assert len(_questions(recorded)) == 1


def test_voiceprint_embedding_has_hard_deadline(win, qapp, dialogs, monkeypatch, tmp_path):
    """模型下载 / 网络卡住：按超时放弃，产物照常导出（声纹绝不能拖住交付）。"""
    exports, export_threads, calls, seen = [], [], {}, []
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)
    monkeypatch.setattr(appmod, "VOICEPRINT_TIMEOUT", 0.05)

    def hang(*_a, **_k):
        time.sleep(2.0)                 # 比超时更长
        return {"SPEAKER_00": VEC}

    monkeypatch.setattr(appmod, "compute_speaker_embeddings", hang)
    library = _FakeVoiceprints(hits=[("张总", 0.83)])
    win._voiceprints = library
    win.settings = dataclasses.replace(win.settings, hf_token="hf_faketoken")

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)
    win.bridge.speaker_suggestions.connect(seen.append)

    started = time.time()
    win._stop()
    assert _pump_until(qapp, lambda: len(exports) == 2 and win.start_btn.isEnabled(),
                       timeout=15.0)
    assert time.time() - started < 5.0, "声纹卡住时把产物一起拖住了"
    assert seen == [] and library.thresholds == []
    assert recorder.store.display_name("SPEAKER_00") == "SPEAKER_00"


def test_voiceprint_token_comes_from_settings(win, qapp, dialogs, monkeypatch, tmp_path):
    """HF token 由 settings 传入（密钥在 Windows 凭据管理器里时环境变量是空的）。"""
    exports, export_threads, calls, captured = [], [], {}, {}
    monkeypatch.setattr(appmod, "export_all", _export_stub(exports, export_threads))
    _fake_diarize_tools(monkeypatch, calls)

    def fake(audio, sample_rate, spans, embed_fn=None, token=None):
        captured["token"] = token
        return {}

    monkeypatch.setattr(appmod, "compute_speaker_embeddings", fake)
    win._voiceprints = _FakeVoiceprints()
    win.settings = dataclasses.replace(win.settings, hf_token="hf_from_keyring")

    recorder = _DiarizableRecorder(_write_wav(tmp_path))
    _prepare_recording(win, recorder)

    win._stop()
    assert _pump_until(qapp, lambda: bool(captured) and len(exports) == 2, timeout=15.0)
    assert captured["token"] == "hf_from_keyring"


# ---------------------------------------------------------------------------
# 云端自检提示条（Groq 403 地区封禁 → 提醒开代理）
# ---------------------------------------------------------------------------

def test_probe_groq_states(monkeypatch):
    """成功/401/403/断网 四种结果分别给出正确 state 与可重试标记。

    探测必须走 openai SDK（与真实识别同链路）；urllib 的默认 UA 会被 WAF 403。
    """
    import openai
    from meeting_bot.app import probe_groq

    def _client_raising(exc):
        class _Models:
            @staticmethod
            def list():
                raise exc

        class _Client:
            def __init__(self, **kw):
                self.models = _Models()

        return _Client

    def _resp(code):
        return type("R", (), {"status_code": code, "request": None, "headers": {}})()

    # 1) 成功
    class _OkModels:
        @staticmethod
        def list():
            return []

    class _OkClient:
        def __init__(self, **kw):
            self.models = _OkModels()

    monkeypatch.setattr("openai.OpenAI", _OkClient)
    assert probe_groq("k")[0] == "ok"

    # 2) key 无效 → 不可重试
    monkeypatch.setattr("openai.OpenAI", _client_raising(
        openai.AuthenticationError("bad", response=_resp(401), body=None)))
    state, text, retry = probe_groq("k")
    assert state == "error" and retry is False and "无效" in text

    # 3) 403 → 可重试
    monkeypatch.setattr("openai.OpenAI", _client_raising(
        openai.PermissionDeniedError("no", response=_resp(403), body=None)))
    state, text, retry = probe_groq("k")
    assert state == "warn" and retry is True and "403" in text

    # 4) 断网/超时 → 可重试
    monkeypatch.setattr("openai.OpenAI", _client_raising(TimeoutError("t")))
    state, text, retry = probe_groq("k")
    assert state == "warn" and retry is True


def test_cloud_banner_hidden_in_local_mode(win):
    """local 模式下不提示云端问题。"""
    win.settings = type(win.settings)(**{**win.settings.__dict__, "asr_provider": "local"}) \
        if hasattr(win.settings, "__dict__") else win.settings
    state, text, _ = win._probe_cloud()
    if getattr(win.settings, "asr_provider", "local") == "local":
        assert (state, text) == ("hidden", "")


def test_cloud_banner_shows_retry_hint(win):
    """403 之类的可重试问题：显示提示条 + 重新检测按钮。"""
    win._on_cloud_status("warn", "云端识别不可用：请打开代理", True)
    assert not win.cloud_banner.isHidden()      # 父窗口未 show 时 isVisible() 恒 False
    assert not win.recheck_btn.isHidden()
    assert "代理" in win.cloud_banner.text()

    win._on_cloud_status("hidden", "", False)
    assert win.cloud_banner.isHidden()
    assert win.recheck_btn.isHidden()


def test_recheck_button_tall_enough_for_text(win, qapp):
    """回归：提示条按钮曾被写死 30px 高，QSS padding 要 40px → “重新检测”被裁掉。"""
    win._on_cloud_status("warn", "云端识别不可用：请打开代理", True)
    qapp.processEvents()
    btn = win.recheck_btn
    assert btn.text() == "重新检测"
    assert btn.height() >= btn.sizeHint().height(), "按钮高度不足，文字会被裁切"
    assert btn.width() >= btn.fontMetrics().horizontalAdvance("重新检测"), "按钮宽度不足"
