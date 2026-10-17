# 会议翻译助手（Teams Meeting Bot）设计规格

> 日期：2026-10-16 · 状态：已与用户确认

## 目标

参加用户的 Teams 会议（本地旁听），将各发言人的语音实时识别并翻译成中文，以置顶字幕浮窗呈现；会议结束后自动生成完整双语录（Markdown）和中文会议纪要（Word 文档）。桌面快捷方式双击启动。

## 已确认的需求决策

| 决策点 | 结论 |
|---|---|
| 字幕给谁看 | 仅用户本人（本地浮窗，会议无感知，不改变会议流程） |
| 语言场景 | 多语种混合，自动检测语种，统一翻译成中文 |
| 技术路线 | B 混合方案：本地 whisper 识别（免费、数据不出本机）+ OpenCode Go 云大模型翻译 |
| 翻译 API | 复用 pi agent 当前接入的 OpenCode Go（OpenAI 兼容接口），零额外费用 |
| 说话人区分 | 需要（音频声纹聚类） |
| 会后产物 | 纪要 Word（.docx）+ 完整逐字稿 Markdown |
| 启动方式 | 桌面快捷方式，双击打开 |
| 运行机器 | 用户本机：Intel Core Ultra 5 125H（14C18T，无独显）、32GB RAM、Python 3.12、ffmpeg 8.1 |

## 架构

```
┌─────────────┐   WASAPI loopback   ┌──────────────┐
│ Teams 会议声音│ ──────────────────→ │  音频捕获      │ 截取系统正在播放的声音
└─────────────┘   (无需虚拟声卡)      └──────┬───────┘
                                            ↓ 16kHz 单声道流
                                    ┌──────────────┐
                                    │ 语音活动检测   │ silero-vad 切分出完整语句
                                    └──────┬───────┘
                                           ↓ 语句音频
                                    ┌──────────────┐
                                    │ 本地语音识别   │ faster-whisper medium (int8, CPU)
                                    │ + 语种自动检测 │
                                    └──────┬───────┘
                                           ↓ 原文 + 语种 + 时间戳
                     ┌─────────────────────┴─────────────────────┐
                     ↓                                             ↓
            ┌──────────────┐                             ┌──────────────┐
            │ 翻译（异步队列）│                             │ 说话人标注     │
            │ deepseek-v4- │                             │ pyannote 3.1  │
            │ flash        │                             │ 每60s后台聚类  │ （300s 滑动窗口）
            └──────┬───────┘                             └──────┬───────┘
                   ↓                                            ↓
            ┌────────────────────────────────────────────────────┐
            │              TranscriptStore（统一记录缓冲）          │
            │  时间戳 / 说话人 / 语种 / 原文 / 译文                   │
            └───────┬────────────────────────────────┬───────────┘
                    ↓ 实时事件                          ↓ 会议结束
            ┌──────────────┐                  ┌────────────────────┐
            │ 字幕浮窗       │                  │ 导出 + 纪要生成       │
            │ 置顶/半透明/   │                  │ 逐字稿.md           │
            │ 可拖拽/字号可调 │                  │ deepseek-v4-pro    │
            └──────────────┘                  │ → 纪要.docx          │
                                               └────────────────────┘
```

## 模块与职责

| 模块 | 文件 | 职责 | 对外接口 |
|---|---|---|---|
| 配置 | `meeting_bot/config.py` | 读 .env，校验必填项 | `load_settings() -> Settings` |
| 记录缓冲 | `meeting_bot/transcript.py` | 语句存储、说话人命名、Markdown 渲染 | `TranscriptStore`, `Segment` |
| 翻译/纪要 | `meeting_bot/translator.py` | 调 OpenCode Go；实时翻译 + 纪要生成 | `Translator.translate()`, `.minutes()` |
| 语音识别 | `meeting_bot/asr.py` | whisper 转写 + 语种检测 | `ASREngine.transcribe(audio) -> list[Segment]` |
| 说话人 | `meeting_bot/diarization.py` | 声纹聚类 + 标注归属 | `Diarizer.label()`, `assign_speakers()` |
| 音频捕获 | `meeting_bot/capture.py` | 系统声音 loopback / 默认麦克风采集 | `AudioCapture(source="speaker"\|"mic")`（后台线程，产出 16k 单声道块） |
| 语句切分 | `meeting_bot/vad.py` | VAD 边界切完整语句 | `UtteranceBuffer.push(chunk) -> list[np.ndarray]` |
| 编排 | `meeting_bot/recorder.py` | 串起各模块的线程/队列/事件 | `Recorder(start/stop, on_segment, on_status)` |
| 字幕浮窗 | `meeting_bot/subtitle.py` | 置顶浮窗显示 | `SubtitleWindow.show_segment(seg)` |
| 主窗口 | `meeting_bot/app.py` | 开始/结束、说话人改名、状态 | Qt 应用入口 |
| 导出 | `meeting_bot/export.py` | 逐字稿 md + 纪要 docx | `write_transcript()`, `write_minutes()` |
| 启动 | `run_meeting_bot.bat` / `install_shortcut.py` | 一键启动 + 桌面快捷方式 | — |

## 关键设计决策

1. **音频来源**：WASAPI loopback 捕获系统扬声器输出（Teams 语音），无需虚拟声卡、无需会议改动。用户自己的发言不会被捕获（不影响，本人不需要看自己的翻译）。前提：Teams 扬声器输出走系统默认设备（README 说明）。
   1.1 **音频源可切换**：扬声器 loopback（线上会议）/ 默认麦克风（现场对话），UI 下拉选择（`AudioCapture(source="speaker"|"mic")`，默认 speaker）。麦克风模式用于面对面场景，会录到用户本人声音（中文直接显示原文）；两种模式互斥，不开混音。Teams 会议必须用扬声器模式，否则扬声器声音回声进麦克风会被重复记录（README 警告）。设备在点「开始记录」时先探测一次，失败即弹框并放弃启动，避免录到一半才发现设备不可用。
2. **语句切分**：silero-vad（onnx）在 16kHz 流上检测语音边界，按完整语句送识别，避免句子被切断。
3. **说话人**：pyannote 3.1 在后台对最近 300 秒音频每 60 秒跑一次滑动窗口聚类，结果按时间重叠映射到语句；语句先以“发言人?”上屏，标注后更新。标注刷新周期约 1 分钟（首次标注约在录制 1 分钟后出现）。用户在说话人表格里给每个声纹改名，同一次会议内自动沿用。跨会议的声纹记忆（speakers.json）列为后续迭代（YAGNI）。
4. **翻译延迟**：语句结束 → 识别（1~3s）→ 翻译（1~3s），字幕总延迟约 3~8 秒。可接受（本人旁听用途）。
5. **翻译模型**：默认 `deepseek-v4-flash`（发送时尝试关闭 thinking 以降低延迟，接口不支持则自动降级为默认请求）；执行阶段做一次延迟基准测试，必要时切换模型。纪要默认 `deepseek-v4-pro`。
6. **纪要格式**：标题、会议概要、讨论要点、结论、行动项（含责任人/截止时间，按发言内容推断）、待跟进问题。LLM 输出 JSON，程序解析（`json.loads`，禁用 eval）；解析失败则降级为纯文本纪要。
7. **产物目录**：`meeting-bot/会议记录/YYYY-MM-DD_HHMM_<会议名>/逐字稿.md` + `纪要.docx`。

## 错误处理

- 翻译 API 失败：重试 2 次（指数退避）；仍失败则该句字幕显示原文（不翻译），记录继续，不崩溃。
- API key 缺失：启动时弹窗提示，不静默失败。
- HF token 缺失/未接受 pyannote 协议：启动检查，给出逐步指引。
- whisper 模型未下载：首次运行自动下载（~1.5GB），状态栏提示进度。
- 录制期间 API 断连：本地识别和记录不受影响，翻译降级为原文显示，纪要生成时统一补译。

## 安全

- API key 只存 `.env`（已加入 .gitignore），代码中不硬编码；参考值从 pi 的 `~/.pi/agent/models.json` 复制。
- 会议数据全部留在本机（除翻译文本发送到 OpenCode Go 外）；不做任何云端存储。
- 不使用 eval/exec；解析 LLM JSON 用 `json.loads`。

## 非目标（YAGNI）

- 不做云部署、不推字幕给其他参会者、不做日历自动加入
- 不做会议录制文件保存（只存文字记录）
- 不做自研声纹库（直接用 pyannote）

## 成功标准

1. 双击桌面快捷方式 → 打开主窗口 → 点"开始记录"，Teams 会议中的外文发言在 10 秒内以中文上字幕
2. 说话人标注准确率在清晰单人发言场景 ≥ 80%（用户验收为准）
3. 会议结束后自动产出：完整双语录 md + 结构正确的纪要 docx
4. 断网/API 抖动下录制不中断、数据不丢
