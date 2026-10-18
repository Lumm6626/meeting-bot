# 会议翻译助手（Meeting Bot）

本地运行的会议实时翻译工具：把电脑能"听到"的一切声音 —— 线上会议里对方的话、你自己说话、现场面对面对话 ——
**实时识别并翻译成中文字幕**，会后自动生成**双语逐字稿**和**中文会议纪要**。

> 仅支持 **Windows 10/11**（使用 WASAPI 系统声音回环采集，无需虚拟声卡）。

## 功能

- **自动双路采集**：同时捕获「系统播放的声音」+「麦克风」，线上会议/现场对话不用切换模式
- **多语种自动检测**：英语、西班牙语等自动识别，统一翻译成中文；中文发言直接显示原文
- **低延迟字幕**：云端识别（Groq whisper-large-v3-turbo）约 3~5 秒出字幕；本地识别模式 30~55 秒（零成本、不出本机）
- **Teams 转录式字幕面板**：按说话人回合分块、原文+译文段落化显示；窗口可拉伸，字号随窗口自动缩放（滚轮调节）
- **两阶段显示**：识别完先出原文，翻译完成后原地替换为中文（流式逐字出现）
- **带上下文的翻译**：自动参考前几句内容，统一术语/人名，修复口音导致的识别错字
- **会后产物**：`会议记录/<会议名>/逐字稿.md` + `纪要.docx`（讨论要点/结论/行动项/待跟进）
- **可选说话人区分**：接入 HuggingFace 的 pyannote 模型后可标注"谁在说话"（本机 CPU 标注较慢，标签会有分钟级延迟）

## 安装（一键）

1. 安装 [Python 3.12](https://www.python.org/downloads/)（安装时勾选 **Add python.exe to PATH**）
2. 下载本仓库（Code → Download ZIP）并解压
3. 双击 **`install.bat`**，按提示填写 API Key（详见下一节）
4. 完成后桌面会出现「会议翻译助手」快捷方式，双击即用（只有界面，无命令行窗口）

> 首次使用会下载语音模型（约 500MB）。国内网络已自动配置 HuggingFace 镜像。

## 需要准备的 API Key

| Key | 是否必须 | 获取方式 |
|---|---|---|
| 翻译服务 API Key | ✅ 必须 | 任选一个 OpenAI 兼容服务：<br>· [DeepSeek 官方](https://platform.deepseek.com/)（推荐，便宜稳定）<br>· [OpenCode Go](https://opencode.ai/)（有 pi/opencode 账号可直接用）<br>· 通义/月之暗面/本地 Ollama 等自定义接口 |
| Groq API Key | 可选（推荐） | [console.groq.com](https://console.groq.com) 免费注册 → API Keys → 创建（`gsk_` 开头）。用于云端快速识别；不填则退回本地识别（慢但免费） |
| HuggingFace Token | 可选 | 仅"说话人区分"需要：注册 [huggingface.co](https://huggingface.co/settings/tokens) 创建 token，并在三个模型页点 Agree：[speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)、[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)、[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) |

配置向导也可随时手动重跑：删除项目根目录的 `.env`，再执行 `install.bat`。

## 使用

1. 双击桌面「会议翻译助手」→ 窗口 1 秒内出现（会议名默认按「年-月-日 时:分」自动生成，可手动改）
2. 点 **开始记录** → 字幕浮窗出现
   - 字幕面板：按说话人回合分块，译文整段显示、原文灰色小字跟随
   - 拖拽窗口可移动；拖边缘可缩放；滚轮调字号；向上翻看历史时暂停自动跟随，滚回底部恢复
3. 会后点 **结束并生成纪要** → 在 `会议记录/` 下生成逐字稿与纪要
4. 关闭主窗口 = 退出程序（录制中关闭会确认：先结束录制并保存再退出）

## 常见问题

**Q：字幕要等多久？**
云端识别（Groq）下：说话结束约 2 秒先出原文，4~6 秒出中文译文。本地识别（`ASR_PROVIDER=local`）按 30 秒一批，延迟 30~55 秒。

**Q：翻译没反应 / 提示翻译失败？**
多半是翻译服务的 Key 过期或余额不足。换新的 key 后更新 `.env` 里的 `OPENCODE_API_KEY`（或你所用服务的 key）即可，无需重装。

**Q：没有声音进入字幕？**
检查：① Windows 麦克风权限（设置 → 隐私 → 麦克风 → 允许桌面应用）；② 系统默认输入设备是否正确；
③ 想排查采集电平，可运行 `.venv\Scripts\python scripts\diagnose_mic.py`（60 秒录音诊断，逐层打点）。

**Q：说话人显示"发言人?"或标签乱？**
未配置 HuggingFace token 时无法区分说话人，属正常。配置后本机 CPU 标注较慢，标签会有分钟级延迟。

**Q：能不能同时听系统声音和麦克风，会不会重复？**
默认自动双路。笔记本内置麦克风阵列通常带回声消除，会滤掉本机外放的声音，所以不会重复记录；用耳机时系统声音走回环、你的声音走麦克风，也互不影响。

## 隐私说明

- **音频**：仅在使用云端识别（Groq）时上传音频片段用于转写；使用 `ASR_PROVIDER=local` 时音频全程不出本机
- **文字**：外文原文会发送到你配置的翻译服务（用于翻译与生成纪要）
- **本地存储**：所有记录/纪要保存在 `会议记录/` 目录；`.env` 中的密钥只存本机且已被 git 忽略
- 请遵守所在地区的数据合规要求，并在录音前征得参会人同意

## 目录结构

```
meeting_bot/            核心代码
  capture.py            音频采集（扬声器回环 + 麦克风，双路混合）
  vad.py                语音活动检测/切句
  asr.py                本地 faster-whisper / Groq 云端识别
  translator.py         翻译与纪要（OpenAI 兼容接口）
  recorder.py           录制编排（多线程 + 事件）
  subtitle.py           字幕浮窗（回合分块/可拉伸/自动缩放）
  app.py                主窗口与入口
  export.py             逐字稿 md + 纪要 docx
scripts/                诊断与辅助脚本
assets/app.ico          应用图标
tests/                  单元/集成测试（200+）
install.bat             一键安装
setup_env.py            交互式配置向导
```

## 技术栈

Python 3.12 · PyQt6 · faster-whisper · silero-vad · pyannote.audio（可选）· soundcard(WASAPI) ·
OpenAI 兼容 LLM 接口 · python-docx

## 许可

MIT License，见 [LICENSE](LICENSE)。
第三方模型遵循各自许可证（whisper 权重 MIT；pyannote 模型需在 HuggingFace 接受使用条款，不会被本仓库分发）。
