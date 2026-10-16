# 会议翻译助手（Teams Meeting Bot）

本地旁听 Teams 会议：实时识别多语种发言 → 翻译成中文 → 置顶字幕浮窗；
会议结束后自动生成双语逐字稿（Markdown）与中文会议纪要（Word）。

## 首次安装

1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. 复制 `.env.example` 为 `.env`：
   - `OPENCODE_API_KEY`：从 `C:\Users\<你>\.pi\agent\models.json` 中 provider
     `cc-switch-open-code-go` 的 `apiKey` 复制
   - `WHISPER_MODEL`：默认 `small`（本机快于实时）；追求精度可改 `medium`
     （慢于实时约 1.5 倍，字幕延迟变大）
   - `HF_TOKEN`：**可选**，仅说话人区分需要。到 https://huggingface.co/settings/tokens
     创建 token，并打开 https://huggingface.co/pyannote/speaker-diarization-3.1
     点击 "Agree and access repository" 接受使用协议；留空则关闭说话人区分
3. `python install_shortcut.py` 创建桌面快捷方式
4. 首次运行会自动下载语音模型（small 约 500MB，请耐心等待）

## 使用

1. 开会前：双击桌面"会议翻译助手"；确认 Teams 扬声器输出走系统默认设备
2. 填写会议名 → 点"开始记录"
3. 字幕浮窗可拖拽，滚轮调字号；在说话人表格里把声纹标签改成真实名字
4. 会议结束 → 点"结束并生成纪要"，产物在 `会议记录/日期_时间_会议名/` 下

## 已知限制

- **字幕延迟**：本机 CPU 上短句识别的固定开销约 12~15 秒/句，字幕延迟
  约 15~30 秒（取决于机器负载）；对连续不停讲话的会议，识别可能跟不上实时
  （状态栏会提示丢弃块数）
- 说话人区分基于声纹聚类，多人抢话时准确率下降；同一人超过 5 分钟不发言
  再开口可能被标为新标签
- 只捕获系统播放的声音；你的麦克风声音不会进入记录
- 翻译文本发送到 OpenCode Go 云端（识别与音频全程本地）
