---
name: video-process
description: 一键处理 YouTube 视频：生成中文字幕、视频摘要、AI 封面，并下载最高画质视频，最后把所有产物的绝对路径回贴到对话。当用户提供 YouTube URL 并表达"处理这个视频""下载字幕""生成摘要""一键处理"等意图时（尤其是 Hermes 远程触发进来的对话），自动调用此 skill。
---

# video-process

lecture2note 的一键全流程 skill：URL → 中文字幕 + 摘要 + AI 封面 + 高清视频 → 返回所有路径。

## 何时使用

用户给一个 YouTube URL，并想要字幕 / 摘要 / 封面 / 视频下载中的任意组合——就用这个 skill，默认全做。

不适用场景：
- 只想翻译一段已有 SRT → 用 `/subtitle-translate`
- 只想从本地音频生成笔记 → 用 `/note-generate`
- 一次批量处理多条 URL → 用 `batch-agent`

## 工作目录

**必须** `cd /Users/marlin/学习/agent-dev-project/lecture2note` 再执行，否则 `output/` 相对路径会错。

## 执行命令（后台模式）

整条流水线通常 5–20 分钟，**必须后台执行**，否则 Bash 会超时。

从用户消息解析出 YouTube URL 后：

```bash
cd /Users/marlin/学习/agent-dev-project/lecture2note && \
  python main.py -u '<URL>' --subtitle zh --summary --cover --download
```

**调用 Bash 工具时必须设置 `run_in_background: true`**，拿到 bash_id 后用 Monitor 工具等待完成事件——**不要 sleep / 不要轮询**。

命令依次完成：

1. yt-dlp 下载音频和 YouTube 原始封面
2. mlx-whisper（Apple Silicon GPU）转录英文字幕
3. LLM 4 路并发翻译为中文字幕
4. LLM 参考中文反向校对英文字幕（修正 ASR 错误）
5. LLM 生成 `summary.md`
6. Gemini 生成 3 张 B 站风格 AI 封面
7. yt-dlp 下载最高画质视频（纯视频自动合并音轨）

URL 有特殊字符必须用单引号包裹。

## 产物位置

所有文件落在 `output/subtitle/<video_id>/`：

```
output/subtitle/<video_id>/
├── subtitle_zh.srt           ← 中文字幕（主产物）
├── subtitle_en.srt           英文字幕（Whisper + 校对）
├── subtitle_en_youtube.srt   YouTube 原始英文字幕
├── summary.md                ← 视频摘要（主产物）
├── cover.jpg                 YouTube 原始封面
├── cover_1.jpg / cover_2.jpg / cover_3.jpg   ← AI 封面（主产物）
├── audio.m4a                 音频
└── *.mp4                     ← 视频文件（主产物）
```

命令跑完后用 `ls` 或 Glob 列实际文件名（mp4 文件名是动态的）。

## 执行步骤

1. **解析 URL**：从消息里提取 `https://www.youtube.com/watch?v=...` 或 `https://youtu.be/...`
2. **发一条"已开始"的确认消息**，让用户知道任务已接收（Hermes 会立刻推送这条，用户可以去做别的事）。内容示例：
   > 🚀 已开始处理 `<video_id>`，预计 5–20 分钟。完成后会把所有产物路径发给你。
3. **后台启动命令**：Bash 工具必须用 `run_in_background: true`
4. **用 Monitor 工具等待完成事件**（不要 sleep、不要轮询）
5. **失败诊断**（看命令输出）：
   - yt-dlp / 网络错误 → 提示更新 yt-dlp 或检查 cookies
   - API key 错误 → 检查 `.env` 对应 key 是否存在
   - Whisper OOM → 加 `--whisper-model small` 重跑
6. **列出产物** → `ls output/subtitle/<video_id>/`
7. **发送最终通知消息**（见下节"回贴格式"）——这条消息的到达本身就是对用户的"完成通知"，Hermes 客户端会推送到用户设备

## 回贴格式

命令成功后，按这个格式把**绝对路径**整理给用户——这是 Hermes 场景的最终交付物，用户就靠这条消息拿到文件位置：

```
✅ 处理完成：<video_id>

📺 视频
/Users/marlin/学习/agent-dev-project/lecture2note/output/subtitle/<id>/<文件名>.mp4

📝 中文字幕
/Users/marlin/学习/agent-dev-project/lecture2note/output/subtitle/<id>/subtitle_zh.srt

📄 摘要
/Users/marlin/学习/agent-dev-project/lecture2note/output/subtitle/<id>/summary.md

🖼️ 封面
/Users/marlin/学习/agent-dev-project/lecture2note/output/subtitle/<id>/cover_1.jpg
/Users/marlin/学习/agent-dev-project/lecture2note/output/subtitle/<id>/cover_2.jpg
/Users/marlin/学习/agent-dev-project/lecture2note/output/subtitle/<id>/cover_3.jpg
```

**一律用绝对路径**，方便用户 scp / 浏览器 file:// / Finder 直接打开。

## 禁止项

- 不要改 `--subtitle zh`（用户明确要中文）
- 不要省 `--download`（用户明确要视频本体）
- 不要把 srt / summary.md 的全文贴回对话，用户只要路径
- 不要默认切 `--whisper-model large`，medium 已够用且快得多
- 不要在非 lecture2note 目录下跑 main.py
