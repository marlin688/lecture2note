# lecture2note

基于多模型 AI 的课堂视频/录音 → 结构化学习笔记 & 字幕翻译工具。

支持四大核心功能：
1. **字幕工作流** — 从 YouTube 视频自动转录英文字幕（Whisper GPU 加速），翻译为中文/中英双语字幕，同时保存封面和音视频下载地址
2. **笔记工作流** — 将课堂转录文本（或 YouTube Transcript）通过 LLM 整理为排版精美的 Markdown 学习笔记
3. **摘要工作流** — 基于字幕内容，LLM 自动生成视频摘要 Markdown
4. **批量工作流** — 一次性批量处理多个视频：字幕 + 摘要 + 下载最高画质视频，支持断点续传

## 功能特性

### 字幕模式（`--subtitle`）

- **Whisper 本地转录** — 使用 mlx-whisper（Apple Silicon GPU 加速），断句准确、自带标点
- **字幕翻译** — 支持中文（`zh`）、中英双语（`bilingual`）、纯英文（`en`）三种输出
- **4 路并发翻译** — 长字幕自动分段，4 路并发调用 LLM 翻译，速度快
- **质量回退** — Whisper 转录质量不佳时自动回退到 YouTube 字幕
- **YouTube 字幕保留** — 同时保存 YouTube 原始英文字幕供对照
- **视频封面下载** — 自动保存视频最高分辨率封面图
- **视频下载地址** — 列出所有分辨率的下载格式，方便配合字幕离线观看

### 摘要模式（`--summary`）

- **一键摘要** — 基于已有字幕自动生成结构化视频摘要
- **智能复用** — 优先使用已有的中文/双语/英文字幕，无需重复转录
- **Markdown 输出** — 摘要保存为 `summary.md`，方便阅读和分享

### 批量模式（`--batch`）

- **一键批量** — 从 URL 列表文件批量处理：字幕翻译 + 摘要生成 + 视频下载
- **断点续传** — 中断后重新运行自动跳过已完成的步骤，每个步骤（字幕/摘要/视频）独立检查
- **分辨率控制** — `--max-res` 限制下载视频的最大分辨率（720p/1080p/2k/4k），默认 1080p
- **容错处理** — 单个视频失败不中断整个批次，最后汇总成功/失败列表

### 笔记模式（`-i` / `-u`）

- **ASR 纠错** — 自动修正语音识别的同音替换、术语误识别等错误
- **智能降噪** — 去除口头禅、重复、课堂管理等无关内容
- **逻辑重组** — 按知识体系而非时间顺序组织笔记结构
- **重点标注** — 识别并标记教师反复强调的内容
- **术语整理** — 提取专业术语并标注英文对照与释义
- **复习题生成** — 自动生成 3-5 道覆盖核心概念的复习题
- **长文本分片** — 超长转录文本自动分片处理并智能合并去重
- **多模型支持** — 支持 Claude、Gemini、GPT 系列模型
- **自动续写** — 输出被截断时自动续写，确保完整输出

## 快速开始

### 环境要求

- Python 3.12+
- Apple Silicon Mac（M1/M2/M3/M4）推荐，mlx-whisper 可 GPU 加速
- ffmpeg（`brew install ffmpeg`，用于音频提取）
- Node.js（yt-dlp 需要 nodejs 运行时解析 YouTube）
- 至少一个 AI 模型的 API Key：
  - [Anthropic API Key](https://console.anthropic.com/)（Claude 模型，字幕翻译默认使用）
  - [Google AI API Key](https://aistudio.google.com/)（Gemini 模型）
  - [OpenAI API Key](https://platform.openai.com/)（GPT 模型）

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd lecture2note

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入你需要的 API Key（至少配置一个）
```

### 使用

#### 字幕模式（推荐）

```bash
# Whisper 转录 + 中文翻译（最常用）
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle zh

# 中英双语字幕
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle bilingual

# 仅 Whisper 转录英文字幕
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle en

# 使用 Whisper large 模型（更准确，更慢）
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle zh --whisper-model large

# 不使用 Whisper，回退到 YouTube 字幕
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle zh --no-whisper
```

#### 摘要模式

```bash
# 生成视频摘要（需先有字幕）
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --summary

# 字幕 + 摘要一起生成
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle zh --summary
```

#### 笔记模式

```bash
# 从本地转录文本生成笔记
python main.py -i 转录文本.txt

# 指定课程学科和模型
python main.py -i 转录文本.txt -s "深度学习" -m claude-opus-4-6

# 从 YouTube 视频提取字幕并生成笔记
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" -s "计算机科学"

# 仅提取 YouTube 字幕，不生成笔记
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-only

# 使用 Gemini / GPT 模型
python main.py -i 转录文本.txt -m gemini-3-pro-preview
python main.py -i 转录文本.txt -m gpt-4o
```

#### 批量模式

```bash
# 准备 URL 文件（每行一个，# 开头为注释）
cat > urls.txt << 'EOF'
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
# 这行会被跳过
https://www.youtube.com/watch?v=VIDEO_ID_3
EOF

# 批量处理：字幕 + 视频下载（默认 1080p）
python main.py --batch urls.txt

# 批量处理：字幕 + 摘要 + 视频下载
python main.py --batch urls.txt --summary

# 限制最高 720p，省流量
python main.py --batch urls.txt --summary --max-res 720p

# 指定 2k 分辨率 + 双语字幕
python main.py --batch urls.txt --subtitle bilingual --max-res 2k

# 中断后重新运行，自动跳过已完成的视频
python main.py --batch urls.txt --summary
```

#### 其他

```bash
# 列出视频所有分辨率的下载格式
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --list-formats
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入转录文本文件路径（与 `-u` 二选一） | — |
| `-u, --url` | YouTube 视频 URL（与 `-i` 二选一） | — |
| `-o, --output` | 输出 Markdown 文件路径 | 自动生成 |
| `-s, --subject` | 课程学科 | — |
| `-m, --model` | 模型名称（Claude / Gemini / GPT） | `claude-sonnet-4-5-20250929` |
| `--save-json` | 同时保存 JSON 输出 | `false` |
| `--transcript-only` | 仅提取 YouTube 字幕，不生成笔记 | `false` |
| `--subtitle` | 生成字幕文件（`zh` / `bilingual` / `en`） | — |
| `--whisper-model` | Whisper 模型（`tiny`/`base`/`small`/`medium`/`large`） | `medium` |
| `--no-whisper` | 不使用 Whisper，回退到 YouTube 字幕 | `false` |
| `--summary` | 生成视频摘要 Markdown | `false` |
| `--batch` | 批量处理：指定包含多个 YouTube URL 的文本文件 | — |
| `--max-res` | 下载视频的最大分辨率（`720p`/`1080p`/`2k`/`4k`） | `1080p` |
| `--list-formats` | 列出视频可用的下载格式和地址 | `false` |

## 项目结构

```
lecture2note/
├── main.py                  # CLI 入口
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
├── src/
│   ├── noter.py             # 核心：分片、调用 LLM API、JSON 解析、合并
│   ├── assembler.py         # JSON → Markdown 格式组装
│   ├── transcriber.py       # YouTube Transcript API 提取
│   ├── whisper_transcriber.py # mlx-whisper 转录 + 音频/封面下载
│   ├── subtitle.py          # 字幕翻译（分段、并发、SRT 生成）
│   └── downloader.py        # YouTube 视频格式信息获取 + 视频下载
├── prompts/
│   ├── note_system.md       # 笔记生成系统提示词
│   ├── merge_system.md      # 字幕片段合并提示词
│   ├── translate_zh_system.md   # 中文字幕翻译提示词
│   ├── translate_system.md      # 中英双语字幕翻译提示词
│   └── summary_system.md       # 视频摘要生成提示词
└── output/
    ├── transcript/              # YouTube 字幕提取结果
    ├── subtitle/{video_id}/     # 字幕输出（按视频 ID 分目录）
    │   ├── audio.m4a            # 下载的音频
    │   ├── cover.jpg            # 视频封面
    │   ├── subtitle_en.srt      # Whisper 英文字幕
    │   ├── subtitle_en_youtube.srt  # YouTube 原始英文字幕
    │   ├── subtitle_zh.srt      # 中文翻译字幕
    │   ├── subtitle_bilingual.srt   # 中英双语字幕
    │   ├── summary.md             # 视频摘要
    │   └── *.mp4                  # 下载的视频文件
    └── *.md                     # 生成的笔记
```

## 工作原理

### 字幕工作流

```
YouTube URL
    │
    ├─► yt-dlp 下载音频 (m4a) + 封面图
    │
    ├─► mlx-whisper GPU 转录 → subtitle_en.srt
    │   （质量不佳时自动回退 YouTube 字幕）
    │
    ├─► 同时保存 YouTube 原始字幕 → subtitle_en_youtube.srt
    │
    ├─► LLM 4 路并发翻译 → subtitle_zh.srt / subtitle_bilingual.srt
    │
    ├─► 列出视频下载格式
    │
    ├─► [可选] LLM 生成视频摘要 → summary.md
    │
    └─► [可选] 下载视频 (--batch 模式自动下载)
```

### 批量工作流

```
URL 列表文件 (urls.txt)
    │
    ├─► 逐个视频处理，每个执行：
    │   ├─► 字幕转录 + 翻译
    │   ├─► 摘要生成 (--summary)
    │   └─► 视频下载 (受 --max-res 限制)
    │
    ├─► 断点续传：检查已有产出，跳过已完成步骤
    │
    └─► 汇总报告：成功/跳过/失败统计
```

### 笔记工作流

```
YouTube URL / 转录文本 (.txt)
    │
    ├─► [YouTube] 提取 Transcript
    │
    ├─► 文本分片（超过 80,000 字符时按段落边界切分）
    │
    ├─► LLM API 生成结构化 JSON
    │
    ├─► 多片结果去重合并
    │
    └─► Markdown 组装输出（目录、折叠面板、术语表、复习题）
```

## 技术栈

- [mlx-whisper](https://github.com/ml-explore/mlx-examples) — Apple Silicon GPU 加速的 Whisper 语音转录
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube 音频下载与视频格式获取
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API
- [Google GenAI SDK](https://github.com/googleapis/python-genai) — Gemini API
- [OpenAI Python SDK](https://github.com/openai/openai-python) — GPT API 及 OpenAI 兼容接口
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — YouTube 字幕提取（回退方案）
- [Click](https://click.palletsprojects.com/) — CLI 框架
- [python-dotenv](https://github.com/theskumar/python-dotenv) — 环境变量管理

## License

MIT
