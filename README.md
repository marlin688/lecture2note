# lecture2note

基于多模型 AI 的课堂录音转录文本 → 结构化学习笔记工具。

支持 Claude、Gemini、GPT 等多种大模型，将语音识别生成的课堂转录文本自动整理为排版精美、结构清晰的 Markdown 学习笔记。

## 功能特性

- **ASR 纠错** — 自动修正语音识别的同音替换、术语误识别、英文术语音译等错误
- **智能降噪** — 去除口头禅、重复、课堂管理等无关内容，保留有教学价值的比喻和举例
- **逻辑重组** — 按知识体系而非时间顺序组织笔记结构
- **重点标注** — 识别并标记教师反复强调的内容
- **术语整理** — 提取专业术语并标注英文对照与释义
- **复习题生成** — 自动生成 3-5 道覆盖核心概念的复习题（含理解应用类题目）
- **多格式输出** — 同时支持 Markdown（阅读）和 JSON（结构化数据）
- **长文本分片** — 超长转录文本自动分片处理并智能合并去重
- **多模型支持** — 支持 Claude、Gemini、GPT 系列模型，自动选择对应 API
- **自动续写** — 输出被截断时自动续写，确保完整输出
- **YouTube 字幕提取** — 输入 YouTube 视频 URL，自动提取 Transcript 作为笔记来源

## 快速开始

### 环境要求

- Python 3.12+
- 至少一个 AI 模型的 API Key：
  - [Anthropic API Key](https://console.anthropic.com/)（Claude 模型）
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

```bash
# 基本用法（使用环境变量中配置的模型）
python main.py -i 转录文本.txt

# 指定输出文件名
python main.py -i 转录文本.txt -o 笔记.md

# 指定课程学科
python main.py -i 转录文本.txt -s "深度学习"

# 同时保存 JSON 中间结果
python main.py -i 转录文本.txt --save-json

# 使用 Claude Opus（效果最好，推荐用于重要笔记）
python main.py -i 转录文本.txt -s "大模型与AI编程" -m claude-opus-4-6 --save-json

# 使用 Claude Sonnet（速度更快，性价比高）
python main.py -i 转录文本.txt -m claude-sonnet-4-5-20250929

# 使用 Gemini 模型
python main.py -i 转录文本.txt -m gemini-3-pro-preview

# 使用 GPT 模型
python main.py -i 转录文本.txt -m gpt-4o

# 使用 OpenAI 兼容代理（如第三方 Claude 代理）
python main.py -i 转录文本.txt -m claude-sonnet-4-5-20250929

# 从 YouTube 视频提取字幕并生成笔记
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" -s "计算机科学"

# 仅提取 YouTube 字幕，不生成笔记（保存到 output/transcript/）
python main.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-only
```

### CLI 参数

| 参数 | 说明 | 必填 | 默认值 |
|------|------|:----:|--------|
| `-i, --input` | 输入转录文本文件路径（与 `-u` 二选一） | | — |
| `-u, --url` | YouTube 视频 URL（与 `-i` 二选一） | | — |
| `-o, --output` | 输出 Markdown 文件路径 | | 自动生成 |
| `-s, --subject` | 课程学科 | | — |
| `-m, --model` | 模型名称（支持 Claude / Gemini / GPT） | | 环境变量 `ANTHROPIC_MODEL` / `GEMINI_MODEL` / `GPT_MODEL` 或 `claude-sonnet-4-5-20250929` |
| `--save-json` | 同时保存 JSON 输出 | | `false` |
| `--transcript-only` | 仅提取 YouTube 字幕，不生成笔记 | | `false` |

## 工作原理

```
YouTube URL / 转录文本 (.txt)
    │
    ▼
┌──────────────┐
│  字幕提取     │  YouTube 视频自动提取 Transcript（可选）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  文本分片     │  超过 80,000 字符时按段落边界智能切分
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM API     │  Claude / Gemini / GPT，使用系统提示词指导结构化输出
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  JSON 解析   │  提取标题、章节、术语、复习题等
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  分片合并     │  多片结果去重合并（长文本场景）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Markdown    │  生成目录、折叠面板、术语表、复习题
│  组装输出     │
└──────────────┘
       │
       ▼
  笔记.md / 笔记.json
```

## 项目结构

```
lecture2note/
├── main.py              # CLI 入口
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
├── src/
│   ├── noter.py         # 核心处理：分片、调用 API、解析、合并
│   ├── assembler.py     # JSON → Markdown 格式转换
│   └── transcriber.py   # YouTube Transcript 提取
├── prompts/
│   └── note_system.md   # Claude 系统提示词
└── output/
    ├── transcript/      # YouTube 字幕提取结果
    └── *.md             # 生成的笔记
```

## 输出示例

生成的笔记包含以下结构：

- **标题与摘要** — 自动生成课程标题和 3-5 句概述
- **目录** — 多章节时自动生成可跳转目录
- **分章节内容** — 每节包含正文、要点折叠面板、教师强调内容
- **术语表** — 专业术语与定义的表格汇总
- **复习题** — 覆盖核心知识点的自测题目
- **元信息** — 日期、来源、生成时间等

## 技术栈

- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API 调用（支持原生 API 和 OpenAI 兼容代理）
- [Google GenAI SDK](https://github.com/googleapis/python-genai) — Gemini API 调用
- [OpenAI Python SDK](https://github.com/openai/openai-python) — GPT API 调用及 OpenAI 兼容接口
- [Click](https://click.palletsprojects.com/) — CLI 框架
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — YouTube 字幕提取
- [python-dotenv](https://github.com/theskumar/python-dotenv) — 环境变量管理

## License

MIT
