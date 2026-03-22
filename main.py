import json
import os
import re
from pathlib import Path

import click
from dotenv import load_dotenv

from src.noter import process_transcript
from src.assembler import assemble_markdown
from src.transcriber import fetch_transcript, save_transcript
from src.subtitle import generate_subtitle, generate_summary
from src.downloader import print_formats

load_dotenv()

OUTPUT_DIR = Path("output")


def _slugify(text: str) -> str:
    """将标题转为安全的文件名。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    result = text.strip("_")[:80]
    return result if result else "untitled"


@click.command()
@click.option("-i", "--input", "input_path", default=None, type=click.Path(exists=True), help="输入 txt 文件路径")
@click.option("-u", "--url", "youtube_url", default=None, help="YouTube 视频 URL（自动提取 Transcript）")
@click.option("-o", "--output", "output_path", default=None, type=click.Path(), help="输出 md 文件路径")
@click.option("-s", "--subject", default="", help="课程学科")
@click.option("-m", "--model", default=None, help="模型名称（默认读取环境变量 ANTHROPIC_MODEL / GEMINI_MODEL）")
@click.option("--save-json", is_flag=True, help="同时保存中间 JSON")
@click.option("--transcript-only", is_flag=True, help="仅提取 Transcript，不生成笔记")
@click.option("--subtitle", "subtitle_lang", default=None, type=click.Choice(["zh", "bilingual", "en"]), help="生成字幕文件（zh=中文, bilingual=中英双语, en=英文原文）")
@click.option("--list-formats", "list_fmts", is_flag=True, help="列出视频可用的下载格式和地址")
@click.option("--whisper-model", default="medium", type=click.Choice(["tiny", "base", "small", "medium", "large"]), help="Whisper 模型 (默认 medium)")
@click.option("--no-whisper", is_flag=True, help="不使用 Whisper，回退到 YouTube 字幕")
@click.option("--summary", is_flag=True, help="生成视频摘要 Markdown（含建议中文标题）")
def main(input_path, youtube_url, output_path, subject, model, save_json, transcript_only, subtitle_lang, list_fmts, whisper_model, no_whisper, summary):
    """Lecture2Note - 将课堂录音转写文本整理为结构化笔记"""
    # 0. 参数校验
    if not input_path and not youtube_url:
        raise click.UsageError("请通过 -i 指定输入文件或通过 -u 指定 YouTube 视频 URL")

    # 0.0 列出下载格式（独立功能，不需要其他参数）
    if list_fmts:
        if not youtube_url:
            raise click.UsageError("--list-formats 需要通过 -u 指定 YouTube 视频 URL")
        print_formats(youtube_url)
        return

    # 0.0.1 生成字幕（独立功能）
    if subtitle_lang:
        if not youtube_url:
            raise click.UsageError("--subtitle 需要通过 -u 指定 YouTube 视频 URL")
        if model is None:
            model = "claude-sonnet-4-5-20250929"
        srt_path = generate_subtitle(
            youtube_url, model,
            target_lang=subtitle_lang,
            use_whisper=not no_whisper,
            whisper_model=whisper_model,
        )
        click.echo(f"\n✅ 字幕文件: {srt_path}")
        # 如果同时指定了 --summary，生成摘要
        if summary:
            generate_summary(youtube_url, model)
        # 顺便打印下载地址
        try:
            print_formats(youtube_url)
        except Exception:
            click.echo("⚠️ 获取下载地址失败（可能未安装 yt-dlp），可手动安装: pip install yt-dlp")
        return

    # 0.0.2 单独生成摘要（已有字幕时）
    if summary:
        if not youtube_url:
            raise click.UsageError("--summary 需要通过 -u 指定 YouTube 视频 URL")
        if model is None:
            model = "claude-sonnet-4-5-20250929"
        generate_summary(youtube_url, model)
        return

    # 0.1 如果提供了 YouTube URL，先提取 Transcript
    if youtube_url:
        click.echo(f"🎬 正在从 YouTube 提取 Transcript...")
        video_id, transcript = fetch_transcript(youtube_url)
        txt_path = save_transcript(video_id, transcript)
        click.echo(f"📄 Transcript 已保存: {txt_path}")
        if transcript_only:
            return
    else:
        transcript = Path(input_path).read_text(encoding="utf-8")

    # 0.2 确定模型：优先命令行参数 > 环境变量 > 硬编码默认
    if model is None:
        model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("GEMINI_MODEL") or os.environ.get("GPT_MODEL") or "claude-sonnet-4-5-20250929"

    # 1. 读取输入统计
    char_count = len(transcript)
    line_count = transcript.count("\n") + 1
    source = youtube_url or input_path
    click.echo(f"📄 输入来源: {source}")
    click.echo(f"   字符数: {char_count}  行数: {line_count}")

    # 2. 调用 API 生成笔记
    if model.startswith("gemini"):
        provider = "Gemini"
    elif model.startswith(("gpt", "o1", "o3", "o4")):
        provider = "GPT"
    else:
        provider = "Claude"
    click.echo(f"🤖 正在调用 {provider} ({model}) 生成笔记...")
    notes = process_transcript(transcript, subject, model)

    # 3. 组装 Markdown 并保存
    OUTPUT_DIR.mkdir(exist_ok=True)

    if "_raw_markdown" in notes:
        # 模型直接输出了 Markdown，跳过 assembler
        md_content = notes["_raw_markdown"]
        click.echo("✅ 生成完成 (模型直出 Markdown)")
        if output_path is None:
            # 从 Markdown 内容提取第一个 # 标题行
            title = "untitled"
            for line in md_content.strip().split("\n"):
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break
            filename = _slugify(title) + ".md"
            output_path = OUTPUT_DIR / filename
        else:
            output_path = Path(output_path)
    else:
        section_count = len(notes.get("sections", []))
        term_count = len(notes.get("key_terms", []))
        click.echo(f"✅ 生成完成: {section_count} 个章节, {term_count} 个术语")
        md_content = assemble_markdown(notes)
        if output_path is None:
            title = notes.get("title", "untitled")
            filename = _slugify(title) + ".md"
            output_path = OUTPUT_DIR / filename
        else:
            output_path = Path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content, encoding="utf-8")
    click.echo(f"📝 笔记已保存: {output_path}")

    # 4. 保存 JSON
    if save_json:
        json_path = Path(output_path).with_suffix(".json")
        json_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"💾 JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()
