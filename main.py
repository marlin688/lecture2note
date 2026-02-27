import json
import re
from pathlib import Path

import click
from dotenv import load_dotenv

from src.noter import process_transcript
from src.assembler import assemble_markdown

load_dotenv()

OUTPUT_DIR = Path("output")


def _slugify(text: str) -> str:
    """将标题转为安全的文件名。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return text.strip("_")[:80]


@click.command()
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="输入 txt 文件路径")
@click.option("-o", "--output", "output_path", default=None, type=click.Path(), help="输出 md 文件路径")
@click.option("-s", "--subject", default="", help="课程学科")
@click.option("-m", "--model", default="claude-sonnet-4-5-20250929", help="模型名称")
@click.option("--save-json", is_flag=True, help="同时保存中间 JSON")
def main(input_path, output_path, subject, model, save_json):
    """Lecture2Note - 将课堂录音转写文本整理为结构化笔记"""
    # 1. 读取输入
    transcript = Path(input_path).read_text(encoding="utf-8")
    char_count = len(transcript)
    line_count = transcript.count("\n") + 1
    click.echo(f"📄 已读取: {input_path}")
    click.echo(f"   字符数: {char_count}  行数: {line_count}")

    # 2. 调用 API 生成笔记
    click.echo("🤖 正在调用 Claude 生成笔记...")
    notes = process_transcript(transcript, subject, model)

    section_count = len(notes.get("sections", []))
    term_count = len(notes.get("key_terms", []))
    click.echo(f"✅ 生成完成: {section_count} 个章节, {term_count} 个术语")

    # 3. 组装 Markdown
    OUTPUT_DIR.mkdir(exist_ok=True)

    if output_path is None:
        title = notes.get("title", "untitled")
        filename = _slugify(title) + ".md"
        output_path = OUTPUT_DIR / filename
    else:
        output_path = Path(output_path)

    md_content = assemble_markdown(notes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(md_content, encoding="utf-8")
    click.echo(f"📝 笔记已保存: {output_path}")

    # 4. 保存 JSON
    if save_json:
        json_path = Path(output_path).with_suffix(".json")
        json_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"💾 JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()
