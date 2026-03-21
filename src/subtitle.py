"""字幕提取、翻译与 SRT 生成。"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from src.transcriber import extract_video_id
from youtube_transcript_api import YouTubeTranscriptApi


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SUBTITLE_DIR = Path("output/subtitle")

# 每批发送给 LLM 翻译的 SRT 条目数
BATCH_ENTRIES = 100
MAX_CONCURRENT = 4

# 碎片合并参数
MERGE_MAX_DURATION = 10.0   # 合并后单条字幕最大时长（秒）
MERGE_MAX_CHARS = 80        # 合并后单条字幕最大英文字符数
MERGE_GAP_THRESHOLD = 1.5   # 超过此间隔（秒）强制分段


def _format_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间格式: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fetch_subtitle_snippets(url: str, languages: tuple[str, ...] = ("en",)):
    """获取 YouTube 视频的字幕片段（带时间戳）。

    Returns:
        (video_id, list of snippets with .text, .start, .duration)
    """
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=languages)
    return video_id, transcript.snippets


def snippets_to_srt(snippets) -> str:
    """将字幕片段转为 SRT 格式字符串。"""
    lines = []
    for i, snippet in enumerate(snippets):
        idx = i + 1
        start = _format_srt_time(snippet.start)
        end = _format_srt_time(snippet.start + snippet.duration)
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(snippet.text)
        lines.append("")
    return "\n".join(lines)


from dataclasses import dataclass


@dataclass
class MergedEntry:
    start: float
    end: float
    text: str


def merge_subtitle_fragments(snippets) -> list[MergedEntry]:
    """将 YouTube 自动字幕的碎片合并为完整句子级条目。

    合并策略：
    - 连续碎片如果时间重叠或间隔 < GAP_THRESHOLD 秒，则合并
    - 遇到句末标点（. ? !）且已有一定长度时终止当前组
    - 超过 MAX_DURATION 或 MAX_CHARS 时强制终止
    """
    if not snippets:
        return []

    merged = []
    cur_start = snippets[0].start
    cur_end = snippets[0].start + snippets[0].duration
    cur_texts = [snippets[0].text]

    for i in range(1, len(snippets)):
        s = snippets[i]
        s_end = s.start + s.duration
        gap = s.start - cur_end
        cur_text_joined = " ".join(cur_texts)
        cur_duration = cur_end - cur_start

        # 判断是否应该开始新组
        should_break = False

        # 时间间隔过大 → 说话人停顿或换人
        if gap >= MERGE_GAP_THRESHOLD:
            should_break = True

        # 已超过时长或字符限制
        if cur_duration >= MERGE_MAX_DURATION or len(cur_text_joined) >= MERGE_MAX_CHARS:
            should_break = True

        # 前一条以句末标点结尾且已有一定长度
        if (cur_texts[-1].rstrip().endswith((".", "?", "!"))
                and len(cur_text_joined) > 30):
            should_break = True

        if should_break:
            merged.append(MergedEntry(
                start=cur_start, end=cur_end,
                text=" ".join(cur_texts).strip(),
            ))
            cur_start = s.start
            cur_end = s_end
            cur_texts = [s.text]
        else:
            cur_end = max(cur_end, s_end)
            cur_texts.append(s.text)

    # 最后一组
    if cur_texts:
        merged.append(MergedEntry(
            start=cur_start, end=cur_end,
            text=" ".join(cur_texts).strip(),
        ))

    return merged


def merged_entries_to_srt(entries: list[MergedEntry]) -> str:
    """将合并后的条目转为 SRT 格式。"""
    lines = []
    for i, entry in enumerate(entries):
        idx = i + 1
        start = _format_srt_time(entry.start)
        end = _format_srt_time(entry.end)
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(entry.text)
        lines.append("")
    return "\n".join(lines)


def _fix_timestamp(ts_line: str) -> str:
    """修复 LLM 可能破坏的时间戳格式，确保 HH:MM:SS,mmm --> HH:MM:SS,mmm。"""
    match = re.match(r"(\d{1,2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2},\d{3})", ts_line)
    if not match:
        return ts_line
    start, end = match.group(1), match.group(2)
    # 补齐小时位到2位
    def pad_hours(ts):
        parts = ts.split(":")
        parts[0] = parts[0].zfill(2)
        return ":".join(parts)
    return f"{pad_hours(start)} --> {pad_hours(end)}"


def _normalize_srt(srt_text: str, en_srt: str) -> str:
    """规范化翻译后的 SRT：修复时间戳、确保条目数与原文一致。"""
    text = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    entries = re.split(r"\n\n+", text.strip())

    # 从英文原版提取正确的序号和时间戳
    en_text = en_srt.replace("\r\n", "\n").replace("\r", "\n")
    en_entries = re.split(r"\n\n+", en_text.strip())

    fixed = []
    for i, entry in enumerate(entries):
        lines = entry.strip().split("\n")
        if i < len(en_entries):
            en_lines = en_entries[i].strip().split("\n")
            # 用原版的序号和时间戳替换，只保留翻译后的文本行
            seq = en_lines[0]
            ts = en_lines[1]
            text_lines = lines[2:]  # 翻译后的文本（可能1行或2行）
            fixed.append("\n".join([seq, ts] + text_lines))
        else:
            # 超出原文条目数的部分，仅修复时间戳
            if len(lines) >= 2:
                lines[1] = _fix_timestamp(lines[1])
            fixed.append("\n".join(lines))

    return "\n\n".join(fixed) + "\n"


def _split_srt(srt_text: str, batch_entries: int = BATCH_ENTRIES) -> list[str]:
    """将 SRT 文本按条目数分段。每段包含 batch_entries 个条目。"""
    # SRT 条目之间用空行分隔
    entries = re.split(r"\n\n+", srt_text.strip())
    chunks = []
    for i in range(0, len(entries), batch_entries):
        chunk = "\n\n".join(entries[i:i + batch_entries])
        chunks.append(chunk)
    return chunks


def _load_translate_prompt(mode: str = "bilingual") -> str:
    if mode == "zh":
        prompt_path = PROMPTS_DIR / "translate_zh_system.md"
    else:
        prompt_path = PROMPTS_DIR / "translate_system.md"
    return prompt_path.read_text(encoding="utf-8")


def _call_translate_llm(system_prompt: str, user_message: str, model: str) -> str:
    """调用 LLM 进行翻译。"""
    from src.noter import _is_gemini_model, _is_gpt_model

    if _is_gemini_model(model):
        return _call_translate_gemini(system_prompt, user_message, model)
    if _is_gpt_model(model):
        return _call_translate_gpt(system_prompt, user_message, model)
    return _call_translate_claude(system_prompt, user_message, model)


def _call_translate_claude(system_prompt: str, user_message: str, model: str) -> str:
    import anthropic
    import httpx
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(proxy=None, trust_env=False),
    )
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _call_translate_gemini(system_prompt: str, user_message: str, model: str) -> str:
    from google import genai
    import httpx
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 GEMINI_API_KEY")
    base_url = os.environ.get("GEMINI_BASE_URL")
    http_opts = {"httpxClient": httpx.Client(proxy=None, trust_env=False)}
    if base_url:
        http_opts["base_url"] = base_url
    proxy_vars = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if key in os.environ:
            proxy_vars[key] = os.environ.pop(key)
    try:
        client = genai.Client(api_key=api_key, http_options=http_opts)
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=16000,
            ),
        )
    finally:
        os.environ.update(proxy_vars)
    return response.text


def _call_translate_gpt(system_prompt: str, user_message: str, model: str) -> str:
    from openai import OpenAI
    from src.noter import _is_gpt_model
    import httpx
    if _is_gpt_model(model):
        base_url = os.environ.get("GPT_BASE_URL")
        api_key = os.environ.get("GPT_API_KEY", "")
    else:
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("GPT_BASE_URL")
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GPT_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 API KEY")
    client_kwargs = {}
    if base_url:
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        client_kwargs["base_url"] = base_url
        client_kwargs["http_client"] = httpx.Client(proxy=None, trust_env=False)
    client = OpenAI(api_key=api_key, **client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=16000,
    )
    return response.choices[0].message.content


def _translate_one_chunk(args: tuple) -> tuple[int, str]:
    """翻译单个 SRT 分段，返回 (chunk_index, translated_srt_chunk)。"""
    chunk_idx, srt_chunk, system_prompt, model = args
    raw = _call_translate_llm(system_prompt, srt_chunk, model)
    # 清理可能的 ```srt 包裹
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return chunk_idx, text


def translate_srt(en_srt: str, model: str, mode: str = "bilingual") -> str:
    """将英文 SRT 并发翻译为中文/双语 SRT。

    Args:
        mode: "zh" 纯中文, "bilingual" 中英双语
    """
    system_prompt = _load_translate_prompt(mode)
    chunks = _split_srt(en_srt)
    total_chunks = len(chunks)

    click.echo(f"   📦 共 {total_chunks} 段，{MAX_CONCURRENT} 路并发翻译...")

    batch_args = [(i, chunk, system_prompt, model) for i, chunk in enumerate(chunks)]

    results = [None] * total_chunks
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(_translate_one_chunk, a): a[0] for a in batch_args}
        for future in as_completed(futures):
            chunk_idx, translated = future.result()
            results[chunk_idx] = translated
            completed += 1
            click.echo(f"   ✓ 完成 {completed}/{total_chunks} 段")

    return "\n\n".join(results)


def generate_subtitle(url: str, model: str, target_lang: str = "zh") -> Path:
    """主流程：提取英文字幕 → 翻译 → 输出 SRT 文件。

    Args:
        target_lang: "en" 仅英文, "zh" 纯中文, "bilingual" 中英双语

    Returns:
        输出的 SRT 文件路径
    """
    click.echo("🎬 正在提取英文字幕...")
    video_id, snippets = fetch_subtitle_snippets(url)
    click.echo(f"   ✓ 获取到 {len(snippets)} 条字幕片段")

    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 保存原始碎片英文 SRT
    raw_en_srt = snippets_to_srt(snippets)
    raw_en_path = SUBTITLE_DIR / f"{video_id}_en_raw.srt"
    raw_en_path.write_text(raw_en_srt, encoding="utf-8")
    click.echo(f"   📄 原始英文字幕已保存: {raw_en_path}")

    # 2. 合并碎片为完整句子
    merged = merge_subtitle_fragments(snippets)
    merged_en_srt = merged_entries_to_srt(merged)
    merged_en_path = SUBTITLE_DIR / f"{video_id}_en.srt"
    merged_en_path.write_text(merged_en_srt, encoding="utf-8")
    click.echo(f"   ✓ 合并为 {len(merged)} 条完整字幕: {merged_en_path}")

    if target_lang == "en":
        return merged_en_path

    # 3. 翻译合并后的 SRT（中英文时间轴一致）
    label = "中文" if target_lang == "zh" else "中英双语"
    click.echo(f"🌐 正在生成{label}字幕 (模型: {model})...")
    translated_srt = translate_srt(merged_en_srt, model, mode=target_lang)

    suffix = "_zh" if target_lang == "zh" else "_bilingual"
    out_path = SUBTITLE_DIR / f"{video_id}{suffix}.srt"
    # 用合并后的英文时间戳修复翻译后可能损坏的时间戳
    normalized = _normalize_srt(translated_srt, merged_en_srt)
    out_path.write_text(normalized, encoding="utf-8")
    click.echo(f"   📄 {label}字幕已保存: {out_path}")

    return out_path
