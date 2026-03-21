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
BATCH_ENTRIES = 150
MAX_CONCURRENT = 4


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


def _split_srt(srt_text: str, batch_entries: int = BATCH_ENTRIES) -> list[str]:
    """将 SRT 文本按条目数分段。每段包含 batch_entries 个条目。"""
    # SRT 条目之间用空行分隔
    entries = re.split(r"\n\n+", srt_text.strip())
    chunks = []
    for i in range(0, len(entries), batch_entries):
        chunk = "\n\n".join(entries[i:i + batch_entries])
        chunks.append(chunk)
    return chunks


def _load_translate_prompt() -> str:
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


def translate_srt(en_srt: str, model: str) -> str:
    """将英文 SRT 并发翻译为中文 SRT。

    直接把 SRT 分段发给 LLM，保留格式只翻译文本行。
    """
    system_prompt = _load_translate_prompt()
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


def generate_subtitle(url: str, model: str, target_lang: str = "bilingual") -> Path:
    """主流程：提取英文字幕 → 翻译为双语 → 输出 SRT 文件。

    Args:
        target_lang: "en" 仅英文, "bilingual" 中英双语

    Returns:
        输出的 SRT 文件路径
    """
    click.echo("🎬 正在提取英文字幕...")
    video_id, snippets = fetch_subtitle_snippets(url)
    click.echo(f"   ✓ 获取到 {len(snippets)} 条字幕片段")

    # 生成并保存英文 SRT
    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    en_srt = snippets_to_srt(snippets)
    en_path = SUBTITLE_DIR / f"{video_id}_en.srt"
    en_path.write_text(en_srt, encoding="utf-8")
    click.echo(f"   📄 英文字幕已保存: {en_path}")

    if target_lang == "en":
        return en_path

    # 翻译为中英双语 SRT
    click.echo(f"🌐 正在生成中英双语字幕 (模型: {model})...")
    bilingual_srt = translate_srt(en_srt, model)

    bilingual_path = SUBTITLE_DIR / f"{video_id}_bilingual.srt"
    bilingual_path.write_text(bilingual_srt, encoding="utf-8")
    click.echo(f"   📄 双语字幕已保存: {bilingual_path}")

    return bilingual_path
