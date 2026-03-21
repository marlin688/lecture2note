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

    优先获取人工字幕，没有时才回退到自动生成字幕。

    Returns:
        (video_id, snippets, is_generated)
    """
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    # 按语言优先级查找，优先人工字幕
    manual = None
    generated = None
    for lang in languages:
        for t in transcript_list:
            if t.language_code == lang:
                if not t.is_generated:
                    manual = t
                    break
                elif generated is None:
                    generated = t
        if manual:
            break

    chosen = manual or generated
    if chosen is None:
        # 回退到默认 fetch
        transcript = api.fetch(video_id, languages=languages)
        is_generated = True
    else:
        transcript = chosen.fetch()
        is_generated = chosen.is_generated
        label = "自动生成" if is_generated else "人工"
        click.echo(f"   ℹ️ 使用{label}字幕 ({chosen.language})")

    return video_id, transcript.snippets, is_generated


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


def _load_merge_prompt() -> str:
    prompt_path = PROMPTS_DIR / "merge_system.md"
    return prompt_path.read_text(encoding="utf-8")


def _parse_groups(raw: str, batch_start: int, batch_size: int) -> list[tuple[int, int]]:
    """解析 LLM 返回的分组信息，返回 [(start_idx, end_idx), ...]（全局 0-based）。

    LLM 可能用全局编号（151-156）或批次内编号（1-6），都能处理。
    """
    groups = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(\d+)\s*-\s*(\d+)", line)
        if match:
            s, e = int(match.group(1)), int(match.group(2))
            groups.append((s, e))

    if not groups:
        return None

    # 判断是全局编号还是批次内编号：看第一个分组的起始值
    first_start = groups[0][0]
    if first_start == batch_start + 1:
        # 全局编号（1-based），转为全局 0-based
        groups = [(s - 1, e - 1) for s, e in groups]
    elif first_start == 1:
        # 批次内编号（1-based），转为全局 0-based
        groups = [(batch_start + s - 1, batch_start + e - 1) for s, e in groups]
    else:
        return None

    # 验证：覆盖完整、不重叠、顺序正确
    expected = batch_start
    for s, e in groups:
        if s != expected or e < s or e >= batch_start + batch_size:
            return None
        expected = e + 1
    if expected != batch_start + batch_size:
        return None

    return groups


def _merge_one_chunk(args: tuple) -> tuple[int, str]:
    """对一批碎片发送分组请求，返回 (chunk_idx, raw_response)。"""
    chunk_idx, user_message, system_prompt, model = args
    raw = _call_translate_llm(system_prompt, user_message, model)
    return chunk_idx, raw


def merge_subtitle_fragments(snippets, model: str) -> list[MergedEntry]:
    """用 LLM 判断句子边界，将碎片合并为完整句子。

    LLM 只输出分组编号，时间戳由代码从原始数据计算，保证 100% 准确。
    """
    if not snippets:
        return []

    system_prompt = _load_merge_prompt()
    total = len(snippets)

    # 分批发送（每批 150 条碎片）
    MERGE_BATCH = 150
    all_groups = []
    batch_args = []

    for chunk_idx, batch_start in enumerate(range(0, total, MERGE_BATCH)):
        batch_end = min(batch_start + MERGE_BATCH, total)
        lines = []
        for i in range(batch_start, batch_end):
            lines.append(f"{i + 1}|{snippets[i].text}")
        user_message = "\n".join(lines)
        batch_args.append((chunk_idx, user_message, system_prompt, model))

    total_batches = len(batch_args)
    click.echo(f"   🔄 分组中 ({total_batches} 批, {MAX_CONCURRENT} 路并发)...")

    raw_results = [None] * total_batches
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(_merge_one_chunk, a): a[0] for a in batch_args}
        for future in as_completed(futures):
            idx, raw = future.result()
            raw_results[idx] = raw
            completed += 1

    # 解析分组并合并
    offset = 0
    for chunk_idx in range(total_batches):
        batch_start = chunk_idx * MERGE_BATCH
        batch_end = min(batch_start + MERGE_BATCH, total)
        batch_size = batch_end - batch_start

        groups = _parse_groups(raw_results[chunk_idx], batch_start, batch_size)
        if groups is None:
            # LLM 分组失败，回退到规则合并（每 4 条一组）
            click.echo(f"   ⚠️ 第 {chunk_idx + 1} 批分组解析失败，使用规则回退")
            for i in range(0, batch_size, 4):
                e = min(i + 3, batch_size - 1)
                all_groups.append((batch_start + i, batch_start + e))
        else:
            all_groups.extend(groups)

    # 后处理：拆分过大的组（超过 5 个碎片或 12 秒）
    MAX_GROUP_SIZE = 5
    MAX_GROUP_DURATION = 12.0
    final_groups = []
    for s, e in all_groups:
        group_size = e - s + 1
        group_dur = (snippets[e].start + snippets[e].duration) - snippets[s].start
        if group_size <= MAX_GROUP_SIZE and group_dur <= MAX_GROUP_DURATION:
            final_groups.append((s, e))
        else:
            # 按 MAX_GROUP_SIZE 拆分
            for i in range(s, e + 1, MAX_GROUP_SIZE):
                sub_e = min(i + MAX_GROUP_SIZE - 1, e)
                final_groups.append((i, sub_e))

    # 根据分组生成合并条目，时间戳从原始 snippets 计算
    merged = []
    for start_idx, end_idx in final_groups:
        texts = [snippets[i].text for i in range(start_idx, end_idx + 1)]
        entry_start = snippets[start_idx].start
        entry_end = snippets[end_idx].start + snippets[end_idx].duration
        merged.append(MergedEntry(
            start=entry_start,
            end=entry_end,
            text=" ".join(texts).strip(),
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


def generate_subtitle(
    url: str,
    model: str,
    target_lang: str = "zh",
    use_whisper: bool = True,
    whisper_model: str = "medium",
) -> Path:
    """主流程：提取/转录英文字幕 → 翻译 → 输出 SRT 文件。

    Args:
        url: YouTube 视频 URL
        model: 翻译用的 LLM 模型
        target_lang: "en" 仅英文, "zh" 纯中文, "bilingual" 中英双语
        use_whisper: True 用 Whisper 转录, False 用 YouTube 字幕
        whisper_model: Whisper 模型名称 (tiny/base/small/medium/large)

    Returns:
        输出的 SRT 文件路径
    """
    from src.transcriber import extract_video_id
    video_id = extract_video_id(url)
    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)

    if use_whisper:
        # Whisper 路径：下载音频 → 本地转录
        from src.whisper_transcriber import download_audio, transcribe_to_srt

        click.echo("🎬 Whisper 转录模式")
        video_dir = SUBTITLE_DIR / video_id
        audio_path = download_audio(url, video_dir)

        # 同时保存 YouTube 原始字幕（如果有的话）
        youtube_srt = None
        try:
            _, snippets, _ = fetch_subtitle_snippets(url)
            youtube_srt = snippets_to_srt(snippets)
            youtube_path = video_dir / "subtitle_en_youtube.srt"
            youtube_path.write_text(youtube_srt, encoding="utf-8")
            click.echo(f"   📄 YouTube 原始字幕已保存: {youtube_path}")
        except Exception:
            click.echo("   ℹ️ YouTube 字幕不可用，跳过")

        en_srt = transcribe_to_srt(str(audio_path), model_name=whisper_model)

        # 质量检测：如果 Whisper 结果大量空条目，自动回退到 YouTube 字幕
        srt_entries = re.split(r"\n\n+", en_srt.strip())
        non_empty = sum(1 for e in srt_entries if len(e.strip().split("\n")) >= 3 and e.strip().split("\n")[2].strip())
        if non_empty < 5 and youtube_srt:
            click.echo(f"   ⚠️ Whisper 转录质量不佳（仅 {non_empty} 条有效），回退到 YouTube 字幕")
            en_srt = youtube_srt
    else:
        # YouTube 字幕回退路径
        click.echo("🎬 YouTube 字幕模式")
        video_id, snippets, is_generated = fetch_subtitle_snippets(url)
        click.echo(f"   ✓ 获取到 {len(snippets)} 条字幕片段")

        if is_generated:
            # 自动字幕：用 LLM 合并碎片
            merged = merge_subtitle_fragments(snippets, model)
            en_srt = merged_entries_to_srt(merged)
            click.echo(f"   ✓ 合并为 {len(merged)} 条完整字幕")
        else:
            en_srt = snippets_to_srt(snippets)
            click.echo("   ✓ 人工字幕，无需合并")

    # 输出目录：Whisper 模式放在 video_dir 下，YouTube 模式放在 SUBTITLE_DIR 下
    out_dir = SUBTITLE_DIR / video_id if use_whisper else SUBTITLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存英文 SRT
    en_path = out_dir / "subtitle_en.srt"
    en_path.write_text(en_srt, encoding="utf-8")
    click.echo(f"   📄 英文字幕已保存: {en_path}")

    if target_lang == "en":
        return en_path

    # 翻译
    label = "中文" if target_lang == "zh" else "中英双语"
    click.echo(f"🌐 正在生成{label}字幕 (模型: {model})...")
    translated_srt = translate_srt(en_srt, model, mode=target_lang)

    suffix = "_zh" if target_lang == "zh" else "_bilingual"
    out_path = out_dir / f"subtitle{suffix}.srt"
    normalized = _normalize_srt(translated_srt, en_srt)
    out_path.write_text(normalized, encoding="utf-8")
    click.echo(f"   📄 {label}字幕已保存: {out_path}")

    return out_path
