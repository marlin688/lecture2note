"""字幕提取、翻译与 SRT 生成。"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from l2n.transcriber import extract_video_id
from youtube_transcript_api import YouTubeTranscriptApi


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
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


def _parse_srt_entries(srt_text: str) -> list[tuple[str, str, str]]:
    """解析 SRT 文本为结构化列表。

    Returns:
        [(序号, 时间戳行, 文本内容), ...]
    """
    text = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    entries = re.split(r"\n\n+", text.strip())
    result = []
    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) >= 3:
            result.append((lines[0].strip(), lines[1].strip(), " ".join(lines[2:])))
    return result


def _split_text_batches(entries: list[tuple[str, str, str]], batch_size: int = BATCH_ENTRIES) -> list[list[tuple[str, str, str]]]:
    """将解析后的 SRT 条目列表按批次分段。"""
    batches = []
    for i in range(0, len(entries), batch_size):
        batches.append(entries[i:i + batch_size])
    return batches


def _load_translate_prompt(mode: str = "bilingual") -> str:
    if mode == "zh":
        prompt_path = PROMPTS_DIR / "translate_zh_system.md"
    else:
        prompt_path = PROMPTS_DIR / "translate_system.md"
    return prompt_path.read_text(encoding="utf-8")


def _call_translate_llm(system_prompt: str, user_message: str, model: str) -> str:
    """调用 LLM 进行翻译。"""
    from l2n.noter import _is_gemini_model, _is_gpt_model

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
        http_client=httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0)),
    )
    # 使用流式模式兼容强制返回 SSE 的中转平台
    text_parts = []
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            text_parts.append(text)
    return "".join(text_parts)


def _call_translate_gemini(system_prompt: str, user_message: str, model: str) -> str:
    from google import genai
    import httpx
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 GEMINI_API_KEY")
    base_url = os.environ.get("GEMINI_BASE_URL")
    http_opts = {"httpxClient": httpx.Client()}
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
    from l2n.noter import _is_gpt_model
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
        client_kwargs["http_client"] = httpx.Client()
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


def _translate_one_chunk(args: tuple) -> tuple[int, dict[int, str]]:
    """翻译单个纯文本批次，返回 (chunk_index, {全局序号: 翻译文本})。

    发送格式：每行带序号前缀 "[N] 英文文本"
    期望返回：每行带序号前缀 "[N] 中文翻译"
    按序号匹配，而非位置匹配，确保即使 LLM 漏行/乱序也能精确对齐。
    """
    chunk_idx, numbered_lines, global_indices, system_prompt, model = args
    user_message = "\n".join(numbered_lines)
    raw = _call_translate_llm(system_prompt, user_message, model)

    # 清理可能的 ``` 包裹
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    # 按序号解析返回结果（严格匹配 [N] 格式）
    translated_map = {}
    all_lines = []  # 保留所有非空行用于位置兜底
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[(\d+)\]\s*(.+)", line)
        if m:
            seq = int(m.group(1))
            translated_map[seq] = m.group(2).strip()
            all_lines.append(m.group(2).strip())
        else:
            all_lines.append(line)

    # 如果序号匹配覆盖不足 90%，回退到纯位置匹配
    matched_count = sum(1 for idx in global_indices if idx in translated_map)
    if matched_count < len(global_indices) * 0.9 and len(all_lines) == len(global_indices):
        click.echo(f"   ℹ️ 第 {chunk_idx + 1} 段序号匹配不足，回退到位置匹配")
        translated_map = {}
        for idx, line in zip(global_indices, all_lines):
            translated_map[idx] = line
    else:
        missing = [idx for idx in global_indices if idx not in translated_map]
        if missing:
            click.echo(f"   ⚠️ 第 {chunk_idx + 1} 段有 {len(missing)} 行未匹配到序号，将用英文原文兜底")

    return chunk_idx, translated_map


def translate_srt(en_srt: str, model: str, mode: str = "bilingual") -> str:
    """将英文 SRT 并发翻译为中文/双语 SRT。

    策略：
    1. 从 SRT 中提取纯文本，每行加 [序号] 前缀发给 LLM
    2. LLM 返回带序号的翻译，按序号精确匹配（非位置匹配）
    3. 将翻译文本注入回原始时间戳模板
    LLM 全程不接触时间戳，按序号对齐，双重保障杜绝错位。
    """
    system_prompt = _load_translate_prompt(mode)
    entries = _parse_srt_entries(en_srt)
    batches = _split_text_batches(entries)
    total_chunks = len(batches)

    click.echo(f"   📦 共 {total_chunks} 段，{MAX_CONCURRENT} 路并发翻译...")

    # 构建批次参数：每行带全局序号前缀 "[N] 文本"
    batch_args = []
    global_offset = 0
    for i, batch in enumerate(batches):
        numbered_lines = []
        global_indices = []
        for j, (seq, ts, text) in enumerate(batch):
            global_idx = global_offset + j + 1  # 1-based 全局序号
            numbered_lines.append(f"[{global_idx}] {text}")
            global_indices.append(global_idx)
        batch_args.append((i, numbered_lines, global_indices, system_prompt, model))
        global_offset += len(batch)

    results = [None] * total_chunks
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(_translate_one_chunk, a): a[0] for a in batch_args}
        for future in as_completed(futures):
            chunk_idx, translated_map = future.result()
            results[chunk_idx] = translated_map
            completed += 1
            click.echo(f"   ✓ 完成 {completed}/{total_chunks} 段")

    # 合并所有批次的翻译结果
    all_translated = {}
    for translated_map in results:
        all_translated.update(translated_map)

    # 定点补译：对未翻译的行及其相邻行，带上下文重新翻译
    fallback_indices = set(i + 1 for i, (seq, ts, en_text) in enumerate(entries)
                          if (i + 1) not in all_translated)
    if fallback_indices:
        # 扩展补译范围：未翻译行 + 前后各 2 行（相邻行可能也错位了）
        EXPAND = 2
        retranslate_indices = set()
        for idx in fallback_indices:
            for offset in range(-EXPAND, EXPAND + 1):
                j = idx + offset
                if 1 <= j <= len(entries):
                    retranslate_indices.add(j)
        retranslate_indices = sorted(retranslate_indices)

        click.echo(f"   🔄 补译 {len(retranslate_indices)} 条（含 {len(fallback_indices)} 条未翻译 + 相邻行）...")
        numbered_lines = [f"[{idx}] {entries[idx - 1][2]}" for idx in retranslate_indices]
        retry_prompt = "\n".join(numbered_lines)

        raw = _call_translate_llm(system_prompt, retry_prompt, model)
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\[(\d+)\]\s*(.+)", line)
            if m:
                seq = int(m.group(1))
                if seq in retranslate_indices:
                    all_translated[seq] = m.group(2).strip()

        still_missing = [idx for idx in fallback_indices if idx not in all_translated]
        if still_missing:
            click.echo(f"   ⚠️ 补译后仍有 {len(still_missing)} 条未翻译")
        else:
            click.echo(f"   ✅ 补译完成，所有行已翻译")

    # 将翻译结果注入回原始时间戳模板
    srt_lines = []
    for i, (seq, ts, en_text) in enumerate(entries):
        global_idx = i + 1
        zh_text = all_translated.get(global_idx, en_text)  # 匹配不上则用英文兜底
        srt_lines.append(seq)
        srt_lines.append(ts)
        if mode == "bilingual":
            srt_lines.append(en_text)
        srt_lines.append(zh_text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def proofread_en_srt(en_srt: str, zh_srt: str, model: str) -> str:
    """参考中文翻译校对英文字幕中的 ASR 错误，返回修正后的英文 SRT。

    策略：将英文+中文逐行配对发给 LLM，LLM 只返回需要修正的行。
    分批处理以控制单次请求大小。
    """
    system_prompt = (PROMPTS_DIR / "proofread_system.md").read_text(encoding="utf-8")

    en_entries = _parse_srt_entries(en_srt)
    zh_entries = _parse_srt_entries(zh_srt)

    if len(en_entries) != len(zh_entries):
        click.echo("   ⚠️ 中英条目数不一致，跳过校对")
        return en_srt

    # 分批：每批 100 条，4 路并发
    BATCH_SIZE = 100
    all_corrections: dict[int, str] = {}
    total_batches = (len(en_entries) + BATCH_SIZE - 1) // BATCH_SIZE

    # 构建批次参数
    batch_args = []
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(en_entries))
        numbered_lines = []
        for i in range(start, end):
            seq = i + 1
            en_text = en_entries[i][2]
            zh_text = zh_entries[i][2]
            numbered_lines.append(f"[{seq}] {en_text} ||| {zh_text}")
        batch_args.append((batch_idx, "\n".join(numbered_lines), system_prompt, model))

    click.echo(f"   📦 共 {total_batches} 批，{MAX_CONCURRENT} 路并发校对...")

    def _proofread_one(args):
        batch_idx, user_message, sys_prompt, mdl = args
        raw = _call_translate_llm(sys_prompt, user_message, mdl)
        corrections = {}
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\[(\d+)\]\s*(.+)", line)
            if m:
                seq = int(m.group(1))
                corrected = m.group(2).strip()
                if "|||" in corrected:
                    corrected = corrected.split("|||")[0].strip()
                corrections[seq] = corrected
        return batch_idx, corrections

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(_proofread_one, a): a[0] for a in batch_args}
        for future in as_completed(futures):
            batch_idx, corrections = future.result()
            all_corrections.update(corrections)
            completed += 1
            click.echo(f"   ✓ 校对完成 {completed}/{total_batches} 批")

    if not all_corrections:
        click.echo("   ✅ 校对完成：无需修正")
        return en_srt

    # 应用修正到英文 SRT
    click.echo(f"   ✏️ 校对修正 {len(all_corrections)} 条")
    srt_lines = []
    for i, (seq, ts, en_text) in enumerate(en_entries):
        global_idx = i + 1
        corrected = all_corrections.get(global_idx)
        text = corrected if corrected else en_text
        srt_lines.append(str(seq))
        srt_lines.append(ts)
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def _srt_to_plain_text(srt_text: str) -> str:
    """将 SRT 字幕提取为纯文本（去掉序号和时间戳）。"""
    lines = []
    for entry in re.split(r"\n\n+", srt_text.strip()):
        parts = entry.strip().split("\n")
        if len(parts) >= 3:
            lines.append(" ".join(parts[2:]))
    return "\n".join(lines)


def _get_video_info(url: str) -> dict:
    """获取视频标题和时长等元信息。"""
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True, "no_warnings": True,
            "js_runtimes": {"node": {}},
            "cookiesfrombrowser": ("chrome",),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)
        m, s = divmod(duration, 60)
        return {
            "title": info.get("title", "Unknown"),
            "duration": f"{m}分{s}秒",
        }
    except Exception:
        return {"title": "Unknown", "duration": "未知"}


def generate_summary(
    url: str,
    model: str,
    srt_path: str | Path | None = None,
) -> Path:
    """根据字幕内容生成视频摘要 Markdown。

    Args:
        url: YouTube 视频 URL
        model: LLM 模型名称
        srt_path: 指定字幕文件路径；为 None 时自动查找已有字幕

    Returns:
        生成的 Markdown 文件路径
    """
    from l2n.transcriber import extract_video_id
    video_id = extract_video_id(url)
    video_dir = SUBTITLE_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # 确定字幕来源
    if srt_path:
        srt_text = Path(srt_path).read_text(encoding="utf-8")
    else:
        # 按优先级查找已有字幕
        candidates = [
            video_dir / "subtitle_zh.srt",
            video_dir / "subtitle_zh_youtube.srt",
            video_dir / "subtitle_en.srt",
            video_dir / "subtitle_en_youtube.srt",
        ]
        srt_text = None
        for p in candidates:
            if p.exists():
                srt_text = p.read_text(encoding="utf-8")
                click.echo(f"   📄 使用字幕: {p}")
                break
        if srt_text is None:
            raise click.ClickException(
                "未找到字幕文件，请先用 --subtitle 生成字幕"
            )

    # 获取视频元信息
    click.echo("📝 正在生成视频摘要...")
    info = _get_video_info(url)
    plain_text = _srt_to_plain_text(srt_text)

    # 构建 prompt
    system_prompt = (PROMPTS_DIR / "summary_system.md").read_text(encoding="utf-8")
    user_message = (
        f"视频标题: {info['title']}\n"
        f"视频链接: {url}\n"
        f"视频时长: {info['duration']}\n\n"
        f"--- 字幕内容 ---\n{plain_text}"
    )

    raw = _call_translate_llm(system_prompt, user_message, model)

    # 清理可能的 ```markdown 包裹
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    out_path = video_dir / "summary.md"
    out_path.write_text(text, encoding="utf-8")
    click.echo(f"   ✅ 摘要已保存: {out_path}")
    return out_path


def _extract_titles_from_summary(summary_text: str) -> list[str]:
    """从摘要 Markdown 中提取建议中文标题列表。"""
    titles = []
    in_title_section = False
    for line in summary_text.split("\n"):
        if "建议中文标题" in line:
            in_title_section = True
            continue
        if in_title_section:
            if line.startswith("##"):
                break
            if line.strip().startswith("- "):
                title = line.strip().lstrip("- ").strip()
                if title:
                    titles.append(title)
    return titles if titles else ["视频封面"]


def generate_cover_images(
    url: str,
    model: str | None = None,
    num_images: int = 3,
) -> list[Path]:
    """基于视频摘要生成 B 站风格封面图。

    Args:
        url: YouTube 视频 URL
        model: 支持图片生成的 Gemini 模型
        num_images: 生成封面数量（默认 3）

    Returns:
        生成的封面图路径列表
    """
    from google import genai
    import httpx

    video_id = extract_video_id(url)
    video_dir = SUBTITLE_DIR / video_id

    # 读取摘要
    summary_path = video_dir / "summary.md"
    if not summary_path.exists():
        raise click.ClickException("未找到摘要文件，请先用 --summary 生成摘要")
    summary_text = summary_path.read_text(encoding="utf-8")
    titles = _extract_titles_from_summary(summary_text)

    # 加载 prompt 模板
    cover_prompt_template = (PROMPTS_DIR / "cover_image_system.md").read_text(encoding="utf-8")

    # 图片生成使用独立配置（GEMINI_IMAGE_*），与文本翻译的 GEMINI_* 分开
    if model is None:
        model = os.environ.get("GEMINI_IMAGE_MODEL")
        if not model:
            raise click.ClickException("未配置封面模型：请在 .env 中设置 GEMINI_IMAGE_MODEL")

    # 设置 Gemini 客户端（优先使用图片专用配置，回退到通用 Gemini 配置）
    api_key = os.environ.get("GEMINI_IMAGE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 GEMINI_IMAGE_API_KEY 或 GEMINI_API_KEY")
    base_url = os.environ.get("GEMINI_IMAGE_BASE_URL") or os.environ.get("GEMINI_BASE_URL")
    http_opts = {"httpxClient": httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))}
    if base_url:
        http_opts["base_url"] = base_url

    proxy_vars = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if key in os.environ:
            proxy_vars[key] = os.environ.pop(key)

    click.echo(f"🎨 正在生成 {num_images} 张 B 站风格封面 (模型: {model})...")
    generated_paths = []
    try:
        client = genai.Client(api_key=api_key, http_options=http_opts)

        # 查找参考人像照片（cover--.jpg）
        ref_image_path = video_dir / "cover--.jpg"
        ref_image_part = None
        if ref_image_path.exists():
            click.echo(f"   📷 使用参考照片: {ref_image_path}")
            ref_image_data = ref_image_path.read_bytes()
            ref_image_part = genai.types.Part.from_bytes(
                data=ref_image_data,
                mime_type="image/jpeg",
            )

        for i in range(num_images):
            title = titles[i] if i < len(titles) else titles[0]
            prompt_text = cover_prompt_template.format(
                title=title,
                summary=summary_text[:500],
                index=i + 1,
            )

            # 构建 contents：有参考照片时图文混合，否则纯文本
            if ref_image_part:
                contents = [ref_image_part, prompt_text]
            else:
                contents = prompt_text

            click.echo(f"   🖼️ 生成封面 {i + 1}/{num_images}：{title[:30]}...")
            try:
                response = None
                for attempt in range(1, 3):
                    try:
                        response = client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=genai.types.GenerateContentConfig(
                                response_modalities=["IMAGE", "TEXT"],
                            ),
                        )
                        break
                    except Exception as retry_err:
                        if attempt < 2:
                            click.echo(f"   ⚠️ 第 {attempt} 次失败，重试...")
                        else:
                            raise retry_err
                if response is None:
                    continue

                # 从响应中提取图片（支持 inline 数据和 URL 两种格式）
                image_saved = False
                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        # 方式1：inline 图片数据
                        if part.inline_data and part.inline_data.data:
                            mime = part.inline_data.mime_type or "image/png"
                            ext = ".png" if "png" in mime else ".jpg"
                            cover_path = video_dir / f"cover_{i + 1}{ext}"
                            cover_path.write_bytes(part.inline_data.data)
                            generated_paths.append(cover_path)
                            size_kb = cover_path.stat().st_size / 1024
                            click.echo(f"   ✅ 封面已保存: {cover_path} ({size_kb:.0f}KB)")
                            image_saved = True
                            break
                        # 方式2：中转平台返回 markdown 图片链接
                        if part.text:
                            url_match = re.search(r'!\[.*?\]\((https?://\S+)\)', part.text)
                            if url_match:
                                img_url = url_match.group(1)
                                try:
                                    img_resp = httpx.get(img_url, timeout=30)
                                    if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                                        ct = img_resp.headers.get("content-type", "")
                                        ext = ".png" if "png" in ct else ".jpg"
                                        cover_path = video_dir / f"cover_{i + 1}{ext}"
                                        cover_path.write_bytes(img_resp.content)
                                        generated_paths.append(cover_path)
                                        size_kb = cover_path.stat().st_size / 1024
                                        click.echo(f"   ✅ 封面已保存: {cover_path} ({size_kb:.0f}KB)")
                                        image_saved = True
                                        break
                                except Exception as dl_err:
                                    click.echo(f"   ⚠️ 图片下载失败: {dl_err}")

                if not image_saved:
                    click.echo(f"   ⚠️ 封面 {i + 1} 生成失败（未返回图片数据）")

            except Exception as e:
                click.echo(f"   ⚠️ 封面 {i + 1} 生成失败: {e}")

    finally:
        os.environ.update(proxy_vars)

    return generated_paths


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
    from l2n.transcriber import extract_video_id
    video_id = extract_video_id(url)
    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)

    if use_whisper:
        # Whisper 路径：下载音频 → 本地转录
        from l2n.whisper_transcriber import download_audio, transcribe_to_srt

        click.echo("🎬 Whisper 转录模式")
        video_dir = SUBTITLE_DIR / video_id

        # 检查英文字幕是否已存在，跳过 Whisper 转录
        en_existing = video_dir / "subtitle_en.srt"
        if en_existing.exists() and en_existing.stat().st_size > 0:
            click.echo(f"   ⏭️ 英文字幕已存在，跳过 Whisper 转录: {en_existing}")
            en_srt = en_existing.read_text(encoding="utf-8")
        else:
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

    # 输出目录：统一放在 SUBTITLE_DIR / video_id 下
    out_dir = SUBTITLE_DIR / video_id
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
    out_path.write_text(translated_srt, encoding="utf-8")
    click.echo(f"   📄 {label}字幕已保存: {out_path}")

    # 英文字幕校对：参考中文翻译修正 Whisper ASR 错误
    click.echo("🔍 正在校对英文字幕...")
    proofread_srt = proofread_en_srt(en_srt, translated_srt, model)
    if proofread_srt != en_srt:
        en_path.write_text(proofread_srt, encoding="utf-8")
        click.echo(f"   📄 英文字幕已更新: {en_path}")
        en_srt = proofread_srt

    # 翻译质量校验（纯代码，不消耗 token）
    en_entries = _parse_srt_entries(en_srt)
    zh_entries = _parse_srt_entries(translated_srt)
    check_failed = False
    if len(zh_entries) != len(en_entries):
        click.echo(f"   ⚠️ 校验: 条目数不一致（英文 {len(en_entries)}，中文 {len(zh_entries)}）")
        check_failed = True
    else:
        ts_mismatch = sum(1 for e, z in zip(en_entries, zh_entries) if e[1] != z[1])
        fallback = sum(1 for e, z in zip(en_entries, zh_entries) if e[2] == z[2])
        total = len(en_entries)
        fallback_ratio = fallback / total if total else 0
        if ts_mismatch == 0 and fallback == 0:
            click.echo(f"   ✅ 校验通过: {total} 条字幕全部对齐，无兜底")
        else:
            if ts_mismatch > 0:
                check_failed = True
                click.echo(f"   ⚠️ 校验: {ts_mismatch} 条时间戳不匹配")
            if fallback > 0 and fallback_ratio > 0.05:
                check_failed = True
                click.echo(f"   ⚠️ 校验: {fallback} 条未翻译（{fallback_ratio:.1%}，超过 5% 阈值）")
            elif fallback > 0:
                click.echo(f"   ℹ️ 校验: {fallback} 条使用英文兜底（{fallback_ratio:.1%}），在容忍范围内")

    if check_failed:
        fail_path = out_path.with_suffix(".srt.fail")
        out_path.rename(fail_path)
        click.echo(f"   ❌ 校验失败，已重命名为: {fail_path}")
        return fail_path

    return out_path
