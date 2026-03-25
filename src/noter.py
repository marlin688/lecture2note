import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import anthropic
import click
import httpx
from google import genai
from openai import OpenAI


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
MAX_CHUNK_CHARS = 8_000


def load_system_prompt() -> str:
    """从 prompts/note_system.md 读取 system prompt。"""
    prompt_path = PROMPTS_DIR / "note_system.md"
    return prompt_path.read_text(encoding="utf-8")


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def _is_gpt_model(model: str) -> bool:
    return model.startswith("gpt") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")


def _stream_with_progress_raw(base_url, api_key, model, system_prompt, messages, max_tokens=32000):
    """用原始 httpx SSE 流式调用 Anthropic API（绕过 SDK 的事件解析问题），返回 (文本, stop_reason)。"""
    collected = []
    char_count = 0
    start = time.time()
    first_token = False

    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    stop_event = threading.Event()

    def _waiting_spinner():
        i = 0
        while not stop_event.is_set():
            elapsed = int(time.time() - start)
            click.echo(f"\r   {spinner_frames[i % len(spinner_frames)]} 思考中... {elapsed}s", nl=False)
            i += 1
            stop_event.wait(0.1)

    spinner_thread = threading.Thread(target=_waiting_spinner, daemon=True)
    spinner_thread.start()

    stop_reason = None
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "system": system_prompt,
        "messages": messages,
        "stream": True,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0)) as http:
            with http.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    resp.read()
                    click.echo(f"\n   ⚠️ HTTP {resp.status_code}: {resp.text[:200]}")
                    stop_reason = "error"
                else:
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        evt_type = data.get("type", "")
                        if evt_type == "content_block_delta":
                            text_piece = data.get("delta", {}).get("text", "")
                            if text_piece:
                                if not first_token:
                                    first_token = True
                                    stop_event.set()
                                    spinner_thread.join()
                                    ttft = time.time() - start
                                    click.echo(f"\r   ✓ 首字符到达 ({ttft:.1f}s)")
                                collected.append(text_piece)
                                char_count += len(text_piece)
                                elapsed = int(time.time() - start)
                                click.echo(f"\r   ⏳ 已生成 {char_count} 字符 ({elapsed}s)", nl=False)
                        elif evt_type == "message_delta":
                            sr = data.get("delta", {}).get("stop_reason")
                            if sr:
                                stop_reason = sr
        if stop_reason is None:
            stop_reason = "end_turn" if collected else "error"
    except Exception as e:
        click.echo(f"\n   ⚠️ 流中断: {type(e).__name__}: {e}")
        stop_reason = "error" if not collected else "end_turn"
    finally:
        stop_event.set()
        spinner_thread.join()

    click.echo()
    total = time.time() - start
    click.echo(f"   📊 共 {char_count} 字符, 耗时 {total:.1f}s, 结束原因: {stop_reason}")

    return "".join(collected), stop_reason


def _call_non_stream(client, model, system_prompt, messages, max_tokens=16000):
    """非流式调用 Claude API（流式失败时的降级方案），返回 (文本, stop_reason)。"""
    start = time.time()
    stop_event = threading.Event()
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _waiting():
        i = 0
        while not stop_event.is_set():
            elapsed = int(time.time() - start)
            click.echo(f"\r   {spinner_frames[i % len(spinner_frames)]} 非流式等待中... {elapsed}s", nl=False)
            i += 1
            stop_event.wait(0.1)

    t = threading.Thread(target=_waiting, daemon=True)
    t.start()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.1,
            system=system_prompt,
            messages=messages,
        )
        stop_event.set()
        t.join()
        text = response.content[0].text if response.content else ""
        total = time.time() - start
        click.echo(f"\r   ✓ 非流式完成: {len(text)} 字符, {total:.1f}s, {response.stop_reason}")
        return text, response.stop_reason
    except Exception as e:
        stop_event.set()
        t.join()
        click.echo(f"\n   ⚠️ 非流式失败: {type(e).__name__}: {e}")
        return "", "error"


def call_claude(transcript: str, subject: str, model: str = DEFAULT_MODEL) -> str:
    """调用 Anthropic API 生成笔记，支持重试和截断自动续写。"""
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 ANTHROPIC_API_KEY 环境变量，请在 .env 文件或环境变量中配置")
    system_prompt = load_system_prompt()

    user_message = f"学科：{subject}\n\n以下是课堂转写文本：\n\n{transcript}"
    messages = [{"role": "user", "content": user_message}]

    max_retries = 3
    full_text = ""
    stop_reason = None
    for attempt in range(max_retries):
        full_text, stop_reason = _stream_with_progress_raw(
            base_url, api_key, model, system_prompt, messages
        )
        if full_text.strip() or stop_reason == "end_turn":
            break
        if attempt < max_retries - 1:
            wait = 5 * (attempt + 1)
            click.echo(f"   ⚠️ 未获得有效响应 ({stop_reason})，{wait}s 后重试 ({attempt + 2}/{max_retries})...")
            time.sleep(wait)

    # 需要续写的情况：max_tokens 截断、连接中断(error/None)且已有部分内容
    # 续写时只发送尾部片段作为上下文，避免 context 膨胀超出模型限制
    _CONTINUATION_TAIL_CHARS = 2000
    max_continuations = 5
    for i in range(max_continuations):
        if stop_reason == "end_turn":
            break
        if stop_reason in ("max_tokens", "error", None) and full_text.strip():
            click.echo(f"   🔄 响应未完成 ({stop_reason})，自动续写 ({i + 1}/{max_continuations})...")
            tail = full_text[-_CONTINUATION_TAIL_CHARS:]
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": tail},
                {"role": "user", "content": "你的 JSON 输出被截断了，请从断点处继续输出剩余的 JSON 内容（不要重复已输出的部分，直接接上）："},
            ]
            continuation, stop_reason = _stream_with_progress_raw(
                base_url, api_key, model, system_prompt, messages
            )
            full_text += continuation
        else:
            break

    return full_text


def _spinner(stop_event: threading.Event):
    """在后台显示旋转动画。"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    start = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - start)
        click.echo(f"\r   {frames[i % len(frames)]} 等待响应中... {elapsed}s", nl=False)
        i += 1
        stop_event.wait(0.1)
    click.echo()


def call_gemini(transcript: str, subject: str, model: str) -> str:
    """调用 Gemini API 生成笔记，返回原始文本响应。"""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise click.ClickException("未设置 GEMINI_API_KEY 环境变量，请在 .env 文件或环境变量中配置")
    base_url = os.environ.get("GEMINI_BASE_URL")
    http_opts = {}
    if base_url:
        http_opts["base_url"] = base_url
        # 第三方国内代理不需要走本地代理，直连即可
        http_opts["httpxClient"] = httpx.Client()
    client = genai.Client(
        api_key=api_key,
        http_options=http_opts or None,
    )
    system_prompt = load_system_prompt()
    user_message = f"学科：{subject}\n\n以下是课堂转写文本：\n\n{transcript}"

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=_spinner, args=(stop_event,), daemon=True)
    spinner_thread.start()
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=32000,
            ),
        )
    finally:
        stop_event.set()
        spinner_thread.join()

    return response.text


def call_gpt(transcript: str, subject: str, model: str) -> str:
    """调用 OpenAI 兼容 API 生成笔记，流式输出。支持 GPT 和通过代理的 Claude 模型。"""
    if _is_gpt_model(model):
        base_url = os.environ.get("GPT_BASE_URL")
        api_key = os.environ.get("GPT_API_KEY", "")
    else:
        # Claude 等模型走代理的 OpenAI 兼容接口
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("GPT_BASE_URL")
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GPT_API_KEY", "")
    if not api_key:
        key_name = "GPT_API_KEY" if _is_gpt_model(model) else "GPT_API_KEY 或 ANTHROPIC_API_KEY"
        raise click.ClickException(f"未设置 {key_name} 环境变量，请在 .env 文件或环境变量中配置")

    client_kwargs = {}
    if base_url:
        # 代理地址需要加 /v1 后缀
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        client_kwargs["base_url"] = base_url
        client_kwargs["http_client"] = httpx.Client()

    client = OpenAI(api_key=api_key, timeout=httpx.Timeout(600.0, connect=60.0), **client_kwargs)
    system_prompt = load_system_prompt()
    user_message = f"学科：{subject}\n\n以下是课堂转写文本：\n\n{transcript}"

    collected = []
    char_count = 0
    start = time.time()
    first_token = False
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    stop_event = threading.Event()

    def _waiting_spinner():
        i = 0
        while not stop_event.is_set():
            elapsed = int(time.time() - start)
            click.echo(f"\r   {spinner_frames[i % len(spinner_frames)]} 思考中... {elapsed}s", nl=False)
            i += 1
            stop_event.wait(0.1)

    spinner_thread = threading.Thread(target=_waiting_spinner, daemon=True)
    spinner_thread.start()

    finish_reason = None
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=32000,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if not first_token:
                    first_token = True
                    stop_event.set()
                    spinner_thread.join()
                    ttft = time.time() - start
                    click.echo(f"\r   ✓ 首字符到达 ({ttft:.1f}s)")
                collected.append(delta.content)
                char_count += len(delta.content)
                elapsed = int(time.time() - start)
                click.echo(f"\r   ⏳ 已生成 {char_count} 字符 ({elapsed}s)", nl=False)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
    except Exception as e:
        click.echo(f"\n   ⚠️ 流中断: {type(e).__name__}: {e}")
        finish_reason = "error"
    finally:
        stop_event.set()
        spinner_thread.join()

    click.echo()
    total = time.time() - start
    click.echo(f"   📊 共 {char_count} 字符, 耗时 {total:.1f}s, 结束原因: {finish_reason}")

    full_text = "".join(collected)

    # 如果因 max_tokens 截断，自动续写
    # 续写时只发送尾部片段作为上下文，避免 context 膨胀超出模型限制
    _CONTINUATION_TAIL_CHARS = 2000
    if finish_reason == "length" and full_text.strip():
        max_continuations = 5
        for i in range(max_continuations):
            click.echo(f"   🔄 响应未完成 (length)，自动续写 ({i + 1}/{max_continuations})...")
            cont_collected = []
            cont_count = 0
            tail = full_text[-_CONTINUATION_TAIL_CHARS:]
            try:
                cont_stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": tail},
                        {"role": "user", "content": "你的 JSON 输出被截断了，请从断点处继续输出剩余的 JSON 内容（不要重复已输出的部分，直接接上）："},
                    ],
                    temperature=0.1,
                    max_tokens=32000,
                    stream=True,
                )
                for chunk in cont_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        cont_collected.append(delta.content)
                        cont_count += len(delta.content)
                        click.echo(f"\r   ⏳ 续写 {cont_count} 字符", nl=False)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                click.echo()
                full_text += "".join(cont_collected)
                if finish_reason != "length":
                    break
            except Exception as e:
                click.echo(f"\n   ⚠️ 续写失败: {type(e).__name__}: {e}")
                break

    return full_text


def call_llm(transcript: str, subject: str, model: str = DEFAULT_MODEL) -> str:
    """根据模型名称自动选择 Claude、Gemini 或 GPT API。"""
    if _is_gemini_model(model):
        return call_gemini(transcript, subject, model)
    if _is_gpt_model(model):
        return call_gpt(transcript, subject, model)
    return call_claude(transcript, subject, model)


def _fix_json_escapes(text: str) -> str:
    """修复 JSON 中非法的反斜杠转义（如 LaTeX 的 \\mathcal、\\alpha 等）。"""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def _fix_unescaped_quotes(text: str) -> str:
    """修复 JSON 字符串值中未转义的双引号。

    迭代策略：反复尝试 json.loads，在报错位置将非结构性双引号转义。
    每轮基于当前字符串重新操作，避免替换后索引错位。
    """
    current = text
    max_attempts = 50

    for _ in range(max_attempts):
        try:
            json.loads(current)
            return current
        except json.JSONDecodeError as e:
            pos = e.pos
            if pos is None or pos >= len(current):
                return current
            # 向前找到导致问题的引号（错误通常在引号后一个字符被报告）
            search_pos = pos - 1
            fixed = False
            while search_pos >= 0:
                if current[search_pos] == '"' and (search_pos == 0 or current[search_pos - 1] != '\\'):
                    prev_char = current[search_pos - 1] if search_pos > 0 else ''
                    # JSON 结构引号前面通常是 { [ , : 空白 或字符串开头
                    if prev_char not in '{[,:  \t\n\r\\':
                        current = current[:search_pos] + '\\"' + current[search_pos + 1:]
                        fixed = True
                        break
                search_pos -= 1
            if not fixed:
                return current  # 找不到可修复的引号

    return current


def parse_response(raw_text: str) -> dict:
    """解析模型返回的 JSON，处理代码块包裹、非法转义等异常情况。"""
    text = raw_text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 修复字符串内未转义的双引号后重试
    try:
        return json.loads(_fix_unescaped_quotes(text))
    except json.JSONDecodeError:
        pass

    # 修复 LaTeX 等导致的非法 JSON 转义后重试
    try:
        return json.loads(_fix_json_escapes(text))
    except json.JSONDecodeError:
        pass

    # 同时修复引号和转义
    try:
        return json.loads(_fix_json_escapes(_fix_unescaped_quotes(text)))
    except json.JSONDecodeError:
        pass

    # 去掉 ```json 开头和 ``` 结尾（JSON 内容可能也含反引号，不能用正则贪婪匹配）
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
        for fixer in [lambda t: t, _fix_unescaped_quotes, _fix_json_escapes,
                      lambda t: _fix_json_escapes(_fix_unescaped_quotes(t))]:
            try:
                return json.loads(fixer(text))
            except json.JSONDecodeError:
                pass

    # 尝试找到第一个 { 和最后一个 } 之间的内容
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        for fixer in [lambda t: t, _fix_unescaped_quotes, _fix_json_escapes,
                      lambda t: _fix_json_escapes(_fix_unescaped_quotes(t))]:
            try:
                return json.loads(fixer(candidate))
            except json.JSONDecodeError:
                pass

    # 全部失败，判断是否为有意义的文本内容（非 JSON 的笔记）
    if len(text) > 100:
        # 模型直接输出了 Markdown 笔记而非 JSON，标记为原始 Markdown 直出
        return {"_raw_markdown": raw_text}

    return {
        "title": "笔记（解析失败）",
        "subject": "未知",
        "summary": "无法解析模型返回的 JSON，以下为原始输出。",
        "sections": [
            {
                "heading": "原始输出",
                "content": raw_text,
                "key_points": [],
                "teacher_emphasis": None,
            }
        ],
        "key_terms": [],
        "review_questions": [],
    }


OVERLAP_CHARS = 200

def split_transcript(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """智能分片，优先在双换行（段落边界）处切分，相邻分片之间保留重叠区域避免边界处丢失内容。"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        # 在 max_chars 范围内找最后一个双换行
        search_region = remaining[:max_chars]
        split_pos = search_region.rfind("\n\n")

        if split_pos == -1 or split_pos < max_chars // 2:
            # 没有双换行或太靠前，尝试单换行
            split_pos = search_region.rfind("\n")

        if split_pos == -1 or split_pos < max_chars // 2:
            # 都没有，直接硬切
            split_pos = max_chars

        chunks.append(remaining[:split_pos].strip())
        # 回退 overlap 个字符，让下一片与当前片有重叠
        overlap_start = max(0, split_pos - overlap)
        remaining = remaining[overlap_start:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _extract_chinese_chars(text: str) -> set[str]:
    """提取文本中的所有中文字符集合。"""
    return set(re.findall(r'[\u4e00-\u9fff]', text))


def _jaccard_similarity(set1: set, set2: set) -> float:
    """计算两个集合的 Jaccard 相似度。"""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _sections_similar(sec1: dict, sec2: dict, threshold: float = 0.45) -> bool:
    """判断两个 section 是否内容重复（基于标题+要点的中文字符重叠）。"""
    # 用标题 + key_points 组合判断
    text1 = sec1.get("heading", "") + " ".join(sec1.get("key_points", []))
    text2 = sec2.get("heading", "") + " ".join(sec2.get("key_points", []))
    chars1 = _extract_chinese_chars(text1)
    chars2 = _extract_chinese_chars(text2)
    return _jaccard_similarity(chars1, chars2) > threshold


def _merge_two_sections(sec1: dict, sec2: dict) -> dict:
    """合并两个重复 section，保留更长的 content，合并 key_points。"""
    # 保留 content 更长的作为主体
    if len(sec2.get("content", "")) > len(sec1.get("content", "")):
        base, extra = sec2, sec1
    else:
        base, extra = sec1, sec2

    merged_sec = dict(base)
    # 合并 key_points 去重
    existing_points = set(base.get("key_points", []))
    merged_points = list(base.get("key_points", []))
    for p in extra.get("key_points", []):
        if p not in existing_points:
            merged_points.append(p)
            existing_points.add(p)
    merged_sec["key_points"] = merged_points

    # 保留非空的 teacher_emphasis
    if not merged_sec.get("teacher_emphasis") and extra.get("teacher_emphasis"):
        merged_sec["teacher_emphasis"] = extra["teacher_emphasis"]

    return merged_sec


def _normalize_term_key(term_str: str) -> str:
    """提取术语的中文部分用于去重：'低秩分解（Low-rank Decomposition）' -> '低秩分解'"""
    for sep in ('（', '('):
        idx = term_str.find(sep)
        if idx > 0:
            return term_str[:idx].strip()
    return term_str.strip()


def _merge_notes(notes_list: list[dict]) -> dict:
    """合并多个分片笔记：去重 sections、去重术语、合并复习题。"""
    if len(notes_list) == 1:
        return notes_list[0]

    # 从各分片中取第一个非空的 title/subject
    def _first_non_empty(key):
        for n in notes_list:
            val = n.get(key, "")
            if val:
                return val
        return ""

    # 合并所有分片的 summary，每片取第一句避免过长
    all_summaries = []
    for n in notes_list:
        s = n.get("summary", "").strip()
        if s:
            # 取第一句（按句号/分号切分）
            first_sent = re.split(r'[。；]', s)[0]
            if first_sent:
                all_summaries.append(first_sent.strip() + "。" if not first_sent.endswith("。") else first_sent.strip())
    combined_summary = "".join(all_summaries) if all_summaries else ""
    # 如果合并后仍然过长，截断到合理长度
    if len(combined_summary) > 300:
        # 找到 300 字符内最后一个句号
        cut = combined_summary[:300].rfind("。")
        if cut > 0:
            combined_summary = combined_summary[:cut + 1]

    merged = {
        "title": _first_non_empty("title"),
        "subject": _first_non_empty("subject"),
        "summary": combined_summary,
        "sections": [],
        "key_terms": [],
        "review_questions": [],
    }

    # 收集所有 sections 并去重
    all_sections = []
    for notes in notes_list:
        all_sections.extend(notes.get("sections", []))

    deduped_sections = []
    for sec in all_sections:
        merged_with_existing = False
        for i, existing in enumerate(deduped_sections):
            if _sections_similar(sec, existing):
                deduped_sections[i] = _merge_two_sections(existing, sec)
                merged_with_existing = True
                break
        if not merged_with_existing:
            deduped_sections.append(sec)
    merged["sections"] = deduped_sections

    # 术语去重（按中文部分归一化）
    seen_terms = set()
    for notes in notes_list:
        for term in notes.get("key_terms", []):
            term_name = term.get("term", "")
            norm_key = _normalize_term_key(term_name)
            if norm_key not in seen_terms:
                seen_terms.add(norm_key)
                merged["key_terms"].append(term)

    # 复习题去重（基于中文字符相似度）
    all_questions = []
    for notes in notes_list:
        all_questions.extend(notes.get("review_questions", []))

    unique_questions = []
    for q in all_questions:
        q_chars = _extract_chinese_chars(q)
        is_dup = False
        for existing in unique_questions:
            existing_chars = _extract_chinese_chars(existing)
            if _jaccard_similarity(q_chars, existing_chars) > 0.6:
                is_dup = True
                break
        if not is_dup:
            unique_questions.append(q)
    merged["review_questions"] = unique_questions

    return merged


def _count_sections(parsed: dict) -> int:
    """统计解析结果中的 section 数量。"""
    if "_raw_markdown" in parsed:
        # 按 ## 标题数量估算 section 数
        md = parsed["_raw_markdown"]
        headings = [line for line in md.split("\n") if line.startswith("## ")]
        return max(1, len(headings))
    return len(parsed.get("sections", []))


# 每 1000 字符的原文至少应产出的 section 数（低于此值触发重试）
_MIN_SECTIONS_PER_1K = 0.5


def process_transcript(
    transcript: str, subject: str, model: str = DEFAULT_MODEL
) -> dict:
    """处理转写文本：短文本直接调用，长文本分片后合并。"""
    chunks = split_transcript(transcript)

    if len(chunks) == 1:
        raw = call_llm(chunks[0], subject, model)
        return parse_response(raw)

    notes_list = []
    for i, chunk in enumerate(chunks, 1):
        click.echo(f"📦 处理分片 [{i}/{len(chunks)}]...")
        hint = (
            f"（这是第 {i}/{len(chunks)} 部分，请只整理本部分内容。"
            f"注意：不要遗漏本部分中的任何知识点，即使转写文本噪声较大也要尽力提取。"
            f"特别注意：1）每个具名方法/网络/算法必须有独立描述；"
            f"2）老师对多种方案的对比分析必须完整保留；"
            f"3）训练技巧（如 batch 策略、优化器选择等）必须独立成节；"
            f"4）老师的观点性总结和历史趋势讨论不可省略。）\n\n"
        )
        raw = call_llm(hint + chunk, subject, model)
        parsed = parse_response(raw)

        # 产出过少时重试一次，并合并两次结果以最大化覆盖率
        min_sections = max(1, int(len(chunk) / 1000 * _MIN_SECTIONS_PER_1K))
        if _count_sections(parsed) < min_sections:
            click.echo(f"   ⚠️ 分片 {i} 产出 {_count_sections(parsed)} 个章节，低于预期 {min_sections} 个，重试中...")
            retry_hint = (
                f"（这是第 {i}/{len(chunks)} 部分，请只整理本部分内容。"
                f"重要提示：请仔细逐段阅读以下转写文本，尽可能多地提取知识点，"
                f"每个不同的主题都应生成一个独立的 section。即使文本噪声很大，也请尽力识别。）\n\n"
            )
            raw = call_llm(retry_hint + chunk, subject, model)
            retry_parsed = parse_response(raw)
            retry_count = _count_sections(retry_parsed)
            orig_count = _count_sections(parsed)

            if "_raw_markdown" in parsed or "_raw_markdown" in retry_parsed:
                # 有 raw_markdown 结果时取 section 数更多的那个
                if retry_count > orig_count:
                    parsed = retry_parsed
                click.echo(f"   ✓ 取较优结果: {_count_sections(parsed)} 个章节")
            else:
                # 两个都是 JSON 结构化结果时合并（去重会在 _merge_notes 中处理）
                merged_chunk = _merge_notes([parsed, retry_parsed])
                click.echo(f"   ✓ 合并后共 {_count_sections(merged_chunk)} 个章节")
                parsed = merged_chunk

        notes_list.append(parsed)

    return _merge_notes(notes_list)
