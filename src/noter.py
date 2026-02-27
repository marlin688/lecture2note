import json
import re
from pathlib import Path

import anthropic


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
MAX_CHUNK_CHARS = 80_000


def load_system_prompt() -> str:
    """从 prompts/note_system.md 读取 system prompt。"""
    prompt_path = PROMPTS_DIR / "note_system.md"
    return prompt_path.read_text(encoding="utf-8")


def call_claude(transcript: str, subject: str, model: str = DEFAULT_MODEL) -> str:
    """调用 Anthropic API 生成笔记，返回原始文本响应。"""
    client = anthropic.Anthropic()
    system_prompt = load_system_prompt()

    user_message = f"学科：{subject}\n\n以下是课堂转写文本：\n\n{transcript}"

    with client.messages.stream(
        model=model,
        max_tokens=16000,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()

    return response.content[0].text


def parse_response(raw_text: str) -> dict:
    """解析 Claude 返回的 JSON，处理 ```json``` 包裹和异常情况。"""
    text = raw_text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
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
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 和最后一个 } 之间的内容
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    # 全部失败，降级输出
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


def split_transcript(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """智能分片，优先在双换行（段落边界）处切分。"""
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
        remaining = remaining[split_pos:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _merge_notes(notes_list: list[dict]) -> dict:
    """合并多个分片笔记：拼接 sections、去重术语、合并复习题。"""
    if len(notes_list) == 1:
        return notes_list[0]

    merged = {
        "title": notes_list[0].get("title", ""),
        "subject": notes_list[0].get("subject", ""),
        "summary": notes_list[0].get("summary", ""),
        "sections": [],
        "key_terms": [],
        "review_questions": [],
    }

    seen_terms = set()

    for notes in notes_list:
        merged["sections"].extend(notes.get("sections", []))

        for term in notes.get("key_terms", []):
            term_name = term.get("term", "")
            if term_name not in seen_terms:
                seen_terms.add(term_name)
                merged["key_terms"].append(term)

        merged["review_questions"].extend(notes.get("review_questions", []))

    # 复习题去重
    seen_questions = set()
    unique_questions = []
    for q in merged["review_questions"]:
        if q not in seen_questions:
            seen_questions.add(q)
            unique_questions.append(q)
    merged["review_questions"] = unique_questions

    return merged


def process_transcript(
    transcript: str, subject: str, model: str = DEFAULT_MODEL
) -> dict:
    """处理转写文本：短文本直接调用，长文本分片后合并。"""
    chunks = split_transcript(transcript)

    if len(chunks) == 1:
        raw = call_claude(chunks[0], subject, model)
        return parse_response(raw)

    notes_list = []
    for i, chunk in enumerate(chunks, 1):
        hint = f"（这是第 {i}/{len(chunks)} 部分，请只整理本部分内容）\n\n"
        raw = call_claude(hint + chunk, subject, model)
        notes_list.append(parse_response(raw))

    return _merge_notes(notes_list)
