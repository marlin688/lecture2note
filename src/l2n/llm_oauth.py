"""通过 auth.json 调用 OpenAI 模型（OAuth / ChatGPT Codex Responses）。

支持两种 auth.json schema：
1. Nested OAuth：{"openai": {"type":"oauth","refresh":..,"access":..,"expires":..,"accountId":..}}
   会自动 refresh access token。
2. 扁平 Codex（Claude Code / codex CLI 格式）：
   {"auth_mode":"chatgpt","OPENAI_API_KEY":..,"tokens":{"access_token":..,"refresh_token":..,"account_id":..}}

根据 access token 的 JWT scope 自动选择端点：
- 含 api.* scope → https://api.openai.com/v1/chat/completions
- 否则 → https://chatgpt.com/backend-api/codex/responses（ChatGPT 后端）
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import click
import httpx


_CODEX_BASE_URL_DEFAULT = "https://chatgpt.com/backend-api/codex"
_OPENAI_API_BASE = "https://api.openai.com/v1"


# ── auth.json 文件定位 ─────────────────────────────────────


def _resolve_auth_file() -> str | None:
    """auth.json 查找顺序：env vars > 项目根 > ~/.codex/auth.json。"""
    for env_var in ("OPENAI_AUTH_FILE", "CODEX_AUTH_FILE", "CODEX_OPENAI_AUTH_FILE"):
        candidate = os.environ.get(env_var, "").strip()
        if candidate:
            return candidate

    local_auth = Path("auth.json")
    if local_auth.exists():
        return str(local_auth.resolve())

    default_path = Path.home() / ".codex" / "auth.json"
    if default_path.exists():
        return str(default_path)
    return None


# ── JWT scope 解析 ────────────────────────────────────────


def _read_oauth_scopes(access_token: str) -> set[str] | None:
    """从 JWT access token 中读取 scp（scope）列表。"""
    parts = access_token.split(".")
    if len(parts) < 2:
        return None
    payload_seg = parts[1]
    padding = "=" * (-len(payload_seg) % 4)
    try:
        payload_obj = json.loads(base64.urlsafe_b64decode(payload_seg + padding).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload_obj, dict):
        return None
    scp = payload_obj.get("scp")
    if not isinstance(scp, list):
        return None
    return {str(item).strip() for item in scp if str(item).strip()}


def _has_openai_api_scope(access_token: str) -> bool:
    """access token 是否具备调用 OpenAI 官方 API 的权限。"""
    scopes = _read_oauth_scopes(access_token)
    if scopes is None:
        # 解析不出 scope 时保守地认为有权限（让 API 自己返 401）
        return True
    return any(s == "api" or s.startswith("api.") and not s.startswith("api.connectors")
               for s in scopes)


# ── 凭证加载 ──────────────────────────────────────────────


def _try_load_nested_oauth(auth_file: str):
    """尝试按 nested OAuth schema 加载（带 token refresh）。

    返回 (access_token, account_id) 或 None。
    """
    try:
        from l2n.openai_auth import (
            OpenAIAuthError,
            OpenAIAuthRefreshError,
            OpenAIAuthTokenFileError,
            load_openai_auth_session,
        )
    except ImportError:
        return None

    try:
        with httpx.Client(timeout=30.0) as refresh_client:
            session = load_openai_auth_session(auth_file, http_client=refresh_client)
        return session.access_token, session.account_id
    except OpenAIAuthTokenFileError:
        # schema 不匹配（扁平格式），由调用方走 fallback
        return None
    except (OpenAIAuthRefreshError, OpenAIAuthError) as e:
        raise click.ClickException(f"OpenAI OAuth 凭证不可用 ({auth_file}): {e}") from e


def _try_load_flat_codex(auth_file: str):
    """尝试按扁平 Codex schema 加载。

    返回 dict 形如 {"api_key": str | None, "access_token": str | None, "account_id": str | None}。
    """
    try:
        data = json.loads(Path(auth_file).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    api_key = (data.get("OPENAI_API_KEY") or "").strip() or None
    tokens = data.get("tokens") or {}
    if not isinstance(tokens, dict):
        tokens = {}
    access_token = (tokens.get("access_token") or "").strip() or None
    account_id = (tokens.get("account_id") or "").strip() or None
    if not (api_key or access_token):
        return None
    return {"api_key": api_key, "access_token": access_token, "account_id": account_id}


# ── 代理 ──────────────────────────────────────────────────


def _read_proxy_url() -> str | None:
    """从常见 env vars 读取代理 URL。"""
    for key in ("OPENAI_PROXY",
                "https_proxy", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY",
                "http_proxy", "HTTP_PROXY"):
        val = os.environ.get(key, "").strip()
        if val:
            if val.rstrip("/").endswith("~"):
                val = val.rstrip("~")
            if "://" not in val:
                val = f"http://{val}"
            return val
    return None


def _make_http_client(timeout: float = 600.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=60.0),
        proxy=_read_proxy_url(),
        trust_env=False,
    )


# ── 端点实现 ──────────────────────────────────────────────


def _call_openai_chat(system_prompt: str, user_message: str, model: str,
                      api_key: str, account_id: str | None,
                      temperature: float, max_tokens: int) -> str:
    """走标准 OpenAI /v1/chat/completions（适用于 sk- key 或具备 api.* scope 的 OAuth token）。"""
    from openai import OpenAI

    default_headers = {}
    if account_id:
        default_headers["ChatGPT-Account-ID"] = account_id

    client = OpenAI(
        api_key=api_key,
        base_url=_OPENAI_API_BASE,
        default_headers=default_headers or None,
        http_client=_make_http_client(),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = response.choices[0].message.content or ""
    if not result:
        raise RuntimeError("OpenAI 返回空响应")
    return result


def _parse_codex_sse(raw_text: str) -> str:
    """解析 Codex /responses SSE 流，提取 message 内容。

    优先策略：从 response.completed 事件取最终 response.output 数组，遍历 message item
    提取 output_text/text 内容；fallback 再从 response.output_item.done 事件累积。
    """
    output_items: list[dict] = []
    completed_response: dict | None = None

    for chunk in raw_text.split("\n\n"):
        event_name = ""
        data_lines: list[str] = []
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        if not event_name or not data_lines:
            continue
        try:
            event = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        if event_name == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif event_name == "response.completed":
            resp = event.get("response")
            if isinstance(resp, dict):
                completed_response = resp

    # 从最终 response.output 提取
    final_items = output_items
    if completed_response is not None:
        out = completed_response.get("output")
        if isinstance(out, list) and out:
            final_items = [x for x in out if isinstance(x, dict)]

    content_parts: list[str] = []
    for item in final_items:
        if str(item.get("type", "")) != "message":
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")) in {"output_text", "text", "input_text"}:
                text = block.get("text")
                if isinstance(text, str) and text:
                    content_parts.append(text)

    return "\n".join(content_parts)


def _call_codex_responses(system_prompt: str, user_message: str, model: str,
                          access_token: str, account_id: str | None) -> str:
    """走 ChatGPT Codex /responses 端点（SSE 流式，适用于无 api.* scope 的 OAuth token）。

    Header / payload / SSE 解析方式参考 mako (codex_oauth) 实现：
    - 双发 X-ChatGPT-Account-ID + ChatGPT-Account-Id（后者注意 Id 小写）
    - payload 精简（不带 tool_choice / parallel_tool_calls）
    - 读完整 SSE 流再按 response.completed 取最终 output items
    """
    codex_base = (os.environ.get("OPENAI_CODEX_BASE_URL", "").strip()
                  or _CODEX_BASE_URL_DEFAULT).rstrip("/")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if account_id:
        headers["X-ChatGPT-Account-ID"] = account_id
        headers["ChatGPT-Account-Id"] = account_id  # 注意 Id 小写

    payload = {
        "model": model,
        "instructions": system_prompt,
        "store": False,
        "stream": True,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_message}],
        }],
    }

    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
        text = str(exc).lower()
        return any(needle in text for needle in (
            "unexpected_eof_while_reading", "ssl", "eof occurred",
            "connection reset", "stream disconnected",
        ))

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            with _make_http_client(timeout=180.0) as client:
                with client.stream(
                    "POST",
                    f"{codex_base}/responses",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", errors="ignore")
                        raise click.ClickException(
                            f"ChatGPT Codex 调用失败 ({response.status_code}): {body[:500]}"
                        )
                    raw_bytes = response.read()

            raw_text = raw_bytes.decode("utf-8", errors="replace")
            result = _parse_codex_sse(raw_text).strip()
            if not result:
                raise RuntimeError(f"ChatGPT Codex 返回空响应；SSE 前 300 字: {raw_text[:300]!r}")
            return result
        except Exception as exc:
            last_exc = exc
            if attempt >= 3 or not _retryable(exc):
                break
            wait_s = 0.8 * attempt
            click.echo(f"   ⚠️ ChatGPT Codex 连接波动，{wait_s:.1f}s 后重试 ({attempt}/3)...")
            time.sleep(wait_s)

    if last_exc:
        raise last_exc
    raise RuntimeError("ChatGPT Codex 调用失败")


# ── 统一入口 ──────────────────────────────────────────────


def call_translate_via_oauth(system_prompt: str, user_message: str, model: str,
                             temperature: float = 0.1, max_tokens: int = 16000) -> str:
    """通过 auth.json 调用 OpenAI 模型（仅限 GPT 系列）。

    自动按 schema 与 scope 选择端点：
    - nested OAuth schema (带 refresh) > 扁平 Codex schema
    - API key 字段 / api.* scope → /v1/chat/completions
    - api.connectors.* scope (无 api.*) → /codex/responses
    """
    auth_file = _resolve_auth_file()
    if not auth_file:
        raise click.ClickException(
            "未找到 auth.json（设置 OPENAI_AUTH_FILE 或放在项目根 / ~/.codex/auth.json）"
        )

    # 1. nested OAuth schema (带自动 refresh)
    nested = _try_load_nested_oauth(auth_file)
    if nested:
        access_token, account_id = nested
        if _has_openai_api_scope(access_token):
            return _call_openai_chat(
                system_prompt, user_message, model,
                access_token, account_id, temperature, max_tokens,
            )
        return _call_codex_responses(
            system_prompt, user_message, model,
            access_token, account_id,
        )

    # 2. 扁平 Codex schema
    flat = _try_load_flat_codex(auth_file)
    if flat:
        # 2a. 直接 API key
        if flat["api_key"]:
            return _call_openai_chat(
                system_prompt, user_message, model,
                flat["api_key"], flat["account_id"], temperature, max_tokens,
            )
        # 2b. OAuth access token：按 scope 路由
        if flat["access_token"]:
            access_token = flat["access_token"]
            if _has_openai_api_scope(access_token):
                return _call_openai_chat(
                    system_prompt, user_message, model,
                    access_token, flat["account_id"], temperature, max_tokens,
                )
            return _call_codex_responses(
                system_prompt, user_message, model,
                access_token, flat["account_id"],
            )

    raise click.ClickException(f"auth.json 无可用凭证: {auth_file}")


__all__ = ["call_translate_via_oauth"]
