"""OpenAI OAuth token file loader with proactive refresh.

Compatible token file shape:
{
  "openai": {
    "type": "oauth",
    "refresh": "...",
    "access": "...",
    "expires": 1776932871638,
    "accountId": "..."
  }
}
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import threading
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_JWT_EXPIRATION_CLAIM = "exp"
_JWT_ACCOUNT_ID_CLAIM = "https://api.openai.com/auth.chatgpt_account_id"
_REFRESH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_REFRESH_GRANT_TYPE = "refresh_token"
_REFRESH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_PROACTIVE_REFRESH_WINDOW_MS = 8_000
_TOKEN_FILE_MODE = 0o600

_TOKEN_FILE_LOCKS: dict[str, threading.Lock] = {}
_TOKEN_FILE_LOCKS_GUARD = threading.Lock()


class OpenAIAuthError(RuntimeError):
    """Base error for OpenAI OAuth token file handling."""


class OpenAIAuthTokenFileError(OpenAIAuthError):
    """Token file read/parse/validate error."""


class OpenAIAuthRefreshError(OpenAIAuthError):
    """Refresh call failed."""


class _RefreshHTTPResponse(Protocol):
    status_code: int

    @property
    def text(self) -> str: ...

    def json(self) -> object: ...


class _RefreshHTTPClient(Protocol):
    def post(self, url: str, *, json: object) -> _RefreshHTTPResponse: ...


class OpenAIOAuthToken(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    type: str = Field(min_length=1)
    refresh: str = Field(min_length=1)
    access: str = Field(min_length=1)
    expires: int = Field(ge=0)
    accountId: str | None = Field(default=None, min_length=1)


class OpenAIAuthTokenFile(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    openai: OpenAIOAuthToken


class OpenAIAuthSession(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    access_token: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    expires_at_ms: int = Field(ge=0)


def _token_file_lock(path: Path) -> threading.Lock:
    lock_key = str(path.expanduser().resolve(strict=False))
    with _TOKEN_FILE_LOCKS_GUARD:
        lock = _TOKEN_FILE_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _TOKEN_FILE_LOCKS[lock_key] = lock
        return lock


def _current_time_ms() -> int:
    return int(time.time() * 1000)


def _create_default_http_client():
    import httpx

    return httpx.Client(timeout=10.0)


def _read_optional_string_claim(
    payload: Mapping[str, object], claim_name: str, *, path: Path,
) -> str | None:
    claim_value = payload.get(claim_name)
    if claim_value is None:
        return None
    if not isinstance(claim_value, str):
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}")
    stripped = claim_value.strip()
    return stripped if stripped else None


def _read_optional_expiration_ms(
    payload: Mapping[str, object], *, path: Path,
) -> int | None:
    expiration_value = payload.get(_JWT_EXPIRATION_CLAIM)
    if expiration_value is None:
        return None
    if isinstance(expiration_value, bool):
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}")

    if isinstance(expiration_value, int):
        expiration_seconds = expiration_value
    elif isinstance(expiration_value, float) and expiration_value.is_integer():
        expiration_seconds = int(expiration_value)
    else:
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}")

    if expiration_seconds < 0:
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}")
    return expiration_seconds * 1000


def _read_jwt_claims(access_token: str, *, path: Path) -> tuple[str | None, int | None]:
    token_parts = access_token.split(".")
    if len(token_parts) < 2:
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}")

    payload_segment = token_parts[1]
    padding = "=" * (-len(payload_segment) % 4)

    try:
        decoded_payload = base64.urlsafe_b64decode(payload_segment + padding)
        payload_obj = cast(object, json.loads(decoded_payload.decode("utf-8")))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}") from exc

    if not isinstance(payload_obj, dict):
        raise OpenAIAuthTokenFileError(f"Invalid access token payload: {path}")
    payload = cast(dict[str, object], payload_obj)

    account_id = _read_optional_string_claim(payload, _JWT_ACCOUNT_ID_CLAIM, path=path)
    expires_at_ms = _read_optional_expiration_ms(payload, path=path)
    return account_id, expires_at_ms


def _resolve_auth_session(token_file: OpenAIAuthTokenFile, *, path: Path) -> OpenAIAuthSession:
    account_from_token, exp_from_token = _read_jwt_claims(token_file.openai.access, path=path)
    account_id = token_file.openai.accountId or account_from_token
    if account_id is None:
        raise OpenAIAuthTokenFileError(
            f"OpenAI auth token file is missing accountId claim: {path}"
        )

    expires_at_ms = exp_from_token if exp_from_token is not None else token_file.openai.expires
    return OpenAIAuthSession(
        access_token=token_file.openai.access,
        account_id=account_id,
        expires_at_ms=expires_at_ms,
    )


def _is_refresh_required(
    session: OpenAIAuthSession, *, now_ms: int, refresh_window_ms: int,
) -> bool:
    return session.expires_at_ms <= now_ms + refresh_window_ms


def _read_response_json(response: _RefreshHTTPResponse) -> dict[str, object]:
    try:
        payload_obj = response.json()
    except Exception as exc:
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.") from exc

    if not isinstance(payload_obj, dict):
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.")
    return cast(dict[str, object], payload_obj)


def _read_optional_json_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.")
    stripped = value.strip()
    return stripped if stripped else None


def _refresh_failure_message(payload: Mapping[str, object]) -> str:
    error_code = payload.get("error")
    if isinstance(error_code, str):
        if "expired" in error_code:
            return "OpenAI OAuth refresh token expired, please sign in again."
        if "invalid" in error_code or "revoked" in error_code:
            return "OpenAI OAuth refresh token invalid, please sign in again."
    return "Failed to refresh OpenAI auth token."


def _refresh_token_file(
    token_file: OpenAIAuthTokenFile,
    *,
    path: Path,
    http_client: _RefreshHTTPClient | None,
) -> OpenAIAuthTokenFile:
    refresh_request = {
        "client_id": _REFRESH_CLIENT_ID,
        "grant_type": _REFRESH_GRANT_TYPE,
        "refresh_token": token_file.openai.refresh,
    }

    client_ctx = nullcontext(http_client) if http_client is not None else _create_default_http_client()
    try:
        with client_ctx as resolved_http_client:
            refresh_http_client = cast(_RefreshHTTPClient, resolved_http_client)
            response = refresh_http_client.post(_REFRESH_TOKEN_URL, json=refresh_request)
    except OpenAIAuthRefreshError:
        raise
    except Exception as exc:
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.") from exc

    response_payload = _read_response_json(response)
    if response.status_code < 200 or response.status_code >= 300:
        raise OpenAIAuthRefreshError(_refresh_failure_message(response_payload))

    access_token = _read_optional_json_string(response_payload, "access_token")
    if access_token is None:
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.")

    try:
        account_from_token, exp_from_token = _read_jwt_claims(access_token, path=path)
    except OpenAIAuthTokenFileError as exc:
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.") from exc

    if exp_from_token is None:
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.")

    account_id = token_file.openai.accountId or account_from_token
    if account_id is None:
        raise OpenAIAuthRefreshError("Failed to refresh OpenAI auth token.")

    refresh_token = _read_optional_json_string(
        response_payload, "refresh_token"
    ) or token_file.openai.refresh

    return OpenAIAuthTokenFile(
        openai=OpenAIOAuthToken(
            type="oauth",
            refresh=refresh_token,
            access=access_token,
            expires=exp_from_token,
            accountId=account_id,
        )
    )


def read_openai_auth_token_file(path: str | Path) -> OpenAIAuthTokenFile:
    token_path = Path(path).expanduser()
    try:
        raw_content = token_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise OpenAIAuthTokenFileError(f"OpenAI auth token file not found: {token_path}") from exc
    except OSError as exc:
        raise OpenAIAuthTokenFileError(f"Failed to read OpenAI auth token file: {token_path}") from exc

    try:
        payload = cast(object, json.loads(raw_content))
    except json.JSONDecodeError as exc:
        raise OpenAIAuthTokenFileError(f"Invalid OpenAI auth token file JSON: {token_path}") from exc

    try:
        return OpenAIAuthTokenFile.model_validate(payload)
    except ValidationError as exc:
        raise OpenAIAuthTokenFileError(
            f"Invalid OpenAI auth token file schema: {token_path}"
        ) from exc


def _write_secret_file(path: Path, *, content: str) -> None:
    file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _TOKEN_FILE_MODE)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as tmp_file:
            _ = tmp_file.write(content)
        os.chmod(path, _TOKEN_FILE_MODE)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(file_descriptor)
        raise


def write_openai_auth_token_file(path: str | Path, token_file: OpenAIAuthTokenFile) -> Path:
    token_path = Path(path).expanduser()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = token_path.with_name(f".{token_path.name}.tmp")
    serialized = json.dumps(token_file.model_dump(mode="json", exclude_none=True), indent=2) + "\n"

    try:
        _write_secret_file(temporary_path, content=serialized)
        os.replace(temporary_path, token_path)
        os.chmod(token_path, _TOKEN_FILE_MODE)
    except OSError as exc:
        with contextlib.suppress(OSError):
            if temporary_path.exists():
                temporary_path.unlink()
        raise OpenAIAuthTokenFileError(
            f"Failed to write OpenAI auth token file: {token_path}"
        ) from exc

    return token_path


def load_openai_auth_session(
    path: str | Path,
    *,
    now_ms: int | None = None,
    refresh_window_ms: int = _PROACTIVE_REFRESH_WINDOW_MS,
    http_client: _RefreshHTTPClient | None = None,
) -> OpenAIAuthSession:
    """Load a usable OpenAI OAuth session from token file.

    When the access token is near expiry, this function will refresh it and
    atomically update the token file.
    """
    if refresh_window_ms < 0:
        raise OpenAIAuthError("refresh_window_ms must be >= 0")

    token_path = Path(path).expanduser()
    effective_now_ms = _current_time_ms() if now_ms is None else now_ms

    token_file = read_openai_auth_token_file(token_path)
    session = _resolve_auth_session(token_file, path=token_path)
    if not _is_refresh_required(
        session, now_ms=effective_now_ms, refresh_window_ms=refresh_window_ms,
    ):
        return session

    with _token_file_lock(token_path):
        latest_token_file = read_openai_auth_token_file(token_path)
        latest_session = _resolve_auth_session(latest_token_file, path=token_path)
        if not _is_refresh_required(
            latest_session, now_ms=effective_now_ms, refresh_window_ms=refresh_window_ms,
        ):
            return latest_session

        refreshed = _refresh_token_file(
            latest_token_file, path=token_path, http_client=http_client,
        )
        _ = write_openai_auth_token_file(token_path, refreshed)
        return _resolve_auth_session(refreshed, path=token_path)


__all__ = [
    "OpenAIAuthError",
    "OpenAIAuthRefreshError",
    "OpenAIAuthSession",
    "OpenAIAuthTokenFileError",
    "load_openai_auth_session",
    "read_openai_auth_token_file",
    "write_openai_auth_token_file",
]
