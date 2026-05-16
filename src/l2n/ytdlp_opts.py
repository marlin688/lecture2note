"""yt-dlp 配置工厂：集中管理浏览器 cookies、进度条静默等公共选项。"""

import os
import sys


def _browser() -> str:
    return os.getenv("YTDLP_BROWSER", "chrome")


def _quiet_progress() -> bool:
    # 非 TTY（重定向、nohup）下关闭百分比刷新，避免日志爆炸
    return not sys.stdout.isatty()


def base_opts(*, quiet: bool = False) -> dict:
    """所有 yt-dlp 调用的公共底座。

    js_runtimes 用顶层 {"node": {}}：yt-dlp 2026.03.17 起默认仅启用 deno，
    需把 node 作为顶层参数传入才能被检测（旧的 extractor_args.youtube.js_runtimes 已失效）。
    """
    return {
        "quiet": quiet,
        "no_warnings": True,
        "noprogress": _quiet_progress(),
        "js_runtimes": {"node": {}},
        "cookiesfrombrowser": (_browser(),),
    }


def info_opts() -> dict:
    """只读取元信息时使用（quiet）。"""
    return base_opts(quiet=True)


def download_opts(*, format: str, outtmpl: str, quiet: bool = False) -> dict:
    """下载时使用，调用方提供 format 和 outtmpl。"""
    return {
        **base_opts(quiet=quiet),
        "format": format,
        "outtmpl": outtmpl,
    }
