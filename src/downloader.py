"""通过 yt-dlp 获取 YouTube 视频下载地址。"""

import click
import yt_dlp

from src.transcriber import extract_video_id


def list_formats(url: str) -> list[dict]:
    """获取视频的所有可用格式信息。

    Returns:
        格式列表，每项包含 format_id, ext, resolution, filesize, url 等
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = []
    for f in info.get("formats", []):
        # 跳过纯音频或无 URL 的格式
        fmt = {
            "format_id": f.get("format_id", ""),
            "ext": f.get("ext", ""),
            "resolution": f.get("resolution", "audio only"),
            "fps": f.get("fps"),
            "vcodec": f.get("vcodec", "none"),
            "acodec": f.get("acodec", "none"),
            "filesize": f.get("filesize") or f.get("filesize_approx"),
            "url": f.get("url", ""),
        }
        formats.append(fmt)

    return formats


def _format_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "未知"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f}MB"
    return f"{size_bytes / (1024*1024*1024):.2f}GB"


def print_formats(url: str):
    """打印视频的可用下载格式（按清晰度分组）。"""
    click.echo("📡 正在获取视频格式信息...")
    formats = list_formats(url)

    # 分为视频+音频、纯视频、纯音频
    combined = []  # 有视频也有音频
    video_only = []
    audio_only = []

    for f in formats:
        has_video = f["vcodec"] != "none"
        has_audio = f["acodec"] != "none"
        if has_video and has_audio:
            combined.append(f)
        elif has_video:
            video_only.append(f)
        elif has_audio:
            audio_only.append(f)

    click.echo(f"\n{'='*70}")
    click.echo("视频+音频（可直接播放）:")
    click.echo(f"{'='*70}")
    if combined:
        for f in combined:
            click.echo(
                f"  [{f['format_id']:>5}] {f['resolution']:>10} "
                f"{f['ext']:>5}  {_format_size(f['filesize']):>8}"
            )
            if f["url"]:
                click.echo(f"         🔗 {f['url'][:120]}...")
    else:
        click.echo("  (无)")

    click.echo(f"\n{'='*70}")
    click.echo("纯视频（需要单独下载音频后合并）:")
    click.echo(f"{'='*70}")
    for f in video_only:
        fps_str = f"  {f['fps']}fps" if f.get('fps') else ""
        click.echo(
            f"  [{f['format_id']:>5}] {f['resolution']:>10} "
            f"{f['ext']:>5}  {_format_size(f['filesize']):>8}  "
            f"{f['vcodec']}{fps_str}"
        )

    click.echo(f"\n{'='*70}")
    click.echo("纯音频:")
    click.echo(f"{'='*70}")
    for f in audio_only:
        click.echo(
            f"  [{f['format_id']:>5}] {'audio':>10} "
            f"{f['ext']:>5}  {_format_size(f['filesize']):>8}  "
            f"{f['acodec']}"
        )

    click.echo(f"\n💡 使用 yt-dlp 下载: yt-dlp -f <format_id> \"{url}\"")
    click.echo(f"   下载最佳画质: yt-dlp -f 'bestvideo+bestaudio' \"{url}\"")


_RES_MAP = {
    "720p": 720,
    "1080p": 1080,
    "2k": 1440,
    "4k": 2160,
}


def download_video(url: str, output_dir: str | None = None, max_res: str = "1080p") -> str:
    """下载视频（bestvideo+bestaudio 自动合并）。

    Args:
        url: YouTube 视频 URL
        output_dir: 输出目录，默认为 output/subtitle/<video_id>/
        max_res: 最大分辨率限制，可选 720p/1080p/2k/4k

    Returns:
        下载的视频文件路径
    """
    video_id = extract_video_id(url)
    if output_dir is None:
        output_dir = f"output/subtitle/{video_id}"

    import os
    os.makedirs(output_dir, exist_ok=True)

    height = _RES_MAP.get(max_res, 1080)
    fmt = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"

    ydl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "quiet": False,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
    }

    click.echo(f"📥 正在下载视频 (最高 {max_res})...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # 合并后扩展名可能变成 .mp4
        from pathlib import Path
        final = Path(filename).with_suffix(".mp4")
        if final.exists():
            filename = str(final)

    click.echo(f"   ✅ 视频已保存: {filename}")
    return filename
