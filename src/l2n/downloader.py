"""通过 yt-dlp 获取 YouTube 视频下载地址。"""

import subprocess
from pathlib import Path

import click
import yt_dlp

from l2n.transcriber import extract_video_id

# 视频文件扩展名
_VIDEO_EXTS = (".mp4", ".webm", ".mkv")
MAX_RETRIES = 3


def list_formats(url: str) -> list[dict]:
    """获取视频的所有可用格式信息。

    Returns:
        格式列表，每项包含 format_id, ext, resolution, filesize, url 等
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
        "cookiesfrombrowser": ("chrome",),
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


def _check_video_complete(video_path: Path, expected_duration: float, expected_filesize: int = 0) -> bool:
    """检查视频文件是否完整（时长 + 文件大小双重校验）。"""
    if not video_path.exists():
        return False
    # .part 文件一定是不完整的
    if video_path.name.endswith(".part"):
        return False
    actual_size = video_path.stat().st_size
    if actual_size == 0:
        return False

    # 文件大小校验：与预期大小对比（允许 10% 误差）
    if expected_filesize and actual_size < expected_filesize * 0.9:
        return False

    # 时长校验：用 ffprobe 检测实际时长
    if expected_duration:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, timeout=10,
            )
            actual = float(result.stdout.strip())
            if abs(actual - expected_duration) > 5:
                return False
        except Exception:
            pass  # ffprobe 不可用时跳过时长检查

    # 兜底：基于时长估算最小合理文件大小（至少 50kbps）
    if expected_duration and actual_size < expected_duration * 6250:
        return False

    return True


def _find_existing_video(video_dir: Path) -> Path | None:
    """查找目录中已有的视频文件（mp4/webm/mkv），也检测 .part 未完成文件。"""
    if not video_dir.exists():
        return None
    # 优先返回完整格式的文件
    for f in video_dir.iterdir():
        if f.suffix in _VIDEO_EXTS and f.stat().st_size > 0:
            return f
    # 其次返回 .part 文件（yt-dlp 下载中断的残留）
    for f in video_dir.iterdir():
        if f.name.endswith(".part") and any(ext in f.name for ext in _VIDEO_EXTS):
            return f
    return None


def download_video(url: str, output_dir: str | None = None) -> str:
    """下载最高画质纯视频（bestvideo），支持断点续传和完整性校验。

    Args:
        url: YouTube 视频 URL
        output_dir: 输出目录，默认为 output/subtitle/<video_id>/

    Returns:
        下载的视频文件路径
    """
    video_id = extract_video_id(url)
    if output_dir is None:
        output_dir = f"output/subtitle/{video_id}"

    video_dir = Path(output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # 先获取视频信息（用于时长和文件大小校验）
    ydl_info_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
        "cookiesfrombrowser": ("chrome",),
    }
    with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    expected_duration = info.get("duration", 0)

    # 估算预期文件大小：让 yt-dlp 解析 bestvideo 实际选中的格式
    expected_filesize = 0
    try:
        with yt_dlp.YoutubeDL({**ydl_info_opts, "format": "bestvideo"}) as ydl_fs:
            sel_info = ydl_fs.extract_info(url, download=False)
            expected_filesize = sel_info.get("filesize") or sel_info.get("filesize_approx") or 0
    except Exception:
        pass

    # 检查已有视频文件是否完整
    existing = _find_existing_video(video_dir)
    if existing and (expected_duration or expected_filesize):
        if _check_video_complete(existing, expected_duration, expected_filesize):
            size_mb = existing.stat().st_size / (1024 * 1024)
            click.echo(f"   ✓ 视频已存在且完整: {existing} ({size_mb:.1f}MB)")
            return str(existing)
        else:
            click.echo(f"   ⚠️ 已有视频文件不完整，重新下载...")
            existing.unlink()

    fmt = "bestvideo"
    ydl_opts = {
        "format": fmt,
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "quiet": False,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
        "cookiesfrombrowser": ("chrome",),
    }

    # 下载视频（最多重试 MAX_RETRIES 次）
    for attempt in range(1, MAX_RETRIES + 1):
        click.echo(f"📥 正在下载视频 (bestvideo)...{f' (第 {attempt} 次尝试)' if attempt > 1 else ''}")

        # 清理上次不完整的文件（包括 .part 残留）
        for f in video_dir.iterdir():
            if f.suffix in _VIDEO_EXTS or (f.name.endswith(".part") and any(ext in f.name for ext in _VIDEO_EXTS)):
                f.unlink()

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
        except Exception as e:
            if attempt < MAX_RETRIES:
                click.echo(f"   ⚠️ 下载出错: {e}，重试...")
                continue
            raise click.ClickException(f"视频下载失败（已重试 {MAX_RETRIES} 次）: {e}")

        video_path = Path(filename)
        # 也检查 webm 等格式
        if not video_path.exists():
            found = _find_existing_video(video_dir)
            if found:
                video_path = found
                filename = str(found)

        if not video_path.exists():
            if attempt < MAX_RETRIES:
                click.echo(f"   ⚠️ 未找到视频文件，重试...")
                continue
            raise click.ClickException("视频下载失败，未找到文件")

        # 校验下载完整性
        if not _check_video_complete(video_path, expected_duration, expected_filesize):
            if attempt < MAX_RETRIES:
                click.echo(f"   ⚠️ 视频下载不完整，重试...")
                continue
            raise click.ClickException(
                f"视频下载不完整（已重试 {MAX_RETRIES} 次），请检查网络后重新运行"
            )

        # 下载成功
        break

    size_mb = video_path.stat().st_size / (1024 * 1024)
    click.echo(f"   ✅ 视频已保存: {filename} ({size_mb:.1f}MB)")

    # 检测视频是否有音频轨道，没有则自动合并目录下的 audio.m4a
    video_path = Path(filename)
    has_audio = _check_has_audio(video_path)
    if not has_audio:
        audio_path = video_dir / "audio.m4a"
        if audio_path.exists():
            merged = _merge_audio(video_path, audio_path)
            if merged:
                filename = str(merged)

    return filename


def _check_has_audio(video_path: Path) -> bool:
    """用 ffprobe 检测视频文件是否包含音频轨道。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout
    except Exception:
        return True  # ffprobe 不可用时保守跳过


def _merge_audio(video_path: Path, audio_path: Path) -> Path | None:
    """将纯视频和音频合并，返回合并后的文件路径。"""
    # 输出格式：mp4 容器兼容性最好
    merged_path = video_path.with_name(video_path.stem + "_merged.mp4")
    click.echo(f"   🔄 视频无音频轨道，正在合并 {audio_path.name}...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0",
             str(merged_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and merged_path.exists() and merged_path.stat().st_size > 0:
            # 合并成功，删除无音频的原文件
            video_path.unlink()
            # 重命名去掉 _merged 后缀
            final_path = merged_path.with_name(video_path.stem + ".mp4")
            merged_path.rename(final_path)
            size_mb = final_path.stat().st_size / (1024 * 1024)
            click.echo(f"   ✅ 音视频合并完成: {final_path.name} ({size_mb:.1f}MB)")
            return final_path
        else:
            click.echo(f"   ⚠️ 合并失败: {result.stderr[:200]}")
            if merged_path.exists():
                merged_path.unlink()
            return None
    except Exception as e:
        click.echo(f"   ⚠️ 合并失败: {e}")
        if merged_path.exists():
            merged_path.unlink()
        return None
