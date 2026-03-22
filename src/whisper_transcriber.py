"""使用 mlx-whisper (Apple Silicon GPU 加速) 转录音频为 SRT 字幕。"""

from pathlib import Path

import click
import mlx_whisper

# mlx-whisper 模型名映射到 HuggingFace 路径
_MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx-q4",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}


def transcribe_to_srt(audio_path: str, model_name: str = "medium") -> str:
    """用 mlx-whisper 转录音频文件（Apple Silicon GPU 加速），输出 SRT 格式字符串。

    Args:
        audio_path: 音频文件路径
        model_name: 模型名称 (tiny/base/small/medium/large)

    Returns:
        SRT 格式字符串
    """
    model_path = _MLX_MODELS.get(model_name, model_name)
    click.echo(f"   🔄 使用 mlx-whisper {model_name} 模型 (GPU 加速)...")

    click.echo(f"   🎙️ 正在转录...")
    # 临时清除代理环境变量，防止 HuggingFace Hub 读取 SOCKS 代理报错
    import os
    proxy_vars = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if key in os.environ:
            proxy_vars[key] = os.environ.pop(key)
    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            language="en",
            verbose=False,
            condition_on_previous_text=False,  # 防止音乐片段"污染"后续转录
            no_speech_threshold=0.6,           # 更积极地检测语音
        )
    finally:
        os.environ.update(proxy_vars)

    # 将 segments 转为 SRT
    srt_lines = []
    for i, seg in enumerate(result["segments"]):
        idx = i + 1
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")

    click.echo(f"   ✓ 转录完成: {len(result['segments'])} 条字幕")
    return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间格式: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _check_audio_duration(audio_path: Path, expected_duration: float) -> bool:
    """检查音频文件时长是否与预期一致（允许 5 秒误差）。"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=10,
        )
        actual = float(result.stdout.strip())
        return abs(actual - expected_duration) < 5
    except Exception:
        return True  # ffprobe 不可用时跳过检查


def download_audio(url: str, output_dir: Path) -> Path:
    """用 yt-dlp 下载音频。

    Returns:
        下载的音频文件路径
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "audio.%(ext)s")

    # 先获取视频信息（用于时长校验和检测已有文件）
    ydl_info_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
    }
    with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    expected_duration = info.get("duration", 0)

    # 检查已有音频文件是否完整
    existing = None
    for f in output_dir.glob("audio.*"):
        if f.suffix in (".m4a", ".mp3", ".wav", ".webm", ".ogg"):
            existing = f
            break

    if existing and expected_duration:
        if _check_audio_duration(existing, expected_duration):
            size_mb = existing.stat().st_size / (1024 * 1024)
            click.echo(f"   ✓ 音频已存在且完整: {existing} ({size_mb:.1f}MB)")
            return existing
        else:
            click.echo(f"   ⚠️ 已有音频文件不完整，重新下载...")
            existing.unlink()

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
    }

    # 下载音频（最多重试 3 次）
    MAX_RETRIES = 3
    ext = "m4a"
    for attempt in range(1, MAX_RETRIES + 1):
        click.echo(f"   📥 正在下载音频...{f' (第 {attempt} 次尝试)' if attempt > 1 else ''}")

        # 清理上次不完整的文件
        for f in output_dir.glob("audio.*"):
            if f.suffix in (".m4a", ".mp3", ".wav", ".webm", ".ogg"):
                f.unlink()

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info.get("ext", "m4a")

        # 找到下载的文件
        audio_path = output_dir / f"audio.{ext}"
        if not audio_path.exists():
            for f in output_dir.glob("audio.*"):
                if f.suffix in (".m4a", ".mp3", ".wav", ".webm", ".ogg"):
                    audio_path = f
                    break

        if not audio_path.exists():
            if attempt < MAX_RETRIES:
                click.echo(f"   ⚠️ 未找到音频文件，重试...")
                continue
            raise click.ClickException("音频下载失败，未找到文件")

        # 校验下载完整性
        if expected_duration and not _check_audio_duration(audio_path, expected_duration):
            if attempt < MAX_RETRIES:
                click.echo(f"   ⚠️ 音频下载不完整，重试...")
                continue
            raise click.ClickException(
                f"音频下载不完整（已重试 {MAX_RETRIES} 次），请检查网络后重新运行"
            )

        # 下载成功
        break

    # 下载封面
    thumbnail_url = info.get("thumbnail")
    if thumbnail_url:
        try:
            import httpx
            resp = httpx.get(thumbnail_url, follow_redirects=True, timeout=30)
            if resp.status_code == 200:
                # 根据 content-type 确定扩展名
                ct = resp.headers.get("content-type", "")
                ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
                img_ext = ext_map.get(ct, ".jpg")
                cover_path = output_dir / f"cover{img_ext}"
                cover_path.write_bytes(resp.content)
                click.echo(f"   🖼️ 封面已保存: {cover_path}")
        except Exception as e:
            click.echo(f"   ⚠️ 封面下载失败: {e}")

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    click.echo(f"   ✓ 音频已下载: {audio_path} ({size_mb:.1f}MB)")
    return audio_path
