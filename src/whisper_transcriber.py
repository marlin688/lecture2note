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


def download_audio(url: str, output_dir: Path) -> Path:
    """用 yt-dlp 下载音频。

    Returns:
        下载的音频文件路径
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    click.echo("   📥 正在下载音频...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # 找到实际下载的文件
        ext = info.get("ext", "m4a")

    audio_path = output_dir / f"audio.{ext}"
    if audio_path.exists():
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        click.echo(f"   ✓ 音频已下载: {audio_path} ({size_mb:.1f}MB)")
        return audio_path

    # 如果文件名不匹配，找目录下的音频文件
    for f in output_dir.glob("audio.*"):
        if f.suffix in (".m4a", ".mp3", ".wav", ".webm", ".ogg"):
            click.echo(f"   ✓ 音频已下载: {f}")
            return f

    raise click.ClickException(f"音频下载失败，未找到文件")
