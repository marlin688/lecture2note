"""使用 Whisper 本地模型转录音频为 SRT 字幕。"""

from pathlib import Path

import click
import whisper


def transcribe_to_srt(audio_path: str, model_name: str = "medium") -> str:
    """用 Whisper 转录音频文件，输出 SRT 格式字符串。

    Args:
        audio_path: 音频文件路径
        model_name: Whisper 模型名称 (tiny/base/small/medium/large)

    Returns:
        SRT 格式字符串
    """
    click.echo(f"   🔄 加载 Whisper {model_name} 模型...")
    model = whisper.load_model(model_name)

    click.echo(f"   🎙️ 正在转录 (这可能需要几分钟)...")
    result = model.transcribe(
        audio_path,
        language="en",
        verbose=False,
    )

    # 将 Whisper segments 转为 SRT
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
