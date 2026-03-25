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

    # 过滤 Whisper 幻觉（同一个词重复多次）并转为 SRT
    segments = result["segments"]
    filtered = []
    hallucination_count = 0
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        words = text.split()
        # 检测重复幻觉：如果某个词占总词数的 70% 以上且词数 > 5，判定为幻觉
        if len(words) > 5:
            from collections import Counter
            most_common_word, most_common_count = Counter(words).most_common(1)[0]
            if most_common_count / len(words) > 0.7:
                hallucination_count += 1
                continue
        # 单条字幕过长（超过 500 字符）也可能是幻觉
        if len(text) > 500:
            hallucination_count += 1
            continue
        filtered.append(seg)

    if hallucination_count:
        click.echo(f"   ⚠️ 过滤了 {hallucination_count} 条 Whisper 幻觉字幕")

    srt_lines = []
    for i, seg in enumerate(filtered):
        idx = i + 1
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")

    click.echo(f"   ✓ 转录完成: {len(filtered)} 条字幕（原始 {len(segments)} 条）")
    return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间格式: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _check_audio_complete(audio_path: Path, expected_duration: float, expected_filesize: int = 0) -> bool:
    """检查音频文件是否完整（时长 + 文件大小双重校验）。"""
    actual_size = audio_path.stat().st_size

    # 文件大小校验：与预期大小对比（允许 10% 误差）
    if expected_filesize and actual_size < expected_filesize * 0.9:
        return False

    # 时长校验：用 ffprobe 检测实际时长
    if expected_duration:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(audio_path)],
                capture_output=True, text=True, timeout=10,
            )
            actual = float(result.stdout.strip())
            if abs(actual - expected_duration) > 5:
                return False
        except Exception:
            pass  # ffprobe 不可用时跳过时长检查

    # 兜底：基于时长估算最小合理文件大小（至少 32kbps = 4000 bytes/s）
    if expected_duration and actual_size < expected_duration * 4000:
        return False

    return True


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

    # 获取预期文件大小（从 bestaudio 格式信息中提取）
    expected_filesize = 0
    for fmt in info.get("formats", []):
        if fmt.get("format_id") == "140":  # m4a bestaudio
            expected_filesize = fmt.get("filesize") or fmt.get("filesize_approx") or 0
            break
    if not expected_filesize:
        for fmt in info.get("formats", []):
            if fmt.get("acodec") != "none" and fmt.get("vcodec") in ("none", None):
                expected_filesize = fmt.get("filesize") or fmt.get("filesize_approx") or 0
                if expected_filesize:
                    break

    # 检查已有音频文件是否完整
    existing = None
    for f in output_dir.glob("audio.*"):
        if f.suffix in (".m4a", ".mp3", ".wav", ".webm", ".ogg"):
            existing = f
            break

    if existing and (expected_duration or expected_filesize):
        if _check_audio_complete(existing, expected_duration, expected_filesize):
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

        # 校验下载完整性（时长 + 文件大小双重校验）
        if not _check_audio_complete(audio_path, expected_duration, expected_filesize):
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
