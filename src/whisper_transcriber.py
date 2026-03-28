"""使用 mlx-whisper (Apple Silicon GPU 加速) 转录音频为 SRT 字幕。"""

import re
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
            word_timestamps=True,              # 词级时间戳，用于智能断句
        )
    finally:
        os.environ.update(proxy_vars)

    # 过滤 Whisper 幻觉（同一个词重复多次）
    segments = result["segments"]
    filtered_segments = []
    hallucination_count = 0
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        words_list = text.split()
        # 检测重复幻觉：如果某个词占总词数的 70% 以上且词数 > 5，判定为幻觉
        if len(words_list) > 5:
            from collections import Counter
            _, most_common_count = Counter(words_list).most_common(1)[0]
            if most_common_count / len(words_list) > 0.7:
                hallucination_count += 1
                continue
        # 单条字幕过长（超过 500 字符）也可能是幻觉
        if len(text) > 500:
            hallucination_count += 1
            continue
        filtered_segments.append(seg)

    if hallucination_count:
        click.echo(f"   ⚠️ 过滤了 {hallucination_count} 条 Whisper 幻觉字幕")

    # 提取所有词级时间戳，展平为 [(word, start, end), ...]
    all_words = []
    for seg in filtered_segments:
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            if word:
                all_words.append((word, w["start"], w["end"]))

    if not all_words:
        # 没有词级时间戳时回退到原始段
        all_words = []
        for seg in filtered_segments:
            text = seg["text"].strip()
            if text:
                all_words.append((text, seg["start"], seg["end"]))

    # 按标点和长度规则重新组段
    regroups = _regroup_words(all_words)

    srt_lines = []
    for i, (text, start, end) in enumerate(regroups):
        srt_lines.append(f"{i + 1}")
        srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        srt_lines.append(text)
        srt_lines.append("")

    click.echo(f"   ✓ 转录完成: {len(regroups)} 条字幕（原始 {len(segments)} 条，断句优化 {len(filtered_segments)} → {len(regroups)}）")
    return "\n".join(srt_lines)


# ── 断句参数 ──
_MIN_WORDS = 3        # 单段最少词数
_MIN_DURATION = 0.8   # 单段最短时长(秒)
_SOFT_MAX_WORDS = 18  # P3: 词数软上限（触发前瞻）
_HARD_MAX_WORDS = 22  # P3: 词数硬上限（无条件断）
_LOOKAHEAD = 4        # P3: 软上限后最多再看几个词
_MAX_DURATION = 7.0   # P4: 时长硬上限(秒)
_COMMA_MIN = 10       # P2: 逗号断开的最低词数门槛
_PAUSE_THRESH = 0.5   # P1.5: 停顿断句阈值(秒)
_PAUSE_MIN_WORDS = 6  # P1.5: 停顿断句最低词数

# P1: 转折词 — 强信号(8词触发) / 弱信号(12词触发)
_STRONG_DISCOURSE = {
    "but", "however", "instead", "although", "though", "therefore",
    "yet", "still", "nevertheless", "furthermore", "moreover",
    "first", "second", "third", "next", "finally", "lastly",
    "for", "moving",  # "For example", "Moving on"
}
_WEAK_DISCOURSE = {
    "now", "so", "also", "okay", "ok", "alright", "basically",
    "actually", "essentially", "anyway", "plus", "another",
}
_STRONG_DISCOURSE_MIN = 8
_WEAK_DISCOURSE_MIN = 12


def _regroup_words(words: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """将词级时间戳按分层规则重新组合为字幕段。

    优先级:
      P0:   句末标点 (.?!) → 强制断开
      P1:   转折词(大写+前词标点或停顿) → 在该词之前断开
      P1.5: 词间停顿 > 0.5s 且 >= 6 词 → 断开
      P2:   逗号 → >= 10 词时断开
      P3:   18 词软上限(前瞻 4 词找标点) + 22 词硬上限
      P4:   时长硬上限 7s → 找最长停顿处断开

    断后 <=3 词且 <1s 的短尾合并回前一段。
    """
    segments: list[tuple[str, float, float]] = []
    buf: list[tuple[str, float, float]] = []  # [(word, start, end), ...]
    in_soft_zone = False  # 是否在 18-22 词的软上限前瞻区

    def _flush():
        nonlocal in_soft_zone
        if not buf:
            return
        text = " ".join(w for w, _, _ in buf)
        segments.append((text, buf[0][1], buf[-1][2]))
        buf.clear()
        in_soft_zone = False

    def _flush_split(split_idx: int):
        """在 split_idx 处断开: [0..split_idx] 输出, [split_idx+1..] 留在 buf。"""
        nonlocal in_soft_zone
        left = buf[:split_idx + 1]
        right = buf[split_idx + 1:]
        text = " ".join(w for w, _, _ in left)
        segments.append((text, left[0][1], left[-1][2]))
        buf.clear()
        buf.extend(right)
        in_soft_zone = False

    prev_end = 0.0

    for idx, (word, start, end) in enumerate(words):
        n = len(buf)
        gap = start - prev_end if buf else 0.0

        # ── P1: 转折词 — 在该词之前断开（先于 append）──
        if n > 0 and word[0].isupper():
            bare = re.sub(r'[^a-zA-Z]', '', word).lower()
            prev_word = buf[-1][0]
            at_boundary = bool(re.search(r'[.?!,;:]$', prev_word)) or gap > 0.3
            if at_boundary:
                if bare in _STRONG_DISCOURSE and n >= _STRONG_DISCOURSE_MIN:
                    _flush()
                elif bare in _WEAK_DISCOURSE and n >= _WEAK_DISCOURSE_MIN:
                    _flush()

        # ── P1.5: 停顿断句 ──
        if gap > _PAUSE_THRESH and len(buf) >= _PAUSE_MIN_WORDS:
            _flush()

        buf.append((word, start, end))
        prev_end = end
        n = len(buf)
        duration = buf[-1][2] - buf[0][1]

        # ── P0: 句末标点 → 强制断开 ──
        if re.search(r'[.?!]$', word):
            _flush()
            continue

        # ── P2: 逗号断句 ──
        if re.search(r'[,;:]$', word) and n >= _COMMA_MIN:
            _flush()
            continue

        # ── 软上限前瞻区: 遇到标点立刻断，超时也断 ──
        if in_soft_zone:
            if re.search(r'[,;:]$', word):
                _flush()
                continue
            if duration >= _MAX_DURATION:
                split_at = _find_pause_break(buf) or _find_clause_break(buf, urgent=True)
                if split_at is not None:
                    _flush_split(split_at)
                else:
                    _flush()
                continue

        # ── P3: 词数软上限 → 前瞻找标点 ──
        if n >= _SOFT_MAX_WORDS and not in_soft_zone:
            # 前瞻: 未来 _LOOKAHEAD 个词里有标点吗?
            has_punct_ahead = False
            for lookahead_i in range(idx + 1, min(idx + 1 + _LOOKAHEAD, len(words))):
                future_word = words[lookahead_i][0]
                if re.search(r'[.?!,;:]$', future_word):
                    has_punct_ahead = True
                    break
            if has_punct_ahead:
                in_soft_zone = True  # 进入前瞻等待区，等标点触发断开
            else:
                # 没有标点要来 → 回溯找逗号或转折词断开
                split_at = _find_clause_break(buf) or _find_discourse_break(buf)
                if split_at is not None:
                    _flush_split(split_at)
                else:
                    _flush()
            continue

        # ── P4: 时长硬上限 → 找最长停顿断开（优先于 P3 硬上限）──
        if duration >= _MAX_DURATION:
            split_at = _find_pause_break(buf)
            if split_at is not None:
                _flush_split(split_at)
            else:
                split_at = _find_clause_break(buf, urgent=True)
                if split_at is not None:
                    _flush_split(split_at)
                else:
                    _flush()
            continue

        # ── P3: 硬上限 22 词 → 找逗号或转折词断开 ──
        if n >= _HARD_MAX_WORDS:
            split_at = _find_clause_break(buf) or _find_discourse_break(buf)
            if split_at is not None:
                _flush_split(split_at)
            else:
                _flush()
            continue

    _flush()

    # ── 后处理: 合并碎片 ──
    # 1) <=2 词且 <0.5s → 一定合并回前一段（"right?" "Okay." 等附和词太短看不到）
    # 2) <=3 词且 <1.0s 且不以句号结尾 → 合并回前一段
    # 3) 其余保留（包括短但以句号结尾的完整句如 "I don't know."）
    merged: list[tuple[str, float, float]] = []
    for text, s, e in segments:
        wc = len(text.split())
        dur = e - s
        ends_sent = bool(re.search(r'[.?!]$', text.strip()))
        tiny = wc <= 2 and dur < 0.6
        short_tail = wc <= _MIN_WORDS and dur < 1.0 and not ends_sent
        if merged and (tiny or short_tail):
            prev_text, prev_s, _ = merged[-1]
            merged[-1] = (prev_text + " " + text, prev_s, e)
        else:
            merged.append((text, s, e))

    return merged


def _find_clause_break(buf: list[tuple[str, float, float]], urgent: bool = False) -> int | None:
    """在 buf 中从后往前找最近的逗号/分号/冒号断点。

    Args:
        urgent: True 时用更宽松的 min_pos（P4 时长超限场景）。
    """
    min_pos = 2 if urgent else max(len(buf) // 3, _MIN_WORDS)
    for i in range(len(buf) - 1, min_pos - 1, -1):
        if re.search(r'[,;:]$', buf[i][0]):
            return i
    return None


def _find_discourse_break(buf: list[tuple[str, float, float]]) -> int | None:
    """在 buf 中从后往前找转折词断点（无标点时的回溯策略）。

    找到大写开头的转折词，在其**之前**断开。
    """
    all_discourse = _STRONG_DISCOURSE | _WEAK_DISCOURSE
    min_pos = max(_MIN_WORDS, 4)  # 前半段至少 4 词
    max_pos = len(buf) - _MIN_WORDS  # 后半段也至少留 _MIN_WORDS 词
    for i in range(max_pos, min_pos - 1, -1):
        word = buf[i][0]
        if word[0].isupper():
            bare = re.sub(r'[^a-zA-Z]', '', word).lower()
            if bare in all_discourse:
                return i - 1  # 在转折词之前断开
    return None


def _find_pause_break(buf: list[tuple[str, float, float]]) -> int | None:
    """在 buf 中找最长停顿处断开（用于 P4 时长超限）。"""
    if len(buf) < 2:
        return None
    best_idx = None
    best_gap = 0.2  # P4 场景下降低停顿阈值
    min_pos = 2
    max_pos = len(buf) - 2
    for i in range(min_pos, max_pos):
        gap = buf[i + 1][1] - buf[i][2]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    return best_idx


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
