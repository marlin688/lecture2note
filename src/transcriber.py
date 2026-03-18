"""从 YouTube 视频提取 Transcript 并保存为文本文件。"""

import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi


TRANSCRIPT_DIR = Path("output/transcript")


def extract_video_id(url: str) -> str:
    """从 YouTube URL 中提取 video_id。

    支持格式:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    # 清理 shell 转义带入的反斜杠
    url = url.replace("\\", "")
    patterns = [
        r"(?:v=|/embed/|youtu\.be/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"无法从 URL 中提取 video_id: {url}")


def fetch_transcript(url: str, languages: tuple[str, ...] = ("zh-Hans", "zh", "en")) -> tuple[str, str]:
    """获取 YouTube 视频的 Transcript。

    Args:
        url: YouTube 视频 URL
        languages: 优先语言列表

    Returns:
        (video_id, transcript_text)
    """
    video_id = extract_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=languages)

    lines = [snippet.text for snippet in transcript.snippets]
    text = "\n".join(lines)
    return video_id, text


def save_transcript(video_id: str, text: str) -> Path:
    """将 transcript 保存到 output/transcript/ 目录。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRANSCRIPT_DIR / f"{video_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path
