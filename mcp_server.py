"""l2n MCP Server — 将 lecture2note 核心能力暴露为 MCP 工具。

启动方式：
    python mcp_server.py

注册到 Claude Code（添加到 ~/.claude/mcp_servers.json）：
    {
      "l2n": {
        "command": "python",
        "args": ["/path/to/lecture2note/mcp_server.py"],
        "env": {}
      }
    }

或使用 claude mcp add 命令：
    claude mcp add l2n python /path/to/lecture2note/mcp_server.py
"""

from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("l2n", description="Lecture2Note: YouTube download, Whisper transcription, subtitle translation & proofreading")


@mcp.tool()
def download_audio(url: str, output_dir: str = "") -> str:
    """Download audio from a YouTube video URL.

    Args:
        url: YouTube video URL
        output_dir: Output directory path. Defaults to output/subtitle/<video_id>/

    Returns:
        Path to the downloaded audio file
    """
    from l2n.transcriber import extract_video_id
    from l2n.whisper_transcriber import download_audio as _download_audio

    video_id = extract_video_id(url)
    out = Path(output_dir) if output_dir else Path(f"output/subtitle/{video_id}")
    audio_path = _download_audio(url, out)
    return str(audio_path)


@mcp.tool()
def transcribe_audio(audio_path: str, model: str = "medium") -> str:
    """Transcribe an audio file to SRT format using Whisper (Apple Silicon GPU).

    Args:
        audio_path: Path to the audio file (.m4a, .mp3, .wav, etc.)
        model: Whisper model size: tiny, base, small, medium (default), large

    Returns:
        SRT formatted subtitle string
    """
    from l2n.whisper_transcriber import transcribe_to_srt
    return transcribe_to_srt(audio_path, model_name=model)


@mcp.tool()
def translate_subtitle(en_srt: str, model: str, mode: str = "zh") -> str:
    """Translate English SRT subtitles to Chinese.

    Args:
        en_srt: English SRT content string
        model: LLM model name (e.g. claude-sonnet-4-5-20250929, gemini-2.0-flash)
        mode: Output mode — "zh" (Chinese only) or "bilingual" (Chinese + English)

    Returns:
        Translated SRT string
    """
    from l2n.subtitle import translate_srt
    return translate_srt(en_srt, model, mode=mode)


@mcp.tool()
def proofread_subtitle(en_srt: str, zh_srt: str, model: str) -> str:
    """Proofread English subtitles using Chinese translation as reference to fix ASR errors.

    Args:
        en_srt: English SRT content (may contain Whisper ASR errors)
        zh_srt: Chinese translation SRT content (used as reference)
        model: LLM model name

    Returns:
        Corrected English SRT string
    """
    from l2n.subtitle import proofread_en_srt
    return proofread_en_srt(en_srt, zh_srt, model)


@mcp.tool()
def run_subtitle_pipeline(url: str, model: str, lang: str = "zh", whisper_model: str = "medium") -> dict:
    """Run the full subtitle generation pipeline: download → transcribe → translate → proofread.

    Args:
        url: YouTube video URL
        model: LLM model for translation (e.g. claude-sonnet-4-5-20250929)
        lang: Output language — "zh" (Chinese), "bilingual" (Chinese+English), "en" (English only)
        whisper_model: Whisper model size: tiny, base, small, medium (default), large

    Returns:
        Dict with output file paths: {"en_srt": "...", "zh_srt": "..."}
    """
    from l2n.subtitle import generate_subtitle
    from l2n.transcriber import extract_video_id

    video_id = extract_video_id(url)
    out_path = generate_subtitle(url, model, target_lang=lang, whisper_model=whisper_model)

    result = {
        "video_id": video_id,
        "output_path": str(out_path),
        "lang": lang,
    }

    en_path = Path(f"output/subtitle/{video_id}/subtitle_en.srt")
    if en_path.exists():
        result["en_srt_path"] = str(en_path)

    return result


@mcp.tool()
def generate_notes(transcript: str, subject: str = "", model: str = "claude-sonnet-4-5-20250929") -> str:
    """Generate structured lecture notes from a transcript.

    Args:
        transcript: Lecture transcript text
        subject: Subject/course name (optional, auto-detected if empty)
        model: LLM model name

    Returns:
        Markdown formatted lecture notes
    """
    from l2n.noter import process_transcript
    from l2n.assembler import assemble_markdown

    notes = process_transcript(transcript, subject, model)

    if "_raw_markdown" in notes:
        return notes["_raw_markdown"]

    return assemble_markdown(notes)


if __name__ == "__main__":
    mcp.run()
