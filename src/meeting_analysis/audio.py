"""Audio probing and normalization.

Everything downstream consumes one canonical 16 kHz mono WAV so ASR and
diarization share an identical time base — mixing sample rates between stages
silently breaks the speaker/word merge.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .models import AudioInfo

REQUIRED_TOOLS = ("ffmpeg", "ffprobe")


class FFmpegMissingError(RuntimeError):
    """Raised when ffmpeg/ffprobe are not on PATH."""


def require_ffmpeg() -> None:
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise FFmpegMissingError(
            f"Missing required tool(s): {', '.join(missing)}. Install FFmpeg and restart "
            "the shell so PATH is refreshed (Windows: "
            "winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements)."
        )


def probe(path: Path) -> AudioInfo:
    """Read duration/sample-rate/channel metadata via ffprobe."""
    require_ffmpeg()
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    data = json.loads(completed.stdout)
    fmt = data.get("format", {})
    audio = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {}
    )
    return AudioInfo(
        path=str(path),
        duration=float(fmt.get("duration") or 0.0),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
    )


def normalize(src: Path, dst: Path, sample_rate: int = 16_000) -> Path:
    """Convert any container/codec to mono PCM WAV at ``sample_rate``."""
    require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-vn",                 # drop video streams
            "-ac", "1",            # mono
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return dst
