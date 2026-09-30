"""Command-line entry point.

    python -m meeting_analysis.cli <audio> -o <out_dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .pipeline import emit, run_pipeline
from .utils.logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting_analysis",
        description="Vietnamese meeting audio analysis -> [Gender Speaker - Emotion] transcript",
    )
    parser.add_argument("audio", type=Path, help="path to the meeting recording")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("data/artifacts"), help="output directory"
    )
    parser.add_argument("--asr-model", default=None, help="faster-whisper size (default: medium)")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument("--num-speakers", type=int, default=None, help="fix the speaker count")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    settings = Settings()
    if args.asr_model:
        settings.asr_model = args.asr_model
    if args.device:
        settings.device = args.device
    if args.num_speakers:
        settings.num_speakers = args.num_speakers

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    result = run_pipeline(args.audio, args.out, settings)
    txt_path, json_path = emit(result, args.out, args.audio.stem)

    print()
    print("=" * 78)
    print(result.to_text())
    print("=" * 78)
    print()
    print(f"Detected speakers : {result.num_speakers}")
    print(f"Utterances        : {len(result.utterances)}")
    print(f"TXT  -> {txt_path}")
    print(f"JSON -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
