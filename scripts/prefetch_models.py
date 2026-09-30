"""Pre-download every model weight into the project cache.

Run once before the first pipeline run. Otherwise the first execution downloads
several GB and looks like a hang.

    python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from meeting_analysis.config import configure_hf_home, get_hf_token  # noqa: E402

REPOS = [
    # ASR (CTranslate2 conversion used by faster-whisper)
    "Systran/faster-whisper-medium",
    # Diarization pipeline + its gated segmentation dependency
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    # Gender (audio) and emotion (Vietnamese text)
    "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    "wonrax/phobert-base-vietnamese-sentiment",
]


def main() -> int:
    configure_hf_home()
    token = get_hf_token()
    if not token:
        print("WARNING: no HF token found; gated pyannote repos will fail")

    failures = 0
    for repo in REPOS:
        print(f"-> {repo}", flush=True)
        try:
            path = snapshot_download(repo, token=token)
            print(f"   ok: {path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"   FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    print(f"\nDone. {len(REPOS) - failures}/{len(REPOS)} repos available.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
