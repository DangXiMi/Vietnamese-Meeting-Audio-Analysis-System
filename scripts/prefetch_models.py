"""Pre-download every model weight into the project cache.

Run once before the first pipeline run. Otherwise the first execution downloads
several GB and looks like a hang.

    python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# IMPORT ORDER IS LOAD-BEARING.
# Importing the package configures HF_HOME, and huggingface_hub copies HF_HOME
# into a module constant when *it* is first imported. Importing huggingface_hub
# first therefore pins the cache to the default (~/.cache/huggingface), which is
# frequently unwritable — every download then fails with a PermissionError.
from meeting_analysis.config import configure_hf_home, get_hf_token  # noqa: E402

configure_hf_home()

from huggingface_hub import snapshot_download  # noqa: E402

REPOS = [
    # ASR (CTranslate2 conversion used by faster-whisper)
    "Systran/faster-whisper-medium",
    # Diarization pipeline + its dependencies.
    # community-1 is listed EXPLICITLY: pyannote.audio 4.x resolves the 3.1
    # pipeline through it to fetch plda/xvec_transform.npz, and that fetch is
    # what actually fails with a 403 when the licence has not been accepted.
    # Without it here, this script would report success and diarization would
    # still fail later.
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    # Gender (audio) and emotion (Vietnamese text)
    "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    "wonrax/phobert-base-vietnamese-sentiment",
]


def _is_symlink_limitation(exc: BaseException) -> bool:
    """Windows without Developer Mode cannot create the cache's symlinks.

    huggingface_hub normally falls back to copying, but the fallback is partial
    for some repositories and surfaces as WinError 1314 or a missing path. This
    is a cache-population quirk, NOT an access problem: ``from_pretrained``
    handles it at load time, so the pipeline still runs.
    """
    text = str(exc)
    return (
        "1314" in text
        or "privilege is not held" in text
        or isinstance(exc, FileNotFoundError)
    )


def main() -> int:
    configure_hf_home()
    token = get_hf_token()
    if not token:
        print("WARNING: no HF token found; gated pyannote repos will fail")
        print("         Set HF_TOKEN in .env (see .env.example).\n")

    ok = partial = failed = 0
    for repo in REPOS:
        print(f"-> {repo}", flush=True)
        try:
            path = snapshot_download(repo, token=token)
            print(f"   ok: {path}", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            if _is_symlink_limitation(exc):
                partial += 1
                print(
                    "   partial: cached, but Windows symlinks could not be created "
                    f"({type(exc).__name__}).\n"
                    "            Harmless - the pipeline copies at load time. Enable "
                    "Developer Mode to silence this.",
                    flush=True,
                )
            else:
                failed += 1
                print(f"   FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    print(f"\nDone. {ok} ready, {partial} partial, {failed} failed (of {len(REPOS)}).")
    if failed:
        print(
            "\nA FAILED entry usually means the Hugging Face licence has not been "
            "accepted for that repository.\nAccept the terms for BOTH:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
