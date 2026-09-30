"""Configuration loading.

Reads `.env` (simple ``KEY=VALUE``) plus the process environment. Deliberately
dependency-free so the pipeline runs without the full settings stack.

Note: Hugging Face tooling only auto-reads the UPPERCASE ``HF_TOKEN`` env var,
so a lowercase ``hf_token`` in ``.env`` is normalised here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"


def configure_hf_home() -> Path:
    """Keep Hugging Face downloads inside the project.

    The default ``~/.cache/huggingface`` lives outside the workspace and is
    write-denied in this environment, so the cache is redirected into the repo
    (``models_cache/``, gitignored) before any model is fetched.
    """
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODELS_CACHE_DIR))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return MODELS_CACHE_DIR


configure_hf_home()


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` .env file. Returns an empty dict if absent."""
    env_path = path or DEFAULT_ENV_PATH
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_hf_token() -> str | None:
    """Resolve the Hugging Face token and export it as ``HF_TOKEN``.

    Accepts either casing in ``.env`` because ``huggingface_hub`` only looks at
    the uppercase environment variable.
    """
    dotenv = load_dotenv()
    token = (
        os.environ.get("HF_TOKEN")
        or dotenv.get("HF_TOKEN")
        or dotenv.get("hf_token")
        or os.environ.get("hf_token")
    )
    if token:
        os.environ["HF_TOKEN"] = token
    return token or None


@dataclass
class Settings:
    """Runtime settings for the analysis pipeline."""

    # Audio
    sample_rate: int = 16_000

    # ASR
    asr_model: str = os.environ.get("ASR_MODEL", "medium")
    asr_compute_type: str = os.environ.get("ASR_COMPUTE_TYPE", "int8_float16")
    asr_language: str = "vi"

    # Diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    num_speakers: int | None = None  # None => let pyannote detect

    # Gender (per speaker, pooled audio)
    gender_model: str = os.environ.get(
        "GENDER_MODEL", "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    )
    gender_pool_seconds: float = 20.0
    male_f0_max_hz: float = 180.0
    female_f0_min_hz: float = 165.0

    # Emotion (per utterance, Vietnamese text)
    emotion_model: str = os.environ.get(
        "EMOTION_MODEL", "wonrax/phobert-base-vietnamese-sentiment"
    )

    # Device
    device: str = os.environ.get("DEVICE", "cuda")

    def resolve_device(self) -> str:
        """Return the requested device, degrading to CPU if CUDA is unavailable."""
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
            except Exception:
                return "cpu"
        return self.device
