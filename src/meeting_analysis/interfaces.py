"""Provider seams.

Every model-backed stage sits behind one of these protocols, so a different
implementation (larger model, cloud API, or a custom diarization fallback) can
be dropped in without touching alignment, orchestration, or the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import Diarization, EmotionPrediction, GenderPrediction, Transcript

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@runtime_checkable
class Transcriber(Protocol):
    """Speech-to-text provider (MVP: faster-whisper)."""

    def transcribe(self, audio_path: Path) -> Transcript: ...


@runtime_checkable
class Diarizer(Protocol):
    """Speaker diarization provider (MVP: pyannote speaker-diarization-3.1).

    Kept deliberately narrow so a speaker-embedding + clustering fallback, or a
    cloud diarization API, can implement it later.
    """

    def diarize(self, audio_path: Path) -> Diarization: ...


@runtime_checkable
class GenderClassifier(Protocol):
    """Per-SPEAKER gender classification over pooled audio."""

    def classify(self, audio: "np.ndarray", sample_rate: int) -> GenderPrediction: ...


@runtime_checkable
class EmotionAnalyzer(Protocol):
    """Per-UTTERANCE emotion classification over Vietnamese text."""

    def analyze(self, text: str) -> EmotionPrediction: ...
