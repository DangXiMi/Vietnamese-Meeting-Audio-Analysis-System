"""Domain models — the output contract.

These types are intentionally independent of any ML library so providers can be
swapped without touching the schema or the emitters.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Gender(str, Enum):
    """Deliverable gender labels (Vietnamese)."""

    MALE = "Nam"
    FEMALE = "Nữ"


class Emotion(str, Enum):
    """Deliverable emotion labels (Vietnamese)."""

    HAPPY = "Vui vẻ"
    ANGRY = "Nóng giận"
    NEUTRAL = "Bình thường"


class AudioInfo(BaseModel):
    path: str
    duration: float
    sample_rate: int
    channels: int


class WordTiming(BaseModel):
    start: float
    end: float
    word: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordTiming] = Field(default_factory=list)
    confidence: float | None = None


class Transcript(BaseModel):
    language: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class DiarizationTurn(BaseModel):
    start: float
    end: float
    speaker: str


class Diarization(BaseModel):
    turns: list[DiarizationTurn] = Field(default_factory=list)

    @property
    def num_speakers(self) -> int:
        return len({turn.speaker for turn in self.turns})


class GenderPrediction(BaseModel):
    gender: Gender
    confidence: float
    f0_median: float | None = None
    source: str = "model"
    note: str | None = None


class EmotionPrediction(BaseModel):
    emotion: Emotion
    confidence: float
    raw_label: str | None = None


class Utterance(BaseModel):
    """A merged, speaker-attributed, labeled segment of speech."""

    start: float
    end: float
    speaker: str
    gender: Gender
    emotion: Emotion
    text: str
    notes: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Top-level result. Single source of truth for both emitted artifacts."""

    utterances: list[Utterance] = Field(default_factory=list)
    num_speakers: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Human-readable transcript: ``[<Gender> <Speaker> - <Emotion>]: "<text>"``."""
        return "\n".join(
            f'[{u.gender.value} {u.speaker} - {u.emotion.value}]: "{u.text}"'
            for u in self.utterances
        )

    def to_json(self) -> list[dict[str, Any]]:
        """Machine-readable array keyed exactly as the assignment requires."""
        return [
            {
                "start": round(u.start, 3),
                "end": round(u.end, 3),
                "speaker": u.speaker,
                "gender": u.gender.value,
                "emotion": u.emotion.value,
                "text": u.text,
            }
            for u in self.utterances
        ]
