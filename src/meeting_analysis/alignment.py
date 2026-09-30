"""Merge word-level ASR timestamps with diarization turns.

Word-level attribution matters: segment-level timestamps are too coarse to
attribute short back-and-forth turns correctly.

Edge cases handled explicitly:
  * spans several speakers -> assigned to the dominant speaker, and the split
    is recorded in ``notes``;
  * falls inside a diarization gap -> the previous speaker is carried forward;
  * a diarization turn with no ASR text -> produces no utterance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .models import Diarization, Transcript

logger = logging.getLogger(__name__)

UNKNOWN_SPEAKER = "UNKNOWN"
MERGE_GAP_SECONDS = 1.0


@dataclass
class MergedUtterance:
    speaker: str
    start: float
    end: float
    text: str
    notes: list[str] = field(default_factory=list)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_scores(start: float, end: float, turns) -> list[tuple[float, str]]:
    scored = [
        (_overlap(start, end, turn.start, turn.end), turn.speaker)
        for turn in turns
    ]
    scored = [item for item in scored if item[0] > 0.0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _clean(text: str) -> str:
    return " ".join(text.split())


def detected_speakers(utterances: list[MergedUtterance]) -> list[str]:
    """Real speakers in first-appearance order, excluding the UNKNOWN bucket."""
    return [
        speaker
        for speaker in dict.fromkeys(u.speaker for u in utterances)
        if speaker != UNKNOWN_SPEAKER
    ]


def assign_speakers(transcript: Transcript, diarization: Diarization) -> list[MergedUtterance]:
    """Attribute each ASR word to a speaker and group into utterances.

    Resolution runs in passes so units without diarization coverage are filled
    sensibly: a gap *after* a known speaker carries that speaker forward, while a
    gap at the very start of the recording (no preceding speaker) borrows the
    following speaker. Without that look-ahead the opening words become a bogus
    extra speaker, which corrupts the reported speaker count.
    """
    turns = sorted(diarization.turns, key=lambda t: (t.start, t.end))

    # --- pass 1: collect units and their dominant speaker where covered -------
    units: list[dict] = []
    for segment_index, segment in enumerate(transcript.segments):
        if segment.words:
            raw_units = [(w.start, w.end, w.word) for w in segment.words]
        else:
            # No word timings: fall back to one unit for the whole segment.
            raw_units = [(segment.start, segment.end, segment.text)]

        for start, end, raw_text in raw_units:
            scores = _speaker_scores(start, end, turns)
            notes: list[str] = []
            speaker: str | None = None

            if scores:
                speaker = scores[0][1]
                span = max(end - start, 1e-6)
                # Note meaningful secondary overlap (a split turn).
                if len(scores) > 1 and scores[1][0] / span > 0.35:
                    others = sorted({s for _, s in scores[1:]})
                    notes.append(
                        "overlaps multiple speakers "
                        f"({', '.join(others)}); assigned to dominant speaker"
                    )

            units.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": raw_text,
                    "speaker": speaker,
                    "notes": notes,
                    "segment": segment_index,
                }
            )

    # --- pass 2: fill uncovered units ---------------------------------------
    last_seen: str | None = None
    for unit in units:
        if unit["speaker"] is not None:
            last_seen = unit["speaker"]
        elif last_seen is not None:
            unit["speaker"] = last_seen
            unit["notes"].append("no diarization coverage; carried previous speaker forward")

    following: str | None = None
    for unit in reversed(units):
        if unit["speaker"] is not None:
            following = unit["speaker"]
        elif following is not None:
            unit["speaker"] = following
            unit["notes"].append("no preceding speaker; used the following speaker")

    for unit in units:
        if unit["speaker"] is None:
            unit["speaker"] = UNKNOWN_SPEAKER
            unit["notes"].append("no diarization coverage anywhere in the recording")

    # --- pass 3: group consecutive units by speaker --------------------------
    utterances: list[MergedUtterance] = []
    current_segment: int | None = None
    for unit in units:
        speaker = unit["speaker"]
        # Words from the same ASR segment are one phrase and stay together even
        # across a long internal pause (Whisper timings can drift widely inside
        # a single segment). Different segments merge only when the gap is short.
        same_turn = bool(utterances) and utterances[-1].speaker == speaker and (
            unit["segment"] == current_segment
            or unit["start"] - utterances[-1].end < MERGE_GAP_SECONDS
        )
        if same_turn:
            current = utterances[-1]
            current.end = max(current.end, unit["end"])
            current.text = _clean(f"{current.text} {unit['text']}")
            for note in unit["notes"]:
                if note not in current.notes:
                    current.notes.append(note)
        else:
            utterances.append(
                MergedUtterance(
                    speaker=speaker,
                    start=unit["start"],
                    end=unit["end"],
                    text=_clean(unit["text"]),
                    notes=list(unit["notes"]),
                )
            )
            current_segment = unit["segment"]

    utterances = [u for u in utterances if u.text]
    logger.info(
        "Aligned %d utterance(s) across %d detected speaker(s)",
        len(utterances),
        len(detected_speakers(utterances)),
    )
    return utterances
