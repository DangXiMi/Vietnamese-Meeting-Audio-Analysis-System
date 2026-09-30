"""Alignment edge-case tests — no ML dependencies required."""

from __future__ import annotations

from meeting_analysis.alignment import UNKNOWN_SPEAKER, assign_speakers, detected_speakers
from meeting_analysis.models import (
    Diarization,
    DiarizationTurn,
    Transcript,
    TranscriptSegment,
    WordTiming,
)


def _word(start: float, end: float, text: str) -> WordTiming:
    return WordTiming(start=start, end=end, word=text)


def test_assigns_dominant_speaker_by_overlap() -> None:
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="xin chào",
                words=[_word(0.0, 1.0, "xin"), _word(1.0, 2.0, "chào")],
            )
        ],
    )
    diarization = Diarization(
        turns=[
            DiarizationTurn(start=0.0, end=0.9, speaker="S1"),
            DiarizationTurn(start=0.9, end=2.0, speaker="S2"),
        ]
    )

    utterances = assign_speakers(transcript, diarization)
    # "xin" (0-1s) overlaps S1 by 0.9s, "chào" (1-2s) overlaps S2 by 1.0s.
    assert [u.speaker for u in utterances] == ["S1", "S2"]


def test_gap_carries_previous_speaker_forward() -> None:
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.5,
                text="aaa bbb",
                words=[_word(0.0, 1.0, "aaa"), _word(1.5, 2.5, "bbb")],
            )
        ],
    )
    # Diarization covers only 0-1s, so "bbb" (1.5-2.5s) sits in a trailing gap.
    diarization = Diarization(turns=[DiarizationTurn(start=0.0, end=1.0, speaker="S1")])

    utterances = assign_speakers(transcript, diarization)
    assert len(utterances) == 1
    assert utterances[0].speaker == "S1"
    assert any("carried previous speaker forward" in n for n in utterances[0].notes)


def test_leading_gap_borrows_following_speaker() -> None:
    """A gap at the very start must not invent a bogus extra speaker.

    Regression: the opening word of the demo recording became its own
    "UNKNOWN" speaker, inflating the reported speaker count to 3 for a
    2-speaker conversation.
    """
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="chào cô",
                words=[_word(0.0, 1.0, "chào"), _word(1.0, 2.0, "cô")],
            )
        ],
    )
    # Diarization starts at 1.0s, so the first word has no coverage at all.
    diarization = Diarization(turns=[DiarizationTurn(start=1.0, end=2.0, speaker="S1")])

    utterances = assign_speakers(transcript, diarization)
    assert len(utterances) == 1
    assert utterances[0].speaker == "S1"
    assert utterances[0].text == "chào cô"
    assert detected_speakers(utterances) == ["S1"]


def test_no_coverage_anywhere_yields_unknown() -> None:
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=5.0,
                end=6.0,
                text="orphan",
                words=[_word(5.0, 6.0, "orphan")],
            )
        ],
    )
    diarization = Diarization(turns=[DiarizationTurn(start=0.0, end=1.0, speaker="S1")])

    utterances = assign_speakers(transcript, diarization)
    assert utterances[0].speaker == UNKNOWN_SPEAKER
    # UNKNOWN must not count as a detected speaker.
    assert detected_speakers(utterances) == []


def test_multiple_speaker_overlap_is_noted() -> None:
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="overlap",
                words=[_word(0.0, 2.0, "overlap")],
            )
        ],
    )
    diarization = Diarization(
        turns=[
            DiarizationTurn(start=0.0, end=1.3, speaker="S1"),
            DiarizationTurn(start=0.7, end=2.0, speaker="S2"),  # 0.7s of overlap
        ]
    )

    utterances = assign_speakers(transcript, diarization)
    assert utterances[0].speaker == "S1"
    assert any("overlaps multiple speakers" in note for note in utterances[0].notes)


def test_diarization_turn_without_text_produces_no_utterance() -> None:
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0, end=1.0, text="hello", words=[_word(0.0, 1.0, "hello")]
            )
        ],
    )
    diarization = Diarization(
        turns=[
            DiarizationTurn(start=0.0, end=1.0, speaker="S1"),
            DiarizationTurn(start=4.0, end=9.0, speaker="S2"),  # silent turn
        ]
    )

    utterances = assign_speakers(transcript, diarization)
    assert len(utterances) == 1
    assert utterances[0].speaker == "S1"


def test_adjacent_same_speaker_words_are_merged() -> None:
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="a b",
                words=[_word(0.0, 1.0, "a"), _word(1.0, 2.0, "b")],
            )
        ],
    )
    diarization = Diarization(turns=[DiarizationTurn(start=0.0, end=2.0, speaker="S1")])

    utterances = assign_speakers(transcript, diarization)
    assert len(utterances) == 1
    assert utterances[0].text == "a b"


def test_words_in_one_asr_segment_survive_a_long_internal_pause() -> None:
    """Regression: "Chào" and "cô" were emitted on separate lines.

    Whisper placed them in one segment but 4.4s apart, which exceeded the merge
    gap. Words from the same ASR segment are one phrase and must stay together.
    """
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.4,
                end=6.74,
                text="Chào cô",
                words=[_word(0.4, 1.84, "Chào"), _word(6.22, 6.74, "cô")],
            )
        ],
    )
    diarization = Diarization(turns=[DiarizationTurn(start=0.0, end=8.0, speaker="S1")])

    utterances = assign_speakers(transcript, diarization)
    assert len(utterances) == 1
    assert utterances[0].text == "Chào cô"


def test_different_segments_with_a_long_gap_are_not_merged() -> None:
    """The same-segment rule must not collapse genuinely separate turns."""
    transcript = Transcript(
        language="vi",
        segments=[
            TranscriptSegment(
                start=0.0, end=1.0, text="one", words=[_word(0.0, 1.0, "one")]
            ),
            TranscriptSegment(
                start=9.0, end=10.0, text="two", words=[_word(9.0, 10.0, "two")]
            ),
        ],
    )
    diarization = Diarization(turns=[DiarizationTurn(start=0.0, end=10.0, speaker="S1")])

    utterances = assign_speakers(transcript, diarization)
    assert len(utterances) == 2
