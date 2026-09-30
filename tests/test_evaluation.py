"""Known-answer tests for the evaluation metrics.

Every expected value here is computed by hand, so the harness itself is
verified before it is trusted to report accuracy numbers.
"""

from __future__ import annotations

import pytest

from meeting_analysis.evaluation import (
    DER_FRAME_SECONDS,
    Segment,
    character_error_rate,
    diarization_error_rate,
    evaluate,
    labelling_template,
    match_segments,
    normalize_text,
    speaker_mapping_from_pairs,
    word_error_rate,
)


def _seg(start: float, end: float, speaker: str, **kwargs) -> Segment:
    return Segment(
        start=start,
        end=end,
        speaker=speaker,
        gender=kwargs.get("gender"),
        emotion=kwargs.get("emotion"),
        text=kwargs.get("text", ""),
    )


# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #
def test_normalize_preserves_diacritics_and_drops_punctuation() -> None:
    # Diacritics are semantically meaningful in Vietnamese and must survive.
    assert normalize_text("Chào cô, anh tên là gì?") == "chào cô anh tên là gì"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("  chào   cô  ") == "chào cô"


# --------------------------------------------------------------------------- #
# WER / CER
# --------------------------------------------------------------------------- #
def test_wer_identical_is_zero() -> None:
    score = word_error_rate("chào cô", "chào cô")
    assert score.rate == 0.0
    assert score.reference_units == 2


def test_wer_single_substitution() -> None:
    # ref: 2 syllables, 1 substituted -> 1/2
    score = word_error_rate("chào cô", "chào anh")
    assert score.rate == pytest.approx(0.5)
    assert (score.substitutions, score.deletions, score.insertions) == (1, 0, 0)


def test_wer_single_deletion() -> None:
    score = word_error_rate("chào cô", "chào")
    assert score.rate == pytest.approx(0.5)
    assert (score.substitutions, score.deletions, score.insertions) == (0, 1, 0)


def test_wer_single_insertion() -> None:
    # ref: 1 syllable, 1 inserted -> 1/1
    score = word_error_rate("chào", "chào cô")
    assert score.rate == pytest.approx(1.0)
    assert (score.substitutions, score.deletions, score.insertions) == (0, 0, 1)


def test_wer_empty_reference_is_zero() -> None:
    assert word_error_rate("", "").rate == 0.0


def test_wer_punctuation_and_case_are_ignored() -> None:
    assert word_error_rate("Chào cô!", "chào cô").rate == 0.0


def test_cer_character_level() -> None:
    # "abc" vs "abd": 1 substitution over 3 characters
    assert character_error_rate("abc", "abd").rate == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Diarization error rate
# --------------------------------------------------------------------------- #
def test_der_identical_is_zero() -> None:
    ref = [_seg(0.0, 10.0, "A")]
    score = diarization_error_rate(ref, [_seg(0.0, 10.0, "A")])
    assert score.der == pytest.approx(0.0)
    assert score.reference_speech_seconds == pytest.approx(10.0, abs=DER_FRAME_SECONDS)


def test_der_all_missed_when_hypothesis_empty() -> None:
    ref = [_seg(0.0, 10.0, "A")]
    score = diarization_error_rate(ref, [])
    assert score.der == pytest.approx(1.0)
    assert score.missed_seconds == pytest.approx(10.0, abs=0.05)


def test_der_is_zero_when_only_speaker_names_differ() -> None:
    """A system must not be penalised for labelling speakers differently."""
    ref = [_seg(0.0, 10.0, "A"), _seg(10.0, 20.0, "B")]
    hyp = [_seg(0.0, 10.0, "SPEAKER_00"), _seg(10.0, 20.0, "SPEAKER_01")]
    score = diarization_error_rate(ref, hyp)
    assert score.der == pytest.approx(0.0)
    assert score.mapping == {"SPEAKER_00": "A", "SPEAKER_01": "B"}


def test_der_confusion_when_two_reference_speakers_merge_into_one() -> None:
    # Two reference speakers (20 s of speech) collapse into a single predicted
    # speaker, so at best half the speech can be matched.
    ref = [_seg(0.0, 10.0, "A"), _seg(10.0, 20.0, "B")]
    hyp = [_seg(0.0, 20.0, "X")]
    score = diarization_error_rate(ref, hyp)
    assert score.der == pytest.approx(0.5, abs=0.02)
    assert score.confusion_seconds == pytest.approx(10.0, abs=0.05)


def test_der_counts_false_alarm() -> None:
    ref = [_seg(0.0, 10.0, "A")]
    hyp = [_seg(0.0, 10.0, "A"), _seg(10.0, 20.0, "A")]  # 10 s of extra speech
    score = diarization_error_rate(ref, hyp)
    assert score.false_alarm_seconds == pytest.approx(10.0, abs=0.05)
    assert score.der == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------- #
# Segment matching
# --------------------------------------------------------------------------- #
def test_match_segments_is_one_to_one_by_overlap() -> None:
    ref = [_seg(0.0, 2.0, "A"), _seg(2.0, 4.0, "B")]
    hyp = [_seg(0.1, 2.1, "A"), _seg(2.2, 4.2, "B")]
    pairs = match_segments(ref, hyp)
    assert pairs == [(0, 0), (1, 1)]


def test_match_segments_ignores_non_overlapping() -> None:
    ref = [_seg(0.0, 2.0, "A")]
    hyp = [_seg(50.0, 52.0, "A")]
    assert match_segments(ref, hyp) == []


def test_speaker_mapping_resolves_relabelled_speakers() -> None:
    ref = [_seg(0.0, 2.0, "A"), _seg(2.0, 4.0, "B")]
    hyp = [_seg(0.0, 2.0, "S1"), _seg(2.0, 4.0, "S2")]
    assert speaker_mapping_from_pairs(ref, hyp, [(0, 0), (1, 1)]) == {"S1": "A", "S2": "B"}


# --------------------------------------------------------------------------- #
# End-to-end scoring
# --------------------------------------------------------------------------- #
def _reference() -> list[dict]:
    return [
        {"start": 0.0, "end": 2.0, "speaker": "A", "gender": "Nam",
         "emotion": "Vui vẻ", "text": "chào cô"},
        {"start": 2.0, "end": 4.0, "speaker": "B", "gender": "Nữ",
         "emotion": "Bình thường", "text": "chào anh"},
    ]


def _prediction() -> list[dict]:
    # Same text and timings; speakers relabelled; one emotion wrong.
    return [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "gender": "Nam",
         "emotion": "Vui vẻ", "text": "chào cô"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01", "gender": "Nữ",
         "emotion": "Nóng giận", "text": "chào anh"},
    ]


def test_evaluate_perfect_text_and_diarization() -> None:
    report = evaluate(_reference(), _prediction())
    assert report.n_reference == 2
    assert report.matched == 2
    assert report.wer.rate == 0.0
    assert report.der.der == pytest.approx(0.0)


def test_evaluate_speaker_accuracy_uses_optimal_mapping() -> None:
    report = evaluate(_reference(), _prediction())
    # SPEAKER_00->A and SPEAKER_01->B, so both matched segments are correct.
    assert report.speaker_accuracy == pytest.approx(1.0)


def test_evaluate_gender_accuracy() -> None:
    report = evaluate(_reference(), _prediction())
    assert report.gender_accuracy == pytest.approx(1.0)


def test_evaluate_emotion_accuracy_and_confusion() -> None:
    report = evaluate(_reference(), _prediction())
    # One of the two emotions matches.
    assert report.emotion_accuracy == pytest.approx(0.5)
    assert report.emotion_confusion["Bình thường"]["Nóng giận"] == 1


def test_evaluate_detects_wrong_text() -> None:
    prediction = _prediction()
    prediction[1]["text"] = "hoàn toàn khác"  # 3 wrong syllables vs 2 reference
    report = evaluate(_reference(), prediction)
    assert report.wer.rate > 0.0


def test_evaluate_report_serialises() -> None:
    payload = evaluate(_reference(), _prediction()).to_dict()
    assert set(payload) == {
        "segments", "asr", "diarization",
        "speaker_attribution_accuracy", "gender_accuracy",
        "emotion_accuracy", "emotion_confusion",
    }
    assert payload["asr"]["wer"]["rate"] == 0.0


def test_evaluate_markdown_renders() -> None:
    markdown = evaluate(_reference(), _prediction()).to_markdown()
    assert "| **WER** (syllable) |" in markdown
    assert "Emotion confusion" in markdown


# --------------------------------------------------------------------------- #
# Labelling helper
# --------------------------------------------------------------------------- #
def test_labelling_template_mirrors_prediction() -> None:
    template = labelling_template(_prediction())
    assert len(template) == 2
    assert set(template[0]) == {"start", "end", "speaker", "gender", "emotion", "text"}
    assert template[0]["speaker"] == "SPEAKER_00"
