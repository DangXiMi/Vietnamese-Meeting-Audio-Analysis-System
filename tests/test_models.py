"""Contract tests for the output models — no ML dependencies required."""

from __future__ import annotations

import json

from meeting_analysis.models import AnalysisResult, Emotion, Gender, Utterance


def _result() -> AnalysisResult:
    return AnalysisResult(
        utterances=[
            Utterance(
                start=0.031,
                end=2.845,
                speaker="A",
                gender=Gender.MALE,
                emotion=Emotion.HAPPY,
                text="Em làm phần này xong rồi nhé.",
            ),
            Utterance(
                start=3.102,
                end=5.44,
                speaker="B",
                gender=Gender.FEMALE,
                emotion=Emotion.ANGRY,
                text="Tại sao tiến độ lại chậm như vậy?",
            ),
        ],
        num_speakers=2,
    )


def test_label_values_match_assignment() -> None:
    assert Gender.MALE.value == "Nam"
    assert Gender.FEMALE.value == "Nữ"
    assert Emotion.HAPPY.value == "Vui vẻ"
    assert Emotion.ANGRY.value == "Nóng giận"
    assert Emotion.NEUTRAL.value == "Bình thường"


def test_to_text_matches_required_format() -> None:
    lines = _result().to_text().splitlines()
    assert lines[0] == '[Nam A - Vui vẻ]: "Em làm phần này xong rồi nhé."'
    assert lines[1] == '[Nữ B - Nóng giận]: "Tại sao tiến độ lại chậm như vậy?"'


def test_to_json_shape_and_keys() -> None:
    payload = _result().to_json()
    assert isinstance(payload, list)
    assert set(payload[0]) == {"start", "end", "speaker", "gender", "emotion", "text"}
    assert payload[1]["speaker"] == "B"
    assert payload[1]["gender"] == "Nữ"
    assert payload[1]["emotion"] == "Nóng giận"


def test_json_and_text_agree() -> None:
    """Both artifacts must describe the same segments, in the same order."""
    result = _result()
    from_json = json.loads(json.dumps(result.to_json(), ensure_ascii=False))
    txt_lines = result.to_text().splitlines()

    assert len(from_json) == len(txt_lines)
    for entry, line in zip(from_json, txt_lines):
        assert f"[{entry['gender']} {entry['speaker']} - {entry['emotion']}]" in line
        assert entry["text"] in line


def test_diacritics_survive_json_roundtrip() -> None:
    dumped = json.dumps(_result().to_json(), ensure_ascii=False)
    restored = json.loads(dumped)
    assert restored[1]["gender"] == "Nữ"
    assert restored[1]["emotion"] == "Nóng giận"


def test_rejects_unknown_label() -> None:
    import pytest

    with pytest.raises(ValueError):
        Utterance(
            start=0.0,
            end=1.0,
            speaker="A",
            gender="Male",  # type: ignore[arg-type]
            emotion=Emotion.NEUTRAL,
            text="x",
        )
