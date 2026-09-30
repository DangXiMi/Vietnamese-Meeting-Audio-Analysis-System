"""Evaluation metrics for measuring pipeline quality against labelled data.

Implements the standard metrics for this task rather than relying on eyeballing:

* **WER / CER** — ASR accuracy (syllable-level for Vietnamese, since Vietnamese
  orthography separates syllables with spaces).
* **DER** — diarization error rate, decomposed into missed speech, false alarm,
  and speaker confusion, with an **optimal** hypothesis-to-reference speaker
  mapping.
* **Speaker attribution accuracy** — correct speaker on matched segments.
* **Gender / emotion accuracy** — label agreement on matched segments.

The reference file uses **exactly the same JSON shape as the pipeline output**,
so producing a labelled reference is just editing a copy of a result file::

    python -m meeting_analysis.evaluate --prediction out.json --write-template ref.json

Reference and prediction are matched segment-to-segment by maximum temporal
overlap (one-to-one), which tolerates the small boundary differences that any
two segmentations will have.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:  # scipy is already a project dependency; keep a graceful fallback anyway
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Frame resolution for the diarization error rate. 10 ms matches the
# conventional scoring granularity used by NIST md-eval.
DER_FRAME_SECONDS = 0.01

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Text normalisation and edit distance
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, and collapse whitespace.

    Diacritics are deliberately **preserved** — in Vietnamese they are
    semantically meaningful, and treating ``cô`` and ``co`` as equal would
    overstate accuracy.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = _PUNCT.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Tokenise into Vietnamese syllables (whitespace-delimited)."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def edit_operations(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[int, int, int]:
    """Levenshtein backtrace, returning ``(substitutions, deletions, insertions)``."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0

    # dp[i][j] = edit distance between reference[:i] and hypothesis[:j]
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ref_i = reference[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_i == hypothesis[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost,   # match / substitute
                dp[i - 1, j] + 1,          # deletion
                dp[i, j - 1] + 1,          # insertion
            )

    i, j = n, m
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] and reference[i - 1] == hypothesis[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions


@dataclass
class ErrorRateScore:
    """Shared shape for WER and CER."""

    rate: float
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def correct(self) -> int:
        return self.reference_units - self.substitutions - self.deletions

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": round(self.rate, 4),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_units": self.reference_units,
            "correct": self.correct,
        }


def _error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorRateScore:
    substitutions, deletions, insertions = edit_operations(reference, hypothesis)
    total = len(reference)
    rate = (substitutions + deletions + insertions) / total if total else 0.0
    return ErrorRateScore(rate, substitutions, deletions, insertions, total)


def word_error_rate(reference_text: str, hypothesis_text: str) -> ErrorRateScore:
    """Syllable-level WER (appropriate for Vietnamese)."""
    return _error_rate(tokens(reference_text), tokens(hypothesis_text))


def character_error_rate(reference_text: str, hypothesis_text: str) -> ErrorRateScore:
    """Character-level error rate over the normalised strings."""
    ref = normalize_text(reference_text).replace(" ", "")
    hyp = normalize_text(hypothesis_text).replace(" ", "")
    return _error_rate(list(ref), list(hyp))


# --------------------------------------------------------------------------- #
# Segment records
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """A minimal view over either a reference or a prediction segment."""

    start: float
    end: float
    speaker: str
    gender: str | None
    emotion: str | None
    text: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Segment":
        return cls(
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            speaker=str(raw.get("speaker", "")),
            gender=raw.get("gender"),
            emotion=raw.get("emotion"),
            text=str(raw.get("text", "")),
        )


def load_segments(payload: Any) -> list[Segment]:
    """Accept either a bare array or a ``{"segments": [...]}`` wrapper."""
    if isinstance(payload, dict):
        payload = payload.get("segments") or payload.get("utterances") or []
    return [Segment.from_dict(item) for item in payload]


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# --------------------------------------------------------------------------- #
# Diarization error rate
# --------------------------------------------------------------------------- #
@dataclass
class DiarizationScore:
    der: float
    missed_seconds: float
    false_alarm_seconds: float
    confusion_seconds: float
    reference_speech_seconds: float
    mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "der": round(self.der, 4),
            "missed_seconds": round(self.missed_seconds, 3),
            "false_alarm_seconds": round(self.false_alarm_seconds, 3),
            "confusion_seconds": round(self.confusion_seconds, 3),
            "reference_speech_seconds": round(self.reference_speech_seconds, 3),
            "speaker_mapping": self.mapping,
        }


def _frame_speaker_ids(
    segments: Sequence[Segment], speakers: list[str], n_frames: int
) -> np.ndarray:
    """Rasterise segments onto a fixed-frame grid; -1 means 'no speech'."""
    index = {speaker: i for i, speaker in enumerate(speakers)}
    labels = np.full(n_frames, -1, dtype=np.int32)
    for segment in segments:
        start = max(0, int(round(segment.start / DER_FRAME_SECONDS)))
        end = min(n_frames, int(round(segment.end / DER_FRAME_SECONDS)))
        if end > start and segment.speaker in index:
            labels[start:end] = index[segment.speaker]
    return labels


def diarization_error_rate(
    reference: Sequence[Segment], hypothesis: Sequence[Segment]
) -> DiarizationScore:
    """Frame-based DER with an optimal speaker mapping.

    ``DER = (missed + false_alarm + confusion) / reference_speech``.

    Speaker confusion is computed after finding the assignment of hypothesis
    speakers to reference speakers that minimises confusion, so a system is not
    penalised merely for naming speakers differently.
    """
    ref_speakers = sorted({s.speaker for s in reference if s.speaker})
    hyp_speakers = sorted({s.speaker for s in hypothesis if s.speaker})
    if not ref_speakers:
        return DiarizationScore(0.0, 0.0, 0.0, 0.0, 0.0, {})

    duration = max(
        [s.end for s in reference] + [s.end for s in hypothesis] + [0.0]
    )
    n_frames = max(1, int(round(duration / DER_FRAME_SECONDS)))
    frame_seconds = n_frames * DER_FRAME_SECONDS

    ref_ids = _frame_speaker_ids(reference, ref_speakers, n_frames)
    hyp_ids = _frame_speaker_ids(hypothesis, hyp_speakers, n_frames)

    ref_speech = ref_ids >= 0
    hyp_speech = hyp_ids >= 0

    missed = int(np.sum(ref_speech & ~hyp_speech))
    false_alarm = int(np.sum(hyp_speech & ~ref_speech))

    # Confusion: both speaking, but the mapped speakers disagree.
    mapping: dict[str, str] = {}
    confusion = 0
    both = ref_speech & hyp_speech
    if both.any() and hyp_speakers:
        matrix = np.zeros((len(ref_speakers), len(hyp_speakers)), dtype=np.int64)
        ref_both = ref_ids[both]
        hyp_both = hyp_ids[both]
        np.add.at(matrix, (ref_both, hyp_both), 1)

        if _HAVE_SCIPY:
            rows, cols = linear_sum_assignment(-matrix)
            assignment = list(zip(rows.tolist(), cols.tolist()))
        else:  # pragma: no cover - greedy fallback
            order = np.dstack(np.unravel_index(np.argsort(-matrix, axis=None), matrix.shape))[0]
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            assignment = []
            for r, c in order:
                if r not in used_rows and c not in used_cols:
                    used_rows.add(int(r))
                    used_cols.add(int(c))
                    assignment.append((int(r), int(c)))

        matched = 0
        for r, c in assignment:
            mapping[hyp_speakers[c]] = ref_speakers[r]
            matched += int(matrix[r, c])
        confusion = int(both.sum()) - matched
    else:
        for speaker in hyp_speakers:
            mapping[speaker] = speaker

    total_ref = int(ref_speech.sum())
    denom = total_ref * DER_FRAME_SECONDS
    der = ((missed + false_alarm + confusion) * DER_FRAME_SECONDS / denom) if denom else 0.0

    return DiarizationScore(
        der=der,
        missed_seconds=missed * DER_FRAME_SECONDS,
        false_alarm_seconds=false_alarm * DER_FRAME_SECONDS,
        confusion_seconds=confusion * DER_FRAME_SECONDS,
        reference_speech_seconds=denom,
        mapping=mapping,
    )


# --------------------------------------------------------------------------- #
# Segment matching and per-segment metrics
# --------------------------------------------------------------------------- #
def match_segments(
    reference: Sequence[Segment], hypothesis: Sequence[Segment]
) -> list[tuple[int, int]]:
    """One-to-one pairing by descending temporal overlap (greedy)."""
    candidates = [
        (overlap(r.start, r.end, h.start, h.end), ri, hi)
        for ri, r in enumerate(reference)
        for hi, h in enumerate(hypothesis)
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)

    used_ref: set[int] = set()
    used_hyp: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ov, ri, hi in candidates:
        if ov <= 0.0 or ri in used_ref or hi in used_hyp:
            continue
        used_ref.add(ri)
        used_hyp.add(hi)
        pairs.append((ri, hi))
    return sorted(pairs)


def speaker_mapping_from_pairs(
    reference: Sequence[Segment],
    hypothesis: Sequence[Segment],
    pairs: Sequence[tuple[int, int]],
) -> dict[str, str]:
    """Optimal hypothesis->reference speaker mapping over matched segments."""
    ref_speakers = sorted({s.speaker for s in reference if s.speaker})
    hyp_speakers = sorted({s.speaker for s in hypothesis if s.speaker})
    if not ref_speakers or not hyp_speakers:
        return {}

    ref_index = {s: i for i, s in enumerate(ref_speakers)}
    hyp_index = {s: i for i, s in enumerate(hyp_speakers)}
    matrix = np.zeros((len(ref_speakers), len(hyp_speakers)), dtype=np.int64)
    for ri, hi in pairs:
        rs, hs = reference[ri].speaker, hypothesis[hi].speaker
        if rs in ref_index and hs in hyp_index:
            matrix[ref_index[rs], hyp_index[hs]] += 1

    mapping: dict[str, str] = {}
    if _HAVE_SCIPY:
        rows, cols = linear_sum_assignment(-matrix)
        for r, c in zip(rows.tolist(), cols.tolist()):
            mapping[hyp_speakers[c]] = ref_speakers[r]
    else:  # pragma: no cover
        for hyp_speaker in hyp_speakers:
            mapping[hyp_speaker] = hyp_speaker
    return mapping


@dataclass
class EvaluationReport:
    n_reference: int
    n_hypothesis: int
    matched: int
    wer: ErrorRateScore
    cer: ErrorRateScore
    der: DiarizationScore
    speaker_accuracy: float
    gender_accuracy: float | None
    emotion_accuracy: float | None
    emotion_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    reference_speakers: list[str] = field(default_factory=list)
    hypothesis_speakers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": {
                "reference": self.n_reference,
                "hypothesis": self.n_hypothesis,
                "matched": self.matched,
                "reference_speakers": len(self.reference_speakers),
                "hypothesis_speakers": len(self.hypothesis_speakers),
            },
            "asr": {"wer": self.wer.to_dict(), "cer": self.cer.to_dict()},
            "diarization": self.der.to_dict(),
            "speaker_attribution_accuracy": round(self.speaker_accuracy, 4),
            "gender_accuracy": None if self.gender_accuracy is None else round(self.gender_accuracy, 4),
            "emotion_accuracy": None if self.emotion_accuracy is None else round(self.emotion_accuracy, 4),
            "emotion_confusion": self.emotion_confusion,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation report",
            "",
            "## Corpus",
            "",
            "| | |",
            "|---|---|",
            f"| Reference segments | {self.n_reference} |",
            f"| Predicted segments | {self.n_hypothesis} |",
            f"| Matched segments | {self.matched} |",
            f"| Reference speakers | {len(self.reference_speakers)} |",
            f"| Predicted speakers | {len(self.hypothesis_speakers)} |",
            "",
            "## ASR accuracy",
            "",
            "| Metric | Value | Sub | Del | Ins | Ref units |",
            "|---|---|---|---|---|---|",
            f"| **WER** (syllable) | **{self.wer.rate:.2%}** | {self.wer.substitutions} | "
            f"{self.wer.deletions} | {self.wer.insertions} | {self.wer.reference_units} |",
            f"| CER (character) | {self.cer.rate:.2%} | {self.cer.substitutions} | "
            f"{self.cer.deletions} | {self.cer.insertions} | {self.cer.reference_units} |",
            "",
            "## Diarization",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **DER** | **{self.der.der:.2%}** |",
            f"| Missed speech | {self.der.missed_seconds:.1f} s |",
            f"| False alarm | {self.der.false_alarm_seconds:.1f} s |",
            f"| Speaker confusion | {self.der.confusion_seconds:.1f} s |",
            f"| Reference speech | {self.der.reference_speech_seconds:.1f} s |",
            "",
            f"Speaker mapping (predicted → reference): `{self.der.mapping}`",
            "",
            "## Attribution and labels",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Speaker attribution accuracy | {self.speaker_accuracy:.2%} |",
            f"| Gender accuracy | {'n/a' if self.gender_accuracy is None else f'{self.gender_accuracy:.2%}'} |",
            f"| Emotion accuracy | {'n/a' if self.emotion_accuracy is None else f'{self.emotion_accuracy:.2%}'} |",
        ]

        if self.emotion_confusion:
            labels = sorted(self.emotion_confusion)
            lines += ["", "### Emotion confusion (reference → predicted)", ""]
            lines.append("| ref \\ pred | " + " | ".join(labels) + " |")
            lines.append("|" + "---|" * (len(labels) + 1))
            for ref_label in labels:
                row = self.emotion_confusion[ref_label]
                lines.append(
                    f"| **{ref_label}** | "
                    + " | ".join(str(row.get(pred_label, 0)) for pred_label in labels)
                    + " |"
                )

        return "\n".join(lines) + "\n"


def evaluate(reference_payload: Any, hypothesis_payload: Any) -> EvaluationReport:
    """Score a prediction against a labelled reference."""
    reference = load_segments(reference_payload)
    hypothesis = load_segments(hypothesis_payload)

    ref_text = " ".join(s.text for s in reference)
    hyp_text = " ".join(s.text for s in hypothesis)

    pairs = match_segments(reference, hypothesis)
    mapping = speaker_mapping_from_pairs(reference, hypothesis, pairs)

    matched = len(pairs)
    speaker_hits = 0
    gender_total = gender_hits = 0
    emotion_total = emotion_hits = 0
    confusion: dict[str, dict[str, int]] = {}

    for ri, hi in pairs:
        ref, hyp = reference[ri], hypothesis[hi]

        if mapping.get(hyp.speaker, hyp.speaker) == ref.speaker:
            speaker_hits += 1

        if ref.gender is not None and hyp.gender is not None:
            gender_total += 1
            # Gender is a property of the speaker, so compare under the mapping
            # only when genders are directly comparable labels.
            if ref.gender == hyp.gender:
                gender_hits += 1

        if ref.emotion is not None and hyp.emotion is not None:
            emotion_total += 1
            if ref.emotion == hyp.emotion:
                emotion_hits += 1
            confusion.setdefault(ref.emotion, {})
            confusion[ref.emotion][hyp.emotion] = confusion[ref.emotion].get(hyp.emotion, 0) + 1

    return EvaluationReport(
        n_reference=len(reference),
        n_hypothesis=len(hypothesis),
        matched=matched,
        wer=word_error_rate(ref_text, hyp_text),
        cer=character_error_rate(ref_text, hyp_text),
        der=diarization_error_rate(reference, hypothesis),
        speaker_accuracy=(speaker_hits / matched) if matched else 0.0,
        gender_accuracy=(gender_hits / gender_total) if gender_total else None,
        emotion_accuracy=(emotion_hits / emotion_total) if emotion_total else None,
        emotion_confusion=confusion,
        reference_speakers=sorted({s.speaker for s in reference if s.speaker}),
        hypothesis_speakers=sorted({s.speaker for s in hypothesis if s.speaker}),
    )


def labelling_template(hypothesis_payload: Any) -> list[dict[str, Any]]:
    """Build a reference skeleton from a prediction, ready for hand-correction.

    Speaker letters and timings are pre-filled so a labeller only has to fix the
    transcript text and the three label fields.
    """
    return [
        {
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "speaker": s.speaker,
            "gender": s.gender,
            "emotion": s.emotion,
            "text": s.text,
        }
        for s in load_segments(hypothesis_payload)
    ]
