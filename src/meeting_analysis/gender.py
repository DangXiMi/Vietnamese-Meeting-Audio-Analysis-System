"""Per-speaker gender classification with a median-F0 cross-check.

Gender is classified **once per diarized speaker** from that speaker's pooled
audio, then applied to every one of their utterances. Per-utterance
classification is unstable and produces transcripts where a person changes
gender mid-meeting.

Gender is physiological rather than language-specific, so a non-Vietnamese
audio classifier is acceptable here; the median-F0 estimate (male ~85-180 Hz,
female ~165-255 Hz) acts as an independent sanity check.
"""

from __future__ import annotations

import logging

import numpy as np

from .models import Gender, GenderPrediction
from .utils.gpu import free_cuda

logger = logging.getLogger(__name__)

F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
MALE_F0_MAX_HZ = 180.0
FEMALE_F0_MIN_HZ = 165.0


def estimate_median_f0(
    audio: np.ndarray, sample_rate: int, max_seconds: float = 10.0
) -> float | None:
    """Median F0 over voiced frames, via FFT autocorrelation.

    Deliberately avoids ``librosa.yin``: it is a pure-numpy implementation that
    takes minutes over pooled speaker audio and dominated the whole run. Frames
    whose autocorrelation peak is weak are treated as unvoiced.
    """
    if audio.size < sample_rate // 2:
        return None

    segment = np.asarray(audio[: int(max_seconds * sample_rate)], dtype=np.float64)
    frame_length = 1024
    hop_length = 512
    if segment.size < frame_length:
        return None

    frames = np.lib.stride_tricks.sliding_window_view(segment, frame_length)[::hop_length]
    if frames.shape[0] < 5:
        return None

    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = frames * np.hanning(frame_length)

    n_fft = 1 << int(np.ceil(np.log2(2 * frame_length)))
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=1)

    tau_min = max(1, int(sample_rate / F0_MAX_HZ))
    tau_max = min(autocorr.shape[1] - 1, int(sample_rate / F0_MIN_HZ))
    if tau_max <= tau_min:
        return None

    window = autocorr[:, tau_min : tau_max + 1]
    energy = autocorr[:, 0:1].copy()
    energy[energy <= 0] = 1e-12
    normalized = window / energy

    best = np.argmax(normalized, axis=1)
    strength = normalized[np.arange(normalized.shape[0]), best]

    voiced = strength > 0.3
    if int(voiced.sum()) < 5:
        return None

    taus = tau_min + best[voiced]
    f0_values = sample_rate / taus
    f0_values = f0_values[(f0_values > F0_MIN_HZ) & (f0_values < F0_MAX_HZ)]
    if f0_values.size < 5:
        return None
    return float(np.median(f0_values))


def _gender_from_f0(f0: float) -> Gender:
    """Threshold the pitch estimate.

    Adult male speech sits around 85-180 Hz and female around 165-255 Hz. The
    165-180 Hz band is genuinely ambiguous, so 165 Hz is used as the boundary.
    """
    return Gender.FEMALE if f0 >= FEMALE_F0_MIN_HZ else Gender.MALE


def _map_label(label: str) -> Gender | None:
    """Map model-native labels to the deliverable labels.

    Checked most-specific first: "female" contains the substring "male".
    """
    text = label.strip().lower()
    if "female" in text or "woman" in text or text in {"f", "nữ", "nu", "female_1"}:
        return Gender.FEMALE
    if "male" in text or "man" in text or text in {"m", "nam", "male_1"}:
        return Gender.MALE
    return None


class Wav2Vec2GenderClassifier:
    """wav2vec2 audio classifier with an F0 fallback and disagreement flag."""

    def __init__(
        self,
        model_name: str = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = None
        self._failed = False

    def _ensure_pipeline(self):
        if self._pipe is None and not self._failed:
            try:
                from transformers import pipeline

                self._pipe = pipeline(
                    "audio-classification", model=self.model_name, device=self.device
                )
                logger.info("Loaded gender model '%s'", self.model_name)
            except Exception:
                logger.warning(
                    "Gender model '%s' unavailable; falling back to F0 only",
                    self.model_name,
                    exc_info=True,
                )
                self._failed = True
        return self._pipe

    def classify(self, audio: np.ndarray, sample_rate: int) -> GenderPrediction:
        f0 = estimate_median_f0(audio, sample_rate)
        f0_gender = _gender_from_f0(f0) if f0 is not None else None

        model_gender: Gender | None = None
        model_confidence = 0.0
        raw_label = None

        pipe = self._ensure_pipeline()
        if pipe is not None and audio.size:
            try:
                preds = pipe(
                    {"array": audio.astype(np.float32), "sampling_rate": sample_rate},
                    top_k=2,
                )
                for item in preds:
                    mapped = _map_label(str(item.get("label", "")))
                    if mapped is not None:
                        model_gender = mapped
                        model_confidence = float(item.get("score", 0.0))
                        raw_label = str(item.get("label"))
                        break
            except Exception:
                logger.warning("Gender model inference failed", exc_info=True)

        # Resolve, preferring the model but always cross-checking with F0.
        if model_gender is not None and f0_gender is not None:
            if model_gender == f0_gender:
                return GenderPrediction(
                    gender=model_gender,
                    confidence=model_confidence,
                    f0_median=f0,
                    source="model+f0",
                )
            return GenderPrediction(
                gender=model_gender,
                confidence=model_confidence,
                f0_median=f0,
                source="model",
                note=(
                    f"model says {model_gender.value} but median F0 {f0:.1f} Hz "
                    f"suggests {f0_gender.value}; review manually"
                ),
            )

        if model_gender is not None:
            return GenderPrediction(
                gender=model_gender,
                confidence=model_confidence,
                f0_median=f0,
                source="model",
            )

        if f0_gender is not None:
            return GenderPrediction(
                gender=f0_gender,
                confidence=0.5,
                f0_median=f0,
                source="f0-fallback",
                note=f"gender model unavailable or uninterpretable ({raw_label})",
            )

        logger.warning("Gender undetermined; defaulting to %s", Gender.MALE.value)
        return GenderPrediction(
            gender=Gender.MALE,
            confidence=0.0,
            f0_median=None,
            source="default",
            note="insufficient audio to determine gender",
        )

    def unload(self) -> None:
        self._pipe = None
        free_cuda()
