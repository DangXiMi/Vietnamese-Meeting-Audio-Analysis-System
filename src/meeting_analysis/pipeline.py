"""Pipeline orchestration.

Stages run strictly sequentially and each model-backed stage unloads before the
next loads: the 4 GB RTX 3050 cannot hold ASR and diarization at the same time.

Order: normalize -> diarize -> transcribe -> align -> gender -> emotion -> emit.
Diarization runs first because it is the cheaper stage to leave on CPU if VRAM
runs short.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .alignment import assign_speakers, detected_speakers
from .asr import FasterWhisperTranscriber
from .audio import normalize, probe, require_ffmpeg
from .config import Settings, get_hf_token
from .diarization import PyannoteDiarizer
from .emotion import PhoBertEmotionAnalyzer
from .gender import Wav2Vec2GenderClassifier
from .models import AnalysisResult, Gender, Utterance
from .utils.gpu import ensure_vram_headroom, vram_report

logger = logging.getLogger(__name__)


def _speaker_letters_in_order(speakers: list[str]) -> dict[str, str]:
    """Map detected speakers to A, B, C… by first appearance."""
    return {speaker: chr(ord("A") + index) for index, speaker in enumerate(speakers)}


def _pool_speaker_audio(
    wav_path: Path, spans: list[tuple[float, float]], max_seconds: float
) -> tuple[np.ndarray, int]:
    """Concatenate a speaker's turns into one array for a single classification."""
    import soundfile as sf

    data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    chunks: list[np.ndarray] = []
    collected = 0.0
    for start, end in spans:
        begin = max(0, int(start * sample_rate))
        finish = min(len(mono), int(end * sample_rate))
        if finish <= begin:
            continue
        chunks.append(mono[begin:finish])
        collected += (finish - begin) / sample_rate
        if collected >= max_seconds:
            break

    if not chunks:
        return np.zeros(0, dtype=np.float32), sample_rate
    return np.concatenate(chunks), sample_rate


def run_pipeline(
    audio_path: Path, out_dir: Path, settings: Settings | None = None
) -> AnalysisResult:
    """Run the full analysis and return the structured result."""
    settings = settings or Settings()
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_ffmpeg()
    device = settings.resolve_device()

    if device == "cuda":
        # Fail with an actionable message instead of letting cuDNN report a
        # misleading "unable to find an engine" deep inside a convolution.
        ensure_vram_headroom()

    # --- 1. probe + normalize (one canonical time base for all stages) --------
    info = probe(audio_path)
    logger.info(
        "Input: %.1fs, %d Hz, %d channel(s)", info.duration, info.sample_rate, info.channels
    )
    normalized = normalize(
        audio_path, out_dir / f"{audio_path.stem}.16k.wav", settings.sample_rate
    )

    token = get_hf_token()
    if not token:
        logger.warning("No HF token found; gated pyannote models will fail to load")

    # --- 2. diarization ------------------------------------------------------
    diarizer = PyannoteDiarizer(
        model=settings.diarization_model,
        device=device,
        token=token,
        num_speakers=settings.num_speakers,
        batch_size=settings.diarization_batch_size,
    )
    try:
        diarization = diarizer.diarize(normalized)
    finally:
        # Must run even when the stage raises: a stage that fails before
        # unloading leaks its VRAM for the lifetime of the process, which then
        # starves every later run in a long-lived host such as Streamlit.
        diarizer.unload()
    logger.info("After diarization: %s", vram_report())

    # --- 3. transcription ----------------------------------------------------
    transcriber = FasterWhisperTranscriber(
        model_size=settings.asr_model,
        device=device,
        compute_type=settings.asr_compute_type,
        language=settings.asr_language,
    )
    try:
        transcript = transcriber.transcribe(normalized)
    finally:
        transcriber.unload()
    logger.info("After ASR: %s", vram_report())

    # --- 4. align words to speakers -----------------------------------------
    merged = assign_speakers(transcript, diarization)

    # Only real speakers get a letter; the UNKNOWN bucket falls back to "?".
    order = detected_speakers(merged)
    letters = _speaker_letters_in_order(order)

    # --- 5. gender per speaker (pooled audio) --------------------------------
    gender_classifier = Wav2Vec2GenderClassifier(
        model_name=settings.gender_model, device=device
    )
    speaker_gender: dict = {}
    try:
        for speaker in order:
            spans = [
                (turn.start, turn.end)
                for turn in diarization.turns
                if turn.speaker == speaker
            ]
            pooled, sample_rate = _pool_speaker_audio(
                normalized, spans, settings.gender_pool_seconds
            )
            prediction = gender_classifier.classify(pooled, sample_rate)
            speaker_gender[speaker] = prediction
            logger.info(
                "Speaker %s -> %s (%.2f, %s%s)",
                speaker,
                prediction.gender.value,
                prediction.confidence,
                prediction.source,
                f", f0={prediction.f0_median:.1f}Hz" if prediction.f0_median else "",
            )
    finally:
        gender_classifier.unload()

    # --- 6. emotion per utterance (Vietnamese text) --------------------------
    emotion_analyzer = PhoBertEmotionAnalyzer(
        model_name=settings.emotion_model, device="cpu"
    )
    utterances: list[Utterance] = []
    try:
        for item in merged:
            emotion = emotion_analyzer.analyze(item.text)
            gender = speaker_gender.get(item.speaker)
            notes = list(item.notes)
            if gender is not None and gender.note:
                notes.append(gender.note)
            utterances.append(
                Utterance(
                    start=item.start,
                    end=item.end,
                    speaker=letters.get(item.speaker, "?"),
                    gender=gender.gender if gender is not None else Gender.MALE,
                    emotion=emotion.emotion,
                    text=item.text,
                    notes=notes,
                )
            )
    finally:
        emotion_analyzer.unload()

    result = AnalysisResult(
        utterances=utterances,
        num_speakers=len(order),
        metadata={
            "source": str(audio_path),
            "duration_seconds": round(info.duration, 3),
            "asr_model": settings.asr_model,
            "diarization_model": settings.diarization_model,
            "gender_model": settings.gender_model,
            "emotion_model": settings.emotion_model,
            "speaker_labels": letters,
            "speaker_gender_detail": {
                speaker: {
                    "gender": prediction.gender.value,
                    "confidence": round(prediction.confidence, 4),
                    "f0_median": prediction.f0_median,
                    "source": prediction.source,
                }
                for speaker, prediction in speaker_gender.items()
            },
        },
    )
    return result


def emit(result: AnalysisResult, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write the TXT and JSON artifacts from one in-memory result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"

    txt_path.write_text(result.to_text() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return txt_path, json_path
