"""faster-whisper ASR adapter (implements :class:`interfaces.Transcriber`)."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import MODELS_CACHE_DIR
from .models import Transcript, TranscriptSegment, WordTiming
from .utils.gpu import free_cuda, register_cuda_dlls

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """Vietnamese ASR via faster-whisper (CTranslate2).

    ``language`` is forced by default rather than auto-detected: short
    Vietnamese turns frequently misdetect as English/Chinese, and detection
    costs an extra pass. Pass ``language=None`` (or ``--language auto``) for
    mixed Vietnamese/English recordings, where forcing one language mangles
    the other into nonsense syllables.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str | None = "vi",
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            register_cuda_dlls()
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper '%s' (device=%s, compute_type=%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(MODELS_CACHE_DIR),
                )
            except Exception:
                if self.device != "cuda":
                    raise
                logger.warning(
                    "CUDA load failed for faster-whisper; retrying on CPU (int8)", exc_info=True
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODELS_CACHE_DIR),
                )
        return self._model

    def transcribe(self, audio_path: Path) -> Transcript:
        model = self._ensure_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,          # drops silence and improves timestamps
            word_timestamps=True,
            beam_size=self.beam_size,
        )

        collected: list[TranscriptSegment] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            words = [
                WordTiming(start=float(w.start), end=float(w.end), word=w.word)
                for w in (segment.words or [])
                if w.start is not None and w.end is not None
            ]
            collected.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    words=words,
                    confidence=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                )
            )

        logger.info("Transcribed %d segment(s)", len(collected))
        detected = getattr(info, "language", None) or self.language or "unknown"
        return Transcript(language=detected, segments=collected)

    def unload(self) -> None:
        self._model = None
        free_cuda()
