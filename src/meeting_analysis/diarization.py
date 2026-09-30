"""pyannote.audio diarization adapter (implements :class:`interfaces.Diarizer`).

The waveform is loaded with ``soundfile`` rather than handed to pyannote as a
path: this machine's ``torchaudio`` is a CPU build, and passing an explicit
tensor avoids depending on its IO backends entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import Diarization, DiarizationTurn
from .utils.gpu import free_cuda

logger = logging.getLogger(__name__)

# Signature of the cuDNN failure described in PyannoteDiarizer._ensure_pipeline.
_ENGINE_ERROR = "unable to find an engine"


def _is_cudnn_engine_error(exc: BaseException) -> bool:
    return _ENGINE_ERROR in str(exc).lower()


class PyannoteDiarizer:
    """Speaker diarization via ``pyannote/speaker-diarization-3.1``.

    Requires a Hugging Face token with the license accepted for BOTH the
    diarization pipeline and its ``segmentation-3.0`` dependency.
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        token: str | None = None,
        num_speakers: int | None = None,
        batch_size: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.token = token
        self.num_speakers = num_speakers
        self.batch_size = batch_size
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline

            logger.info("Loading diarization pipeline '%s'", self.model)
            try:
                pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.token)
            except TypeError:
                # Newer pyannote renamed the argument to `token`.
                pipeline = Pipeline.from_pretrained(self.model, token=self.token)
            if self.device == "cuda" and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                # cuDNN's default heuristic algorithm selection can fail to find
                # *any* convolution algorithm for the wespeaker ResNet, raising
                # "GET was unable to find an engine to execute this computation"
                # deep inside the embedding forward pass. Benchmark mode makes
                # cuDNN actually search for a working algorithm.
                torch.backends.cudnn.benchmark = True
                # Opt-in only (DIARIZATION_BATCH_SIZE). Lowering the batch size
                # reduced memory but made diarization dramatically slower on
                # this 4 GB card, so pyannote's own default is kept unless the
                # caller explicitly asks otherwise.
                if self.batch_size > 0:
                    for attribute in ("embedding_batch_size", "segmentation_batch_size"):
                        current = getattr(pipeline, attribute, None)
                        if isinstance(current, int) and current > self.batch_size:
                            setattr(pipeline, attribute, self.batch_size)
                            logger.info(
                                "Capped %s: %d -> %d (requested)",
                                attribute,
                                current,
                                self.batch_size,
                            )
            self._pipeline = pipeline
        return self._pipeline

    @staticmethod
    def _load_waveform(audio_path: Path):
        import soundfile as sf
        import torch

        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (channels, samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return {"waveform": waveform, "sample_rate": sample_rate}

    @staticmethod
    def _to_annotation(result):
        """Normalise across pyannote versions (Annotation vs DiarizeOutput)."""
        if hasattr(result, "itertracks"):
            return result
        for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
            candidate = getattr(result, attr, None)
            if candidate is not None and hasattr(candidate, "itertracks"):
                return candidate
        if isinstance(result, tuple) and result and hasattr(result[0], "itertracks"):
            return result[0]
        raise TypeError(f"Unsupported diarization result type: {type(result)!r}")

    def diarize(self, audio_path: Path) -> Diarization:
        pipeline = self._ensure_pipeline()
        payload = self._load_waveform(audio_path)

        kwargs = {}
        if self.num_speakers:
            kwargs["num_speakers"] = self.num_speakers

        try:
            raw = pipeline(payload, **kwargs)
        except RuntimeError as exc:
            if not _is_cudnn_engine_error(exc):
                raise
            # Last resort: PyTorch's native convolution implementation. Slower
            # per operator, but it is always available and it keeps diarization
            # working instead of failing the whole run.
            import torch

            logger.warning(
                "cuDNN could not supply a convolution algorithm (%s); "
                "retrying with cuDNN disabled.",
                str(exc)[:90],
            )
            torch.backends.cudnn.enabled = False
            raw = pipeline(payload, **kwargs)

        annotation = self._to_annotation(raw)

        turns = [
            DiarizationTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))

        result = Diarization(turns=turns)
        logger.info(
            "Diarization found %d turn(s) across %d speaker(s)",
            len(turns),
            result.num_speakers,
        )
        return result

    def unload(self) -> None:
        self._pipeline = None
        free_cuda()
