"""Per-utterance emotion from Vietnamese text (PhoBERT).

The assignment permits audio- or text-based emotion; Vietnamese text is far
more reliable than prosody, so text is the primary signal here.
"""

from __future__ import annotations

import logging

from .models import Emotion, EmotionPrediction
from .utils.gpu import free_cuda

logger = logging.getLogger(__name__)

# PhoBERT sentiment label -> deliverable label. Model-native labels are never
# emitted in the final output.
LABEL_MAP = {
    "POS": Emotion.HAPPY,
    "NEG": Emotion.ANGRY,
    "NEU": Emotion.NEUTRAL,
}


def _map_label(label: str) -> Emotion | None:
    text = label.strip().upper()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    if text.startswith("POS") or "POSITIVE" in text:
        return Emotion.HAPPY
    if text.startswith("NEG") or "NEGATIVE" in text:
        return Emotion.ANGRY
    if text.startswith("NEU") or "NEUTRAL" in text:
        return Emotion.NEUTRAL
    return None


class PhoBertEmotionAnalyzer:
    """Vietnamese sentiment via ``wonrax/phobert-base-vietnamese-sentiment``."""

    def __init__(
        self,
        model_name: str = "wonrax/phobert-base-vietnamese-sentiment",
        device: str = "cpu",
        use_word_segmentation: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_word_segmentation = use_word_segmentation
        self._tokenizer = None
        self._model = None
        self._segmenter = None
        self._segmenter_checked = False

    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading emotion model '%s'", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                self._model.to("cuda")
            else:
                self.device = "cpu"
        return self._tokenizer, self._model

    def _segment(self, text: str) -> str:
        """Apply Vietnamese word segmentation when ``pyvi`` is available."""
        if not self.use_word_segmentation:
            return text
        if not self._segmenter_checked:
            self._segmenter_checked = True
            try:
                from pyvi import ViTokenizer

                self._segmenter = ViTokenizer.tokenize
            except Exception:
                logger.info("pyvi unavailable; using unsegmented text for PhoBERT")
                self._segmenter = None
        return self._segmenter(text) if self._segmenter else text

    def analyze(self, text: str) -> EmotionPrediction:
        cleaned = " ".join(text.split())
        if not cleaned:
            return EmotionPrediction(emotion=Emotion.NEUTRAL, confidence=0.0)

        tokenizer, model = self._ensure_model()

        import torch

        inputs = tokenizer(
            self._segment(cleaned),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        best_index = int(torch.argmax(probs))
        score = float(probs[best_index])
        raw_label = str(model.config.id2label.get(best_index, best_index))
        emotion = _map_label(raw_label) or Emotion.NEUTRAL

        return EmotionPrediction(emotion=emotion, confidence=score, raw_label=raw_label)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        free_cuda()
