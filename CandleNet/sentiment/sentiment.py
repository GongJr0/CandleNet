import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading


def polarity(probs: tuple) -> float:
    return probs[0] - probs[1]


def distil(text: str) -> str:
    text_split = text.lower().split()
    text_distil = " ".join([w.strip() for w in text_split if len(w) > 1])
    return text_distil


class Sentiment:
    def __init__(self, model_id="yiyanghkust/finbert-tone", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # map labels robustly
        self._label2id = {v: k for k, v in self.model.config.id2label.items()}
        self._ipos = self._label2id["Positive"]
        self._ineg = self._label2id["Negative"]
        self._ineu = self._label2id["Neutral"]

    def _tokenize(self, texts):
        # Truncation is not handled specially; may implement chunking later.
        return self.tok(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

    def _logits(self, batch, T: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(**batch).logits
        if T != 1.0:
            logits = logits / T
        return logits

    def _polarity(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return probs[:, self._ipos] - probs[:, self._ineg]  # [-1,1]-ish scalar per row

    def pipe_sentiment(self, texts: str | list[str], T: float = 1.0) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [distil(t or "") for t in texts]
        batch = self._tokenize(texts)
        logits = self._logits(batch, T)
        s = self._polarity(logits)
        return s.detach().cpu().numpy()


_lock = threading.Lock()
_sentiment_obj = None


def get_sentiment_obj():
    global _sentiment_obj
    if _sentiment_obj is None:
        with _lock:
            if _sentiment_obj is None:
                _sentiment_obj = Sentiment()
    return _sentiment_obj
