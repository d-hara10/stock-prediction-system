from typing import Dict, Any
from transformers import pipeline

# Setup
_FINBERT = None

def get_finbert():
    """Lazy-load FinBERT once per process."""
    global _FINBERT
    if _FINBERT is None:
        _FINBERT = pipeline(
            "text-classification",
            model="ProsusAI/finbert"
        )
    return _FINBERT


LABEL_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Run FinBERT sentiment analysis.
    """
    finbert = get_finbert()
    out = finbert(text, truncation=True, max_length=512)[0]

    label = str(out.get("label", "")).lower()
    if label not in LABEL_MAP:
        label = "neutral"

    confidence = float(out.get("score", 0.0))

    return {
        "sentiment": label,
        "confidence": confidence,
        "sentiment_value": LABEL_MAP[label]
    }