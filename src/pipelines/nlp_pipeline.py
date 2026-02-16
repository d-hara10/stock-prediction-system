import datetime as dt
from typing import Dict, Any
from collections import Counter
import numpy as np

from src.utils.nlp.fetching import fetch_news, calculate_time_weight
from src.utils.nlp.analysis import analyze_sentiment
from src.utils.nlp.aggregation import calculate_weighted_score, determine_signal_strength


def run_nlp_pipeline(
    ticker: str,
    time_window_hours: float = 72.0,
    decay_hours: float = 48.0,
    max_articles_to_analyze: int = 25
) -> Dict[str, Any]:
    """
    End-to-end NLP sentiment pipeline.
    Fetches news, analyzes sentiment, returns report.
    """
    now = dt.datetime.now(dt.timezone.utc)

    try:
        articles = fetch_news(ticker, time_window_hours)
    except RuntimeError as e:
        return {
            "ticker": ticker,
            "timestamp": now.isoformat(),
            "error": str(e),
            "articles_analyzed": 0
        }

    articles = articles[:max_articles_to_analyze]

    enriched = []
    for a in articles:
        sentiment = analyze_sentiment(a["title"])
        time_weight = calculate_time_weight(a["published"], decay_hours)

        enriched.append({
            **a,
            **sentiment,
            "time_weight": time_weight
        })

    dist = Counter(e["sentiment"] for e in enriched)
    sentiment_distribution = {
        "positive": dist.get("positive", 0),
        "neutral": dist.get("neutral", 0),
        "negative": dist.get("negative", 0),
    }

    weighted_score = calculate_weighted_score(enriched)
    avg_confidence = (
        float(np.mean([e["confidence"] for e in enriched]))
        if enriched else 0.0
    )

    signal = determine_signal_strength(
        sentiment_distribution,
        weighted_score,
        avg_confidence
    )

    return {
        "ticker": ticker,
        "timestamp": now.isoformat(),
        "time_window_hours": time_window_hours,
        "articles_analyzed": len(enriched),
        "sentiment_distribution": sentiment_distribution,
        "weighted_sentiment_score": weighted_score,
        "average_confidence": avg_confidence,
        "signal_strength": signal,
        "context_headlines": [
            {
                "title": e["title"],
                "published": e["published"].isoformat(),
                "sentiment": e["sentiment"],
                "confidence": e["confidence"]
            }
            for e in enriched[:5]
        ]
    }