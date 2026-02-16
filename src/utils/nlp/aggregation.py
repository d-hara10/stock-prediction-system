from typing import Dict, Any, List
import numpy as np
from collections import Counter


def calculate_weighted_score(
    enriched_articles: List[Dict[str, Any]]
) -> float:
    """
    Weighted sentiment score in [-1, 1].
    """
    if not enriched_articles:
        return 0.0

    numerator = sum(
        a["sentiment_value"] * a["confidence"] * a["time_weight"]
        for a in enriched_articles
    )
    denominator = sum(a["time_weight"] for a in enriched_articles)

    return float(numerator / denominator) if denominator > 0 else 0.0


def determine_signal_strength(
    distribution: Dict[str, int],
    weighted_score: float,
    avg_confidence: float
) -> str:
    """
    Classify signal strength using direction, dominance, and confidence.
    
    Thresholds:
    - strong: |score| >= 0.55
    - moderate: |score| >= 0.30
    - weak: |score| >= 0.10
    - mixed: |score| < 0.10
    
    Can upgrade moderate→strong if 70%+ distribution and 65%+ confidence.
    """
    total = sum(distribution.values())
    if total == 0:
        return "mixed"

    pos = distribution["positive"] / total
    neg = distribution["negative"] / total

    direction = "positive" if weighted_score > 0 else "negative"
    magnitude = abs(weighted_score)

    if magnitude < 0.10:
        return "mixed"

    tier = "weak"
    if magnitude >= 0.30:
        tier = "moderate"
    if magnitude >= 0.55:
        tier = "strong"

    # Upgrade to strong if dominant + confident
    if tier != "strong":
        if direction == "positive" and pos >= 0.70 and avg_confidence >= 0.65:
            tier = "strong"
        if direction == "negative" and neg >= 0.70 and avg_confidence >= 0.65:
            tier = "strong"

    return f"{tier}_{direction}"