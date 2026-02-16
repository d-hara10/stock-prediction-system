import datetime as dt
from typing import Dict, Any, List
import numpy as np
import feedparser
import requests

def fetch_news(
    ticker: str,
    time_window_hours: float = 72.0
) -> List[Dict[str, Any]]:
    """
    Fetch recent news articles from Google News RSS for "{TICKER} stock".
    """
    from urllib.parse import quote_plus
    
    query = quote_plus(f"{ticker} stock")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        # Use requests with User-Agent (Google requires this)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the fetched content with feedparser
        feed = feedparser.parse(response.content)
        
    except Exception as e:
        raise RuntimeError(f"RSS fetch failed: {e}")

    if not feed.entries:
        return []

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=time_window_hours)

    articles = []

    for entry in feed.entries:
        published = None

        # Use published_parsed (this is what Google News RSS provides)
        if getattr(entry, "published_parsed", None):
            try:
                published = dt.datetime(
                    *entry.published_parsed[:6],
                    tzinfo=dt.timezone.utc
                )
            except Exception:
                pass

        if published is None or published < cutoff:
            continue

        age_hours = (now - published).total_seconds() / 3600

        articles.append({
            "title": entry.title,
            "published": published,
            "age_hours": age_hours
        })

    # Most recent first
    articles.sort(key=lambda a: a["published"], reverse=True)
    return articles


def calculate_time_weight(
    published_date: dt.datetime,
    decay_hours: float = 48.0
) -> float:
    """
    Exponential decay weighting by recency.
    """
    now = dt.datetime.now(dt.timezone.utc)
    age_hours = (now - published_date).total_seconds() / 3600
    return float(np.exp(-age_hours / decay_hours))