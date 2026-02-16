from src.pipelines.nlp_pipeline import run_nlp_pipeline
import json

if __name__ == "__main__":
    ticker = "AAPL"

    report = run_nlp_pipeline(
        ticker=ticker,
        time_window_hours=72,
        decay_hours=48,
        max_articles_to_analyze=25
    )

    print(json.dumps(report, indent=2))