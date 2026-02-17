'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface VolatilityData {
  predicted: number;
  current: number;
  change_pct: number;
  confidence: string;
  r2_score: number;
  mae: number;
}

interface SentimentData {
  ticker: string;
  timestamp: string;
  time_window_hours: number;
  articles_analyzed: number;
  sentiment_distribution: {
    positive: number;
    neutral: number;
    negative: number;
  };
  weighted_sentiment_score: number;
  average_confidence: number;
  signal_strength: string;
  context_headlines: Array<{
    title: string;
    published: string;
    sentiment: string;
    confidence: number;
  }>;
}

interface AnalysisResponse {
  ticker: string;
  timestamp: string;
  volatility: VolatilityData;
  sentiment: SentimentData;
  analysis: string;
}

export default function Home() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker.toUpperCase() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze stock');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const formatVolatility = (vol: number) => `${(vol * 100).toFixed(2)}%`;

  const formatSignalStrength = (signal: string) => {
    const mapping: { [key: string]: string } = {
      'strong_positive': 'Very Positive 🤩',
      'moderate_positive': 'Positive 😊',
      'weak_positive': 'Slightly Positive 🙂',
      'mixed': 'Neutral 😐',
      'weak_negative': 'Slightly Negative 🙁',
      'moderate_negative': 'Negative 😞',
      'strong_negative': 'Very Negative 😱'
    };
    return mapping[signal] || signal;
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-white p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-2 text-emerald-400">
            Stock Prediction System
          </h1>
          <p className="text-zinc-400 text-lg">
            AI-Powered Volatility Forecasting and Sentiment Analysis
          </p>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-4">
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="Enter ticker (e.g., AAPL)"
              maxLength={10}
              className="flex-1 px-6 py-4 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-lg transition-all"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="px-8 py-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-800 disabled:cursor-not-allowed rounded-lg font-semibold text-lg transition-colors"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="mb-8 p-4 bg-red-950/50 border border-red-900 rounded-lg text-red-300">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500 mb-4"></div>
            <p className="text-zinc-400">Fetching data and generating analysis...</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Volatility Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors">
              <h2 className="text-xl font-bold mb-4 text-zinc-100">
                📊 Next-Day Volatility Forecast
              </h2>
              <div className="flex items-baseline gap-4">
                <p className="text-4xl font-bold text-emerald-400">
                  {formatVolatility(result.volatility.predicted)}
                </p>
                <p className="text-lg text-zinc-400">
                  Confidence: <span className="text-emerald-400 font-semibold capitalize">{result.volatility.confidence}</span>
                </p>
              </div>
            </div>

            {/* Sentiment Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors">
              <h2 className="text-xl font-bold mb-4 text-zinc-100">
                📰 News Sentiment
              </h2>
              <p className="text-4xl font-bold text-emerald-400">
                {formatSignalStrength(result.sentiment.signal_strength)}
              </p>
            </div>

            {/* AI Analysis Card */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors">
              <h2 className="text-xl font-bold mb-4 text-zinc-100">
                🤖 AI Recommendation
              </h2>
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-6 mb-3 text-emerald-400" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-5 mb-2 text-emerald-400" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-lg font-bold mt-4 mb-2 text-zinc-200" {...props} />,
                    p: ({node, ...props}) => <p className="mb-4 text-zinc-300 leading-relaxed" {...props} />,
                    strong: ({node, ...props}) => <strong className="font-bold text-emerald-400" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc list-inside mb-4 space-y-1 text-zinc-300" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-4 space-y-1 text-zinc-300" {...props} />,
                  }}
                >
                  {result.analysis}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}