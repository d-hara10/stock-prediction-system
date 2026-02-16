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
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
            Stock Prediction System
          </h1>
          <p className="text-slate-400 text-lg">
            AI-powered volatility forecasting and sentiment analysis
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
              className="flex-1 px-6 py-4 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg font-semibold text-lg transition-colors"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="mb-8 p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-slate-400">Fetching data and generating analysis...</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Volatility Card */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-slate-300">
                📊 Next-Day Volatility Forecast
              </h2>
              <div className="flex items-baseline gap-4">
                <p className="text-4xl font-bold text-blue-400">
                  {formatVolatility(result.volatility.predicted)}
                </p>
                <p className="text-lg text-slate-400">
                  Confidence: <span className="text-white font-semibold capitalize">{result.volatility.confidence}</span>
                </p>
              </div>
            </div>

            {/* Sentiment Card */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-slate-300">
                📰 News Sentiment
              </h2>
              <p className="text-4xl font-bold text-purple-400">
                {formatSignalStrength(result.sentiment.signal_strength)}
              </p>
            </div>

            {/* AI Analysis Card */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-slate-300">
                🤖 AI Recommendation
              </h2>
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-6 mb-3" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-5 mb-2" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-lg font-bold mt-4 mb-2" {...props} />,
                    p: ({node, ...props}) => <p className="mb-4 text-slate-200" {...props} />,
                    strong: ({node, ...props}) => <strong className="font-bold text-white" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc list-inside mb-4 space-y-1" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-4 space-y-1" {...props} />,
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