# Stock Prediction System

AI-powered stock analysis combining machine learning volatility forecasting, sentiment analysis, and LLM-generated investment recommendations.

## Features

- **Volatility Forecasting**: Random Forest model predicts next-day volatility using 15 technical indicators
- **Sentiment Analysis**: FinBERT analyzes recent news headlines with time-weighted aggregation  
- **AI Recommendations**: NVIDIA Nemotron generates actionable investment insights
- **Dockerized**: Full-stack application runs anywhere with one command

## Tech Stack

**Backend**: Python, FastAPI, scikit-learn, Transformers (FinBERT), yfinance  
**Frontend**: Next.js, TypeScript, Tailwind CSS, React Markdown  
**AI**: OpenRouter API (NVIDIA Nemotron)  
**Deployment**: Docker, Docker Compose  

## Quick Start

### Prerequisites
- Docker Desktop
- OpenRouter API key ([get free key](https://openrouter.ai/))

### Setup

1. Clone repository
2. Create `.env` file at project root:
```bash
FRONTEND_URL=http://localhost:3000
OPENROUTER_API_KEY=your_key_here
BACKEND_URL=http://backend:8000
```

3. Run with Docker:
```bash
docker compose up --build
```

4. Open http://localhost:3000

## Architecture

- FastAPI backend handles volatility prediction and sentiment analysis
- Next.js frontend provides interactive UI and LLM integration
- Docker containers ensure consistent deployment across environments
- Rate limiting (10 req/min) and CORS protection for security

## Model Details

**Volatility Model**:
- Random Forest regressor with 15 technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- Trained on 2 years of historical data
- RandomizedSearchCV with TimeSeriesSplit for hyperparameter tuning
- Auto-retrains every 7 days

**Sentiment Pipeline**:
- FinBERT (ProsusAI) for financial sentiment classification
- Exponential time decay for recency weighting (72-hour window)
- Aggregates into signal strength (strong positive → strong negative)

## Security

- Environment variables for sensitive credentials
- CORS protection and rate limiting
- Input validation on all endpoints
- Docker best practices (no secrets in images)

## License

MIT

---

**Portfolio project demonstrating full-stack ML engineering, NLP, containerization, and production best practices**