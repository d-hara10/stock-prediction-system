from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.pipelines.volatility_pipeline import run_pipeline
from src.pipelines.nlp_pipeline import run_nlp_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
# Create FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    description="Combined volatility forecasting and sentiment analysis for stocks",
    version="1.0"
)

# Add rate limiter to app
app.state.limiter = limiter  # ← NEW
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # ← NEW

# CORS middleware - allows frontend to call this API
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """
    Health check endpoint.
    Returns API status and version.
    """
    return {
        "status": "online",
        "version": "1.0",
        "endpoints": {
            "predict": "/predict/{ticker}",
            "docs": "/docs"
        }
    }

    
@app.get("/predict/{ticker}")
@limiter.limit("10/minute")
async def predict(request: Request, ticker: str):
    """
    Get volatility forecast + sentiment analysis for a stock ticker.

    Rate limit: 10 requests per minute per IP address.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., AAPL, TSLA)
    
    Returns:
        JSON with volatility forecast and sentiment analysis
    
    Example:
        GET /predict/AAPL
    """
    ticker = ticker.upper().strip()
    
    if not ticker or len(ticker) > 10 or not ticker.isalpha():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker format: {ticker}. Ticker must be 1-10 alphabetic characters."
        )
    
    start_time = time.time()
    logger.info(f"Processing prediction request for {ticker}")
    
    try:
        # Run volatility pipeline
        logger.info(f"Running volatility model for {ticker}")
        _, _, vol_context = run_pipeline(ticker)
        
        # Run sentiment pipeline
        logger.info(f"Running sentiment analysis for {ticker}")
        sentiment = run_nlp_pipeline(ticker)
        
        if "error" in sentiment:
            logger.warning(f"Sentiment analysis failed for {ticker}: {sentiment['error']}")
        
        response = {
            "ticker": ticker,
            "timestamp": sentiment.get("timestamp"),
            "volatility": vol_context,
            "sentiment": sentiment
        }
        
        elapsed = time.time() - start_time
        logger.info(f"{ticker} prediction completed in {elapsed:.2f}s")
        
        return response
        
    except ValueError as e:
        error_msg = str(e).lower()
        
        # Improved detection for invalid tickers
        if any(phrase in error_msg for phrase in [
            "no data found",
            "invalid",
            "ticker",
            "n_samples=0",
            "empty",
            "train set will be empty",
            "number of samples",
            "number of folds"
        ]):
            logger.warning(f"Invalid ticker requested: {ticker}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ticker: {ticker}. No data found for this symbol."
            )
        else:
            logger.error(f"ValueError in prediction for {ticker}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Data processing error: {str(e)}"
            )
    
    except Exception as e:
        logger.error(f"Unexpected error processing {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing {ticker}. Please try again later."
        )


@app.get("/health")
async def health_check():
    """
    Detailed health check - tests if pipelines can be imported.
    """
    try:
        from src.pipelines.volatility_pipeline import run_pipeline
        from src.pipelines.nlp_pipeline import run_nlp_pipeline
        
        return {
            "status": "healthy",
            "pipelines": {
                "volatility": "loaded",
                "sentiment": "loaded"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }