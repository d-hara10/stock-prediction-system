import os
import json
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
BASE_MODEL_DIR = SRC_DIR / "models" / "volatility"
BASE_MODEL_DIR = str(BASE_MODEL_DIR)

def get_paths(ticker, root=BASE_MODEL_DIR):
    """Get directory and file paths for a ticker."""
    ticker_dir = os.path.join(root, ticker)
    params_path = os.path.join(ticker_dir, "params.json")
    return ticker_dir, params_path

def save_model_params(ticker, best_params, feature_list, r2=None, mae=None, root=BASE_MODEL_DIR):
    """
    Save hyperparameters and metadata (not the trained model).
    
    Args:
        ticker: Stock ticker symbol
        best_params: Dictionary of hyperparameters from RandomizedSearchCV
        feature_list: List of feature names used
        r2: R² score from validation
        mae: Mean absolute error from validation
        root: Root directory for model storage
    """
    ticker_dir, params_path = get_paths(ticker, root)
    os.makedirs(ticker_dir, exist_ok=True)

    metadata = {
        "ticker": ticker,
        "trained_on": datetime.now().isoformat(),
        "features": feature_list,
        "model_version": "v1.0",
        "best_params": best_params,
        "r2_score": r2,
        "mae": mae
    }

    with open(params_path, "w") as f:
        json.dump(metadata, f, indent=4)

def load_model_params(ticker, root=BASE_MODEL_DIR):
    """
    Load hyperparameters and metadata.
    
    Returns:
        metadata (dict): Contains best_params, r2_score, mae, etc.
        or None if not found
    """
    ticker_dir, params_path = get_paths(ticker, root)

    if not os.path.exists(params_path):
        return None

    with open(params_path, "r") as f:
        metadata = json.load(f)
    
    return metadata

def is_model_stale(metadata: dict, max_age_days: int = 7) -> bool:
    """
    Returns True if hyperparameters are older than max_age_days.
    
    Args:
        metadata: Dictionary with 'trained_on' timestamp
        max_age_days: Maximum age in days before retraining
    """
    trained_on_str = metadata.get("trained_on")
    if trained_on_str is None:
        return True  # No trained date = treat as stale
    
    trained_on = datetime.fromisoformat(trained_on_str).date()
    today = datetime.today().date()
    age = (today - trained_on).days
    
    return age > max_age_days