from sklearn.ensemble import RandomForestRegressor

from src.utils.volatility.data_fetching import fetch_data
from src.utils.volatility.feature_engineering import compute_features
from src.utils.volatility.dataset_prep import prepare_X_y
from src.utils.volatility.model_io import load_model_params, save_model_params, is_model_stale
from src.utils.volatility.model_training import train_model

FEATURES = [
    "RollingVolatility", "ATR", "RollingMean", "Return",
    "RSI", "MACD", "MACD_Signal",
    "BB_Width",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3", "Return_Lag_5",
    "HV_10", "HV_20", "HV_30"
]

def run_pipeline(ticker):
    """
    Full volatility prediction pipeline.
    
    Strategy:
    - Run RandomizedSearchCV on ALL data (100%) with 5-fold TimeSeriesSplit
    - Calculate BOTH MAE and R² during cross-validation
    - Cache hyperparameters + metrics for 7 days
    - When cached: train new model on latest data using cached params
    - When stale: re-run RandomizedSearchCV to find new best params
    
    Returns: metadata, retrained, volatility_context
    """
    # Always fetch latest data
    data = fetch_data(ticker)
    df = compute_features(data)
    
    # Separate training data (rows with valid targets) from latest row (for prediction)
    df_train = df[df["TargetVolatility"].notna()].copy()
    X_train_full, y_train_full = prepare_X_y(df_train, FEATURES)
    
    # Check if we need to re-search hyperparameters
    metadata = load_model_params(ticker)
    retrained = False
    
    if metadata is None or is_model_stale(metadata):
        # Hyperparameters are missing or stale → Run RandomizedSearchCV
        retrained = True
        
        # Run RandomizedSearchCV on 100% of data
        # Returns: best model (already trained on 100%), mae, r2
        best_model, mae, r2 = train_model(X_train_full, y_train_full)
        
        # Extract best hyperparameters
        best_params = {
            'n_estimators': best_model.n_estimators,
            'max_depth': best_model.max_depth,
            'min_samples_split': best_model.min_samples_split,
            'min_samples_leaf': best_model.min_samples_leaf,
        }
        
        # Save hyperparameters and metrics
        save_model_params(ticker, best_params, FEATURES, r2=r2, mae=mae)
        
        # Use this model (already trained on 100%)
        model = best_model
        
        # Reload metadata
        metadata = load_model_params(ticker)
    else:
        # Hyperparameters are fresh → Use cached params
        best_params = metadata.get('best_params')
        r2 = metadata.get('r2_score', 0.5)
        mae = metadata.get('mae', 0.01)
        
        # Train NEW model on ALL latest data using cached params
        model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=123,
            n_jobs=-1
        )
        model.fit(X_train_full, y_train_full)
    
    # Predict tomorrow using today's features
    X_latest = df[FEATURES].iloc[[-1]]
    predicted_vol = float(model.predict(X_latest)[0])
    
    # Get current volatility (today's actual)
    current_vol = float(df["RollingVolatility"].iloc[-1])
    
    # Determine confidence based on R²
    if r2 > 0.7:
        confidence = "high"
    elif r2 > 0.3:
        confidence = "medium"
    else:
        confidence = "low"
    
    volatility_context = {
        "predicted": predicted_vol,
        "current": current_vol,
        "change_pct": round(((predicted_vol - current_vol) / current_vol) * 100, 1),
        "confidence": confidence,
        "r2_score": round(r2, 3),
        "mae": round(mae, 6)
    }
    
    return metadata, retrained, volatility_context
