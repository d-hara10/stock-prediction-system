import numpy as np
import pandas as pd

def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    
    # Basic Returns
    data["Return"] = data["Close"].pct_change()

    for lag in [1, 2, 3, 5]:
        data[f"Return_Lag_{lag}"] = data["Return"].shift(lag)

    # RSI (Wilder's smoothing)
    delta = data["Close"].diff()

    data["gain"] = np.where(delta > 0, delta, 0)
    data["loss"] = np.where(delta < 0, -delta, 0)

    data["avg_gain"] = data["gain"].ewm(alpha=1/14, adjust=False).mean()
    data["avg_loss"] = data["loss"].ewm(alpha=1/14, adjust=False).mean()

    rs = data["avg_gain"] / (data["avg_loss"] + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    data.drop(columns=["gain", "loss", "avg_gain", "avg_loss"], inplace=True)

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands

    sma20 = data["Close"].rolling(window=20).mean()
    std20 = data["Close"].rolling(window=20).std()

    data["BB_Upper"] = sma20 + 2 * std20
    data["BB_Lower"] = sma20 - 2 * std20
    data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]

    # Rolling & Historical Volatility
    data["RollingVolatility"] = data["Return"].rolling(window=20).std()

    for window in [10, 20, 30]:
        data[f"HV_{window}"] = data["Return"].rolling(window).std()

    # ATR (Wilder optional)
    data["High-Low"] = data["High"] - data["Low"]
    data["High-Close"] = np.abs(data["High"] - data["Close"].shift())
    data["Low-Close"] = np.abs(data["Low"] - data["Close"].shift())

    data["TR"] = data[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
    data["ATR"] = data["TR"].ewm(alpha=1/14, adjust=False).mean()

    data.drop(columns=["High-Low", "High-Close", "Low-Close", "TR"], inplace=True)

    # Momentum
    data["RollingMean"] = data["Return"].rolling(window=10).mean()

    # Target Volatility
    data["TargetVolatility"] = data["RollingVolatility"].shift(-1)

    # Drop warmup rows (from technical indicators)
    # But keep last row even though TargetVolatility is NaN
    data = data.dropna(subset=[col for col in data.columns if col != "TargetVolatility"])

    return data