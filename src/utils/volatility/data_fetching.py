import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetches price history for the given ticker over the past X years.
    Default: 2 years lookback.
    """
    if start is None:
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, group_by="column")

    # Flatten MultiIndex columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    return data