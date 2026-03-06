import numpy as np
import pandas as pd

from .data_fetcher import fetch_stock_data


def predict_future(ticker: str, days: int):

    df = fetch_stock_data(ticker)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    price_col = None
    for c in ["close", "price", "adj close"]:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        raise ValueError("Close price column not found in stock data")

    prices = df[price_col].values

    if len(prices) < 30:
        raise ValueError("Not enough historical data")

    returns = np.diff(prices) / prices[:-1]

    mu = returns.mean()
    sigma = returns.std()

    last_price = prices[-1]

    forecast = []

    current_price = last_price

    for i in range(days):

        noise = np.random.normal(0, sigma * 0.6)

        daily_return = mu + noise

        next_price = current_price * (1 + daily_return)

        band = sigma * np.sqrt(i + 1) * current_price

        forecast.append({
            "expected": float(next_price),
            "lower": float(next_price - band),
            "upper": float(next_price + band)
        })

        current_price = next_price

    return forecast