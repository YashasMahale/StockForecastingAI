import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker="AAPL", period="2y"):

    print("Downloading stock data...")

    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True
    )

    print("Raw rows:", len(df))

    df = df.reset_index()

    df = df[["Date", "Close"]]

    df.rename(columns={
        "Date": "date",
        "Close": "price"
    }, inplace=True)

    df.dropna(inplace=True)

    return df