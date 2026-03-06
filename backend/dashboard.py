import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

st.title("Stock Price Forecast Dashboard")

ticker = st.text_input("Enter Stock Ticker", "META")
days = st.slider("Prediction Days", 1, 30, 7)

if st.button("Predict"):

    data = yf.download(ticker, period="2y")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    history_prices = data["Close"].tail(60)
    history_dates = history_prices.index

    url = "http://127.0.0.1:8000/predict"

    response = requests.post(
        url,
        json={
            "ticker": ticker,
            "days": days
        }
    )

    if response.status_code != 200:
        st.error("Prediction API failed")
        st.stop()

    preds = response.json()["predictions"]

    future_expected = [p["expected"] for p in preds]
    future_lower = [p["lower"] for p in preds]
    future_upper = [p["upper"] for p in preds]

    future_dates = pd.date_range(
        start=history_dates[-1],
        periods=days + 1
    )[1:]

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        history_dates,
        history_prices,
        label="Historical Price",
        color="blue"
    )

    ax.plot(
        future_dates,
        future_expected,
        label="Prediction",
        color="red"
    )

    ax.fill_between(
        future_dates,
        future_lower,
        future_upper,
        alpha=0.25,
        color="orange",
        label="Confidence Interval"
    )

    ax.set_title(f"{ticker} Stock Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.legend()

    st.pyplot(fig)