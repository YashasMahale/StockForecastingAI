import pandas as pd


def create_features(df):

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["lag_1"] = df["price"].shift(1)
    df["lag_7"] = df["price"].shift(7)

    df["rolling_7"] = df["price"].rolling(7).mean()
    df["rolling_30"] = df["price"].rolling(30).mean()

    df.dropna(inplace=True)

    return df