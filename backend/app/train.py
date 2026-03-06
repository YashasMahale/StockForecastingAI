import os
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .data_fetcher import fetch_stock_data
from .features import create_features


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
MLRUNS_PATH = os.path.join(BASE_DIR, "mlruns")


# =========================
# MLflow setup
# =========================

mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
mlflow.set_experiment("stock_forecasting")


def train_model(ticker="AAPL"):

    print("Starting training pipeline...")

    # =========================
    # Fetch Data
    # =========================

    print("Fetching stock data from Yahoo Finance...")
    df = fetch_stock_data(ticker)

    # Fix MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("Rows downloaded:", len(df))

    # =========================
    # Feature Engineering
    # =========================

    print("Creating features...")
    df = create_features(df)

    print("Rows after feature engineering:", len(df))

    print("Columns:", df.columns)

    # =========================
    # Target column detection
    # =========================

    target_column = "price"

    print("Using target column:", target_column)

    # =========================
    # Prepare Dataset
    # =========================

    X = df.drop(["price", "date"], axis=1)
    y = df["price"]

    # =========================
    # Time Series Cross Validation
    # =========================

    tscv = TimeSeriesSplit(n_splits=5)

    maes = []
    rmses = []

    print("\nStarting cross validation...\n")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        with mlflow.start_run(run_name=f"fold_{fold+1}"):

            model.fit(X_train, y_train)

            preds = model.predict(X_val)

            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))

            maes.append(mae)
            rmses.append(rmse)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)

            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_param("max_depth", 5)

            mlflow.xgboost.log_model(model, name="model")

            print(f"Fold {fold+1} → MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # =========================
    # Average Metrics
    # =========================

    avg_mae = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))

    print("\nTraining complete")
    print("Average MAE:", avg_mae)
    print("Average RMSE:", avg_rmse)

    # =========================
    # Save Metrics
    # =========================

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    metrics = {
        "average_mae": avg_mae,
        "average_rmse": avg_rmse
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    print("Metrics saved.")

    # =========================
    # Train Final Model
    # =========================

    final_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_PATH)

    print("Final model saved.")


if __name__ == "__main__":
    train_model()