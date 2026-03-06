from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict import predict_future

app = FastAPI()


class PredictRequest(BaseModel):
    ticker: str
    days: int


@app.get("/")
def root():
    return {"message": "Stock Forecast API running"}


@app.post("/predict")
def predict(req: PredictRequest):

    try:
        preds = predict_future(req.ticker.upper(), req.days)

        if len(preds) == 0:
            raise ValueError("Prediction returned empty list")

        return {
            "ticker": req.ticker.upper(),
            "days": req.days,
            "predictions": preds
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )