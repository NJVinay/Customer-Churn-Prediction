import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "backend/models/churn_model.joblib")


class ChurnRequest(BaseModel):
    gender: str = Field(..., examples=["Female"])
    senior_citizen: int = Field(..., examples=[0])
    partner: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[12])
    phone_service: str = Field(..., examples=["Yes"])
    multiple_lines: str = Field(..., examples=["No"])
    internet_service: str = Field(..., examples=["Fiber optic"])
    online_security: str = Field(..., examples=["No"])
    online_backup: str = Field(..., examples=["Yes"])
    device_protection: str = Field(..., examples=["No"])
    tech_support: str = Field(..., examples=["No"])
    streaming_tv: str = Field(..., examples=["Yes"])
    streaming_movies: str = Field(..., examples=["No"])
    contract: str = Field(..., examples=["Month-to-month"])
    paperless_billing: str = Field(..., examples=["Yes"])
    payment_method: str = Field(..., examples=["Electronic check"])
    monthly_charges: float = Field(..., examples=[70.35])
    total_charges: float = Field(..., examples=[1397.47])


def payload_to_frame(payload: ChurnRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gender": payload.gender,
                "SeniorCitizen": payload.senior_citizen,
                "Partner": payload.partner,
                "Dependents": payload.dependents,
                "tenure": payload.tenure,
                "PhoneService": payload.phone_service,
                "MultipleLines": payload.multiple_lines,
                "InternetService": payload.internet_service,
                "OnlineSecurity": payload.online_security,
                "OnlineBackup": payload.online_backup,
                "DeviceProtection": payload.device_protection,
                "TechSupport": payload.tech_support,
                "StreamingTV": payload.streaming_tv,
                "StreamingMovies": payload.streaming_movies,
                "Contract": payload.contract,
                "PaperlessBilling": payload.paperless_billing,
                "PaymentMethod": payload.payment_method,
                "MonthlyCharges": payload.monthly_charges,
                "TotalCharges": payload.total_charges,
            }
        ]
    )


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run backend/train_model.py first."
        )
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {"model": payload, "threshold": 0.5, "metrics": None}


app = FastAPI(title="Customer Churn Prediction API")

origins = os.getenv("CORS_ORIGINS", "*")
if origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    try:
        payload = load_model()
        return payload.get("metrics") or {}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(payload: ChurnRequest):
    try:
        model_payload = load_model()
        model = model_payload["model"]
        threshold = float(model_payload.get("threshold", 0.5))
        features = payload_to_frame(payload)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(features)[0][1])
        prediction = int(proba >= threshold) if proba is not None else int(
            model.predict(features)[0]
        )
        return {
            "churn_label": prediction,
            "churn_probability": proba,
            "threshold": threshold,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
