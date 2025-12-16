import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

# Import Pydantic models
from src.api.pydantic_models import (
    TransactionFeatures,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResult,
    HealthCheck,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage ML model"""

    def __init__(self):
        self.model = None
        self.model_version = "unknown"
        self.model_name = "credit_risk_model"
        self.loaded_at = None
        self.feature_columns = None

    def load_model(self):
        """Load the model from MLflow or local file"""
        try:
            try:
                import mlflow
                import mlflow.sklearn

                mlflow.set_tracking_uri("file:../mlruns")
                client = mlflow.tracking.MlflowClient()
                model_name = "credit_risk_best_model"

                model_versions = client.search_model_versions(
                    f"name='{model_name}' and status='Production'"
                )

                if model_versions:
                    latest_version = max(model_versions, key=lambda x: x.version)
                    model_uri = f"models:/{model_name}/{latest_version.version}"

                    logger.info(f"Loading model from MLflow: {model_uri}")
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = f"mlflow-{latest_version.version}"
                else:
                    raise FileNotFoundError("No production model found in MLflow")

            except Exception as mlflow_error:
                logger.warning(f"MLflow loading failed: {mlflow_error}. Trying local file...")

                import joblib
                model_path = "../artifacts/best_model.pkl"

                if os.path.exists(model_path):
                    logger.info(f"Loading model from local file: {model_path}")
                    self.model = joblib.load(model_path)
                    self.model_version = "local-v1.0.0"
                else:
                    logger.warning("No model file found, creating dummy model")
                    from sklearn.ensemble import RandomForestClassifier
                    self.model = RandomForestClassifier(n_estimators=10, random_state=42)
                    self.model_version = "dummy-v0.1.0"

            feature_columns_path = "../artifacts/feature_columns.json"
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, "r") as f:
                    self.feature_columns = json.load(f)

            self.loaded_at = datetime.now()
            logger.info(f"Model loaded successfully: {self.model_version}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_features(self, transaction: TransactionFeatures) -> pd.DataFrame:
        """Preprocess transaction features for model prediction"""

        features = {
            "Amount": transaction.amount,
            "Value": transaction.value,
            "ProductCategory": transaction.product_category,
            "ProviderId": transaction.provider_id,
            "ChannelId": transaction.channel_id,
            "transaction_hour": transaction.transaction_hour,
            "transaction_day": transaction.transaction_day,
            "transaction_month": transaction.transaction_month,
            "transaction_year": transaction.transaction_year,
            "PricingStrategy": transaction.pricing_strategy if transaction.pricing_strategy is not None else 2,
            "is_business_hours": transaction.is_business_hours
            if transaction.is_business_hours is not None
            else (1 if 9 <= transaction.transaction_hour <= 17 else 0),
            "is_fee_transaction": transaction.is_fee_transaction
            if transaction.is_fee_transaction is not None
            else (1 if transaction.amount < 0 else 0),
        }

        df = pd.DataFrame([features])
        df["amount_abs"] = df["Amount"].abs()
        df["amount_value_ratio"] = df["Amount"] / df["Value"].replace(0, 1)
        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)

        return df


model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API service...")
    try:
        model_loader.load_model()
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")
    yield
    logger.info("Shutting down API service...")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk based on transaction data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    error_response = ErrorResponse(
        error="Validation Error",
        detail=json.dumps(exc.errors()),
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    error_response = ErrorResponse(
        error=exc.detail,
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    error_response = ErrorResponse(
        error="Internal Server Error",
        detail=str(exc),
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    uptime = time.time() - (model_loader.loaded_at.timestamp() if model_loader.loaded_at else time.time())
    return HealthCheck(
        status="healthy" if model_loader.model else "degraded",
        model_loaded=model_loader.model is not None,
        model_version=model_loader.model_version,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(transaction: TransactionFeatures):
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_df = model_loader.preprocess_features(transaction)
    risk_score = float(model_loader.model.predict_proba(features_df)[0, 1])

    return PredictionResult(
        customer_id=transaction.customer_id,
        risk_score=risk_score,
        risk_category="",
        prediction_time=datetime.now()
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    predictions = []

    for transaction in request.transactions:
        features_df = model_loader.preprocess_features(transaction)
        risk_score = float(model_loader.model.predict_proba(features_df)[0, 1])

        predictions.append(
            PredictionResult(
                customer_id=transaction.customer_id,
                risk_score=risk_score,
                risk_category="",
                prediction_time=datetime.now()
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=model_loader.model_version,
        model_name=model_loader.model_name,
        processing_time_ms=(time.time() - start_time) * 1000,
        total_transactions=len(predictions)
    )


@app.get("/features/example", tags=["Features"])
async def get_feature_example():
    example = TransactionFeatures.model_config["json_schema_extra"]["example"]
    return {
        "example": example,
        "description": "Example transaction features for prediction"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
