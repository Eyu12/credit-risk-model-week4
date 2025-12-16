from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
from datetime import datetime
import numpy as np


class TransactionFeatures(BaseModel):
    """Features for a single transaction prediction"""
    
    # Customer features
    customer_id: str = Field(..., description="Unique customer identifier")
    
    # Transaction features
    amount: float = Field(..., description="Transaction amount", ge=-1000000, le=1000000)
    value: float = Field(..., description="Transaction value", ge=0, le=1000000)
    
    # Categorical features
    product_category: str = Field(
        ..., 
        description="Product category",
        examples=["airtime", "financial_services", "utility_bill", "data_bundles"]
    )
    provider_id: str = Field(
        ..., 
        description="Service provider ID",
        examples=["ProviderId_1", "ProviderId_4", "ProviderId_6"]
    )
    channel_id: str = Field(
        ..., 
        description="Channel ID",
        examples=["ChannelId_1", "ChannelId_2", "ChannelId_3"]
    )
    
    # Temporal features
    transaction_hour: int = Field(..., description="Hour of transaction", ge=0, le=23)
    transaction_day: int = Field(..., description="Day of month", ge=1, le=31)
    transaction_month: int = Field(..., description="Month of transaction", ge=1, le=12)
    transaction_year: int = Field(..., description="Year of transaction", ge=2018, le=2024)
    
    # Derived features (optional)
    pricing_strategy: Optional[int] = Field(None, description="Pricing strategy code", ge=0, le=4)
    is_business_hours: Optional[int] = Field(None, description="1 if business hours, 0 otherwise", ge=0, le=1)
    is_fee_transaction: Optional[int] = Field(None, description="1 if amount is negative, 0 otherwise", ge=0, le=1)
    
    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v == 0:
            raise ValueError("Amount cannot be zero")
        return v
    
    @field_validator("product_category")
    @classmethod
    def validate_product_category(cls, v: str) -> str:
        valid_categories = [
            "airtime", "financial_services", "utility_bill",
            "data_bundles", "tv", "transport", "ticket", "movies"
        ]
        if v not in valid_categories:
            raise ValueError(f"Product category must be one of: {valid_categories}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CustomerId_1234",
                "amount": 1000.0,
                "value": 1000.0,
                "product_category": "airtime",
                "provider_id": "ProviderId_6",
                "channel_id": "ChannelId_3",
                "transaction_hour": 14,
                "transaction_day": 15,
                "transaction_month": 11,
                "transaction_year": 2018,
                "pricing_strategy": 2,
                "is_business_hours": 1,
                "is_fee_transaction": 0
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    
    transactions: List[TransactionFeatures] = Field(
        ..., 
        description="List of transactions to predict",
        min_length=1,
        max_length=1000
    )
    
    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v: List[TransactionFeatures]):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 transactions")
        return v


class PredictionResult(BaseModel):
    """Single prediction result"""
    
    customer_id: str = Field(..., description="Customer identifier")
    risk_score: float = Field(..., description="Risk probability (0-1)", ge=0.0, le=1.0)
    risk_category: str = Field(..., description="Risk category")
    prediction_time: datetime = Field(..., description="Time of prediction")
    
    @field_validator("risk_category", mode="before")
    @classmethod
    def get_risk_category(cls, v, info):
        risk_score = info.data.get("risk_score")
        if risk_score is not None:
            if risk_score < 0.3:
                return "LOW_RISK"
            elif risk_score < 0.7:
                return "MEDIUM_RISK"
            else:
                return "HIGH_RISK"
        return v


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Model version used for predictions")
    model_name: str = Field(..., description="Name of the model")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    total_transactions: int = Field(..., description="Number of transactions processed")


class HealthCheck(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Time when error occurred")
