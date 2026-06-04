"""
FastAPI application for credit risk prediction.
Connects to MLflow server to load models.

Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import mlflow
import mlflow.pyfunc

# ============================================================================
# SET MLFLOW TRACKING URI (IMPORTANT!)
# ============================================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CustomerDataRequest(BaseModel):
    """Request schema for /predict endpoint."""
    recency: float = Field(..., ge=0, description="Days since last transaction")
    frequency: int = Field(..., ge=1, description="Number of transactions")
    monetary_value: float = Field(..., ge=0, description="Total amount spent")
    total_transaction_amount: float = Field(..., ge=0)
    avg_transaction_amount: float = Field(..., ge=0)
    std_transaction_amount: float = Field(..., ge=0)
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour 0-23")
    transaction_day: int = Field(..., ge=1, le=31, description="Day 1-31")
    transaction_month: int = Field(..., ge=1, le=12, description="Month 1-12")
    transaction_year: int = Field(..., ge=2000, description="Year >= 2000")
    channel_id: int = Field(..., description="Channel ID")
    product_category_encoded: int = Field(..., description="Product category")
    country_code: int = Field(..., description="Country code")
    currency_code: int = Field(..., description="Currency code")


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability 0-1")
    risk_category: str = Field(..., description="high_risk, medium_risk, or low_risk")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    recommendation: str = Field(..., description="Loan recommendation")


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str = Field(..., description="healthy or degraded")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    message: str = Field(..., description="Status message")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts customer credit risk probability",
    version="1.0.0",
    docs_url="/docs"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

model = None
model_loaded = False
model_info = "Not loaded"


def load_model_from_mlflow():
    """Load the best model from MLflow server."""
    global model, model_loaded, model_info
    
    try:
        logger.info("Attempting to load model from MLflow server...")
        logger.info(f"MLflow URI: {mlflow.get_tracking_uri()}")
        
        # Try to load from MLflow Model Registry
        try:
            logger.info("  Trying MLflow Model Registry (production)...")
            model = mlflow.pyfunc.load_model("models:/credit-risk-model/production")
            model_loaded = True
            model_info = "Loaded from MLflow Model Registry (production)"
            logger.info(f"✅ {model_info}")
            return True
        except Exception as e:
            logger.warning(f"  Registry failed: {str(e)[:100]}")
        
        # Try to load from latest run
        try:
            logger.info("  Trying latest MLflow run...")
            runs = mlflow.search_runs(max_results=1)
            
            if len(runs) == 0:
                logger.warning("  No MLflow runs found")
                model_info = "No MLflow runs found"
                return False
            
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            
            logger.info(f"  Found run: {run_id}")
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            model_loaded = True
            model_info = f"Loaded from MLflow run {run_id[:8]}"
            logger.info(f"✅ {model_info}")
            return True
            
        except Exception as e:
            logger.warning(f"  Latest run failed: {str(e)[:100]}")
            model_info = f"Could not load from runs: {str(e)[:100]}"
            return False
        
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        model_info = f"Error: {str(e)[:100]}"
        return False


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 70)
    logger.info("Starting Credit Risk Prediction API...")
    logger.info(f"MLflow Tracking URI: http://127.0.0.1:5000")
    logger.info("=" * 70)
    
    load_model_from_mlflow()
    
    if model_loaded:
        logger.info("✅ Model loaded successfully!")
    else:
        logger.warning(f"⚠️  {model_info}")
    
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API shutting down...")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not model_loaded:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            message=model_info
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message=model_info
    )


@app.get("/model-info")
async def model_info_endpoint():
    """Get model information."""
    return {
        "loaded": model_loaded,
        "info": model_info,
        "mlflow_uri": mlflow.get_tracking_uri()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerDataRequest):
    """Predict credit risk for a customer."""
    
    if not model_loaded:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail=f"Model service unavailable. {model_info}"
        )
    
    try:
        logger.info("Processing prediction request...")
        
        # Convert to array
        features = np.array([[
            customer_data.recency,
            customer_data.frequency,
            customer_data.monetary_value,
            customer_data.total_transaction_amount,
            customer_data.avg_transaction_amount,
            customer_data.std_transaction_amount,
            customer_data.transaction_hour,
            customer_data.transaction_day,
            customer_data.transaction_month,
            customer_data.transaction_year,
            customer_data.channel_id,
            customer_data.product_category_encoded,
            customer_data.country_code,
            customer_data.currency_code
        ]])
        
        # Predict
        prediction = model.predict(features)
        risk_probability = float(prediction[0])
        risk_probability = max(0.0, min(1.0, risk_probability))
        
        # Classify
        if risk_probability >= 0.7:
            risk_category = "high_risk"
            recommendation = "Reject application"
        elif risk_probability >= 0.5:
            risk_category = "medium_risk"
            recommendation = "Approve with stricter terms"
        else:
            risk_category = "low_risk"
            recommendation = "Approve with standard terms"
        
        risk_score = int(risk_probability * 100)
        
        logger.info(f"✅ {risk_category} (prob={risk_probability:.3f})")
        
        return PredictionResponse(
            risk_probability=risk_probability,
            risk_category=risk_category,
            risk_score=risk_score,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)