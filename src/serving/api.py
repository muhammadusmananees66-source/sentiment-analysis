"""
Production inference API with:
- FastAPI endpoints
- MLflow model loading
- Circuit breaker pattern
- Request caching
- Prometheus metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import asyncio
import mlflow
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import time
import uuid
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator
import logging

from src.serving.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Pydantic models
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    request_id: Optional[str] = None

class SentimentResponse(BaseModel):
    request_id: str
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., max_length=100)  # Fixed: changed max_items to max_length

class BatchSentimentResponse(BaseModel):
    predictions: List[SentimentResponse]

# Model manager with hot-loading
class ModelManager:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_version = None
        self.load_time = None

    async def load_model(self, version: str = "latest"):
        """Load model from MLflow registry"""
        try:
            mlflow.set_tracking_uri("http://mlflow-service:5000")

            if version == "latest":
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions("sentiment_classifier", stages=["Production"])
                if latest_version:
                    model_uri = latest_version[0].source
                    self.model_version = latest_version[0].version
                else:
                    model_uri = "models:/sentiment_classifier/latest"
                    self.model_version = "latest"
            else:
                model_uri = f"models:/sentiment_classifier/{version}"
                self.model_version = version

            # Load model and vectorizer
            loaded = mlflow.sklearn.load_model(model_uri)

            # If saved as artifact dict
            if isinstance(loaded, dict):
                self.model = loaded.get('model')
                self.vectorizer = loaded.get('vectorizer')
            else:
                self.model = loaded

            self.load_time = time.time()
            logger.info(f"Model loaded: version {self.model_version}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Load fallback - create a simple dummy model if file doesn't exist
            try:
                self.model = joblib.load("/app/models/fallback_model.pkl")
            except:
                # Create a simple fallback model
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.feature_extraction.text import CountVectorizer
                self.model = RandomForestClassifier()
                self.vectorizer = CountVectorizer()
                # Fit on dummy data
                dummy_texts = ["good", "bad", "okay", "great", "terrible"]
                dummy_labels = [1, 0, 2, 1, 0]
                self.vectorizer.fit(dummy_texts)
                self.model.fit(self.vectorizer.transform(dummy_texts), dummy_labels)
                logger.info("Created fallback dummy model")
            self.model_version = "fallback"

    async def predict(self, text: str) -> Dict:
        """Single prediction"""
        # Preprocess if vectorizer exists
        if self.vectorizer:
            X = self.vectorizer.transform([text])
        else:
            # Simple feature extraction
            X = np.array([[len(text), len(text.split())]])

        # Get prediction
        pred_id = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]

        # Map label to sentiment
        sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}
        sentiment = sentiment_map.get(pred_id, "neutral")

        return {
            "sentiment": sentiment,
            "confidence": float(max(pred_proba)),
            "probabilities": {
                "negative": float(pred_proba[0]) if len(pred_proba) > 0 else 0,
                "positive": float(pred_proba[1]) if len(pred_proba) > 1 else 0,
                "neutral": float(pred_proba[2]) if len(pred_proba) > 2 else 0
            }
        }

# Global services
model_manager = ModelManager()
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
cache = None  # Redis cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global cache

    # Load model
    await model_manager.load_model()

    # Connect to Redis (optional - continue even if Redis fails)
    try:
        cache = await redis.from_url("redis://localhost:6379", decode_responses=True)  # Changed from redis://redis:6379
        await cache.ping()
        logger.info("Redis connected")
    except:
        cache = None
        logger.warning("Redis not available, caching disabled")

    yield

    # Cleanup
    if cache:
        await cache.close()

# Create app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-grade sentiment analysis with MLOps",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Metrics
instrumentator = Instrumentator().instrument(app)

# Health checks
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": model_manager.model_version}

@app.get("/ready")
async def readiness_check():
    return {"ready": model_manager.model is not None}

# Prediction endpoints
@app.post("/v1/predict", response_model=SentimentResponse)
async def predict_single(request: SentimentRequest, background_tasks: BackgroundTasks):
    """Predict sentiment for single text"""
    start_time = time.time()
    req_id = request.request_id or str(uuid.uuid4())

    # Circuit breaker check
    if circuit_breaker.is_open():
        raise HTTPException(503, "Service temporarily unavailable")

    # Cache check
    if cache:
        cache_key = f"pred:{hash(request.text)}"
        cached = await cache.get(cache_key)
        if cached:
            import json
            response = json.loads(cached)
            response["latency_ms"] = (time.time() - start_time) * 1000
            return SentimentResponse(**response)

    try:
        # Predict with timeout
        prediction = await asyncio.wait_for(
            model_manager.predict(request.text),
            timeout=1.0
        )

        # Build response
        response = SentimentResponse(
            request_id=req_id,
            text=request.text[:200],  # Truncate for response
            sentiment=prediction["sentiment"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            latency_ms=(time.time() - start_time) * 1000
        )

        # Cache
        if cache:
            await cache.setex(f"pred:{hash(request.text)}", 3600, response.json())

        circuit_breaker.record_success()
        return response

    except asyncio.TimeoutError:
        circuit_breaker.record_failure()
        raise HTTPException(504, "Prediction timeout")
    except Exception as e:
        circuit_breaker.record_failure()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/v1/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest):
    """Batch prediction for multiple texts"""
    start_time = time.time()

    # Process concurrently
    async def process_one(text):
        return await model_manager.predict(text)

    predictions = await asyncio.gather(*[process_one(text) for text in request.texts])

    responses = []
    for text, pred in zip(request.texts, predictions):
        responses.append(SentimentResponse(
            request_id=str(uuid.uuid4()),
            text=text[:200],
            sentiment=pred["sentiment"],
            confidence=pred["confidence"],
            probabilities=pred["probabilities"],
            latency_ms=(time.time() - start_time) * 1000 / len(request.texts)
        ))

    return BatchSentimentResponse(predictions=responses)


# Server startup code - ADD THIS AT THE END
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )



      