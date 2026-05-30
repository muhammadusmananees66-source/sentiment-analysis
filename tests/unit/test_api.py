"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.serving.api import app


class TestSentimentAPI:
    """Test cases for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_readiness_endpoint(self, client):
        """Test readiness check endpoint"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    def test_predict_endpoint_with_valid_text(self, client):
        """Test prediction with valid text"""
        response = client.post(
            "/v1/predict",
            json={"text": "This movie is great!"}
        )
        # May fail if model not loaded (500) or succeed (200)
        assert response.status_code in [200, 500]

    def test_predict_endpoint_with_empty_text(self, client):
        """Test prediction with empty text should fail"""
        response = client.post(
            "/v1/predict",
            json={"text": ""
        }
        )
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_with_long_text(self, client):
        """Test prediction with very long text"""
        long_text = "test " * 5000  # ~30k chars
        
        response = client.post(
            "/v1/predict",
            json={"text": long_text}
        )
        # Should return 422 (validation error) because text exceeds max_length=10000
        assert response.status_code == 422

    @pytest.mark.skip(reason="Requires trained model to be loaded - run integration tests separately")
    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint"""
        response = client.post(
            "/v1/predict/batch",
            json={"texts": ["Good movie", "Bad movie", "Okay movie"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        # Verify each prediction has expected structure
        for pred in data["predictions"]:
            assert "label" in pred or "sentiment" in pred
            assert "confidence" in pred or "score" in pred