"""
tests/test_health.py
Tests for the root and health endpoints.
Run from the deployment/ directory:
    pytest tests/test_health.py -v
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client with the full app (models loaded)."""
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_root_contains_service_name(client):
    data = response = client.get("/").json()
    assert "service" in data
    assert "Janwani" in data["service"]


def test_root_lists_endpoints(client):
    data = client.get("/").json()
    assert "endpoints" in data


def test_root_lists_loaded_models(client):
    data = client.get("/").json()
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], list)


def test_health_returns_200_when_ready(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body_healthy(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"


def test_health_contains_uptime(client):
    data = client.get("/health").json()
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_health_civic_model_listed(client):
    data = client.get("/health").json()
    assert "civic" in data.get("models_loaded", [])
