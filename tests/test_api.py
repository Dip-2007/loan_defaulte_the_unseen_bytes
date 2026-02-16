# tests/test_api.py
"""
Tests for the FastAPI prediction service endpoints.
"""
import pytest
import sys
import os
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app


@pytest.fixture
async def client():
    """Create an async test client compatible with httpx 0.28+."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    async def test_health_reports_model_status(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "shap_available" in data


@pytest.mark.anyio
class TestRiskScoreEndpoint:
    def _customer_payload(self, **overrides):
        base = {
            "LoanID": "API_TEST_001",
            "Age": 32,
            "Income": 85000,
            "LoanAmount": 500000,
            "CreditScore": 720,
            "MonthsEmployed": 36,
            "DTIRatio": 23.5,
        }
        base.update(overrides)
        return base

    async def test_risk_score_returns_200(self, client):
        response = await client.post("/risk-score", json=self._customer_payload())
        assert response.status_code == 200

    async def test_risk_score_has_required_fields(self, client):
        response = await client.post("/risk-score", json=self._customer_payload())
        data = response.json()
        assert "risk_score" in data
        assert "risk_band" in data
        assert "component_scores" in data
        assert "top_risk_factors" in data

    async def test_risk_score_valid_band(self, client):
        response = await client.post("/risk-score", json=self._customer_payload())
        data = response.json()
        valid_bands = {'SAFE', 'LOW RISK', 'MODERATE', 'HIGH RISK', 'CRITICAL'}
        assert data["risk_band"] in valid_bands

    async def test_risk_score_range(self, client):
        response = await client.post("/risk-score", json=self._customer_payload())
        data = response.json()
        assert 0 <= data["risk_score"] <= 100

    async def test_high_dti_increases_risk(self, client):
        low = (await client.post("/risk-score", json=self._customer_payload(DTIRatio=20))).json()
        high = (await client.post("/risk-score", json=self._customer_payload(DTIRatio=70))).json()
        assert high["risk_score"] > low["risk_score"]


@pytest.mark.anyio
class TestPredictEndpoint:
    async def test_predict_returns_200_or_503(self, client):
        response = await client.post("/predict", json={
            "LoanID": "API_TEST_002",
            "Age": 35,
            "Income": 60000,
            "CreditScore": 650,
        })
        # 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]

    async def test_predict_response_format(self, client):
        response = await client.post("/predict", json={
            "LoanID": "API_TEST_003",
            "Age": 35,
            "Income": 60000,
            "CreditScore": 650,
        })
        if response.status_code == 200:
            data = response.json()
            assert "default_probability" in data
            assert "risk_score_0_to_100" in data
            assert "risk_band" in data
            assert 0 <= data["default_probability"] <= 1


@pytest.mark.anyio
class TestExplainEndpoint:
    async def test_explain_returns_200_or_503(self, client):
        response = await client.post("/explain", json={
            "LoanID": "API_TEST_004",
            "Age": 32,
            "Income": 85000,
            "CreditScore": 720,
        })
        assert response.status_code in [200, 503]

    async def test_explain_response_format(self, client):
        response = await client.post("/explain", json={
            "LoanID": "API_TEST_005",
            "Age": 32,
            "Income": 85000,
            "CreditScore": 720,
        })
        if response.status_code == 200:
            data = response.json()
            assert "base_value" in data
            assert "risk_probability" in data
            assert "top_risk_up" in data
            assert "top_risk_down" in data
            assert len(data["top_risk_up"]) > 0
