from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_predict_fraud():
    payload = {
        "transaction_amount": 1200.0,
        "distance_from_home": 350.0,
        "hour_of_day": 3,
        "device_risk_score": 0.95,
        "velocity_last_1h": 6,
        "is_international": 1,
        "previous_fraud_count": 2,
        "account_age_days": 5
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "fraud_risk_score" in data
    assert "top_shap_drivers" in data