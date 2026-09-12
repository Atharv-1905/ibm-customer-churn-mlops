from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "Monthly Charges": 85.50,
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_value" in result
