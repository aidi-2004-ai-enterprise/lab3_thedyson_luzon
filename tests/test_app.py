from fastapi.testclient import TestClient
from app.main import app


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict():
    client = TestClient(app)
    # Example penguin features (now using string values for sex and island)
    features = {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "year": 2007,
        "sex": "male",
        "island": "Dream",
    }
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    assert "prediction" in response.json()
