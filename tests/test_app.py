from fastapi.testclient import TestClient
from app.main import app
import pytest


# This fixture returns a dictionary of features that can be used in all tests.
@pytest.fixture
def features():
    return {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "year": 2007,
        "sex": "male",
        "island": "Dream",
    }


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict(features):
    client = TestClient(app)
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_missing_sex(features):
    client = TestClient(app)
    features_missing_sex = features.copy()
    features_missing_sex.pop("sex")
    response = client.post("/predict", json=features_missing_sex)
    assert response.status_code == 422


def test_predict_invalid_year_type(features):
    client = TestClient(app)
    features_invalid_year = features.copy()
    features_invalid_year["year"] = "not_a_year"
    response = client.post("/predict", json=features_invalid_year)
    assert response.status_code == 422


def test_predict_edge_negative_values(features):
    client = TestClient(app)
    features_negative = features.copy()
    features_negative["bill_length_mm"] = -1.0
    features_negative["bill_depth_mm"] = -1.0
    features_negative["flipper_length_mm"] = -1.0
    features_negative["body_mass_g"] = -1.0
    response = client.post("/predict", json=features_negative)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_edge_large_values(features):
    client = TestClient(app)
    features_large = features.copy()
    features_large["bill_length_mm"] = 1e6
    features_large["bill_depth_mm"] = 1e6
    features_large["flipper_length_mm"] = 1e6
    features_large["body_mass_g"] = 1e6
    features_large["year"] = 9999
    features_large["sex"] = "female"
    features_large["island"] = "Biscoe"
    response = client.post("/predict", json=features_large)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_all_categorical_combinations(features):
    client = TestClient(app)
    sexes = ["male", "female"]
    islands = ["Biscoe", "Dream", "Torgersen"]
    for sex in sexes:
        for island in islands:
            features_combo = features.copy()
            features_combo["sex"] = sex
            features_combo["island"] = island
            response = client.post("/predict", json=features_combo)
            assert response.status_code == 200
            assert "prediction" in response.json()


def test_predict_output_range(features):
    client = TestClient(app)
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    pred = response.json()["prediction"]
    assert pred in [0, 1, 2]
