import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")

app = FastAPI()


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: str
    island: str


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


model = load_model()


def preprocess_features(features: PenguinFeatures):
    """ 
    Note: This must match the preprocessing done during training.
    The model expects the same feature set as it was trained on.
    """
    input_dict = features.model_dump()  # This returns a dictionary of the model's fields
    X_input = pd.DataFrame([input_dict]) 
    X_input = pd.get_dummies(X_input, columns=["sex", "island"]) # Ensure the same 
    expected_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex_Female",
        "sex_Male",
        "island_Biscoe",
        "island_Dream",
        "island_Torgersen",
    ]
    X_input = X_input.reindex(columns=expected_cols, fill_value=0)
    X_input = X_input.astype(float)
    return X_input


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(features: PenguinFeatures):
    X_input = preprocess_features(features)
    pred = model.predict(X_input.values)
    return {"prediction": int(pred[0])}


@app.get("/health")
async def health():
    return {"status": "ok"}
