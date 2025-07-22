# app/main.py
"""
FastAPI service for penguin species prediction.
Matches test expectations:
- GET /  -> {"message": "Hello World"}
- POST /predict -> returns {"prediction": int, "species": str, "probabilities": {...}}
- 422 on Pydantic validation errors (missing/wrong types)
- Accepts negative / huge numbers (no gt/lt validation)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Paths ----------------
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = DATA_DIR / "model.json"
META_PATH = DATA_DIR / "preprocess_meta.json"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("penguins_api")


# --------------- Schemas ----------------
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"


class Sex(str, Enum):
    male = "male"
    female = "female"


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island


class PredictionResponse(BaseModel):
    prediction: int
    species: str
    probabilities: Dict[str, float]


# ------------- Helpers ------------------
def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        msg = "Model or metadata missing. Run train.py first."
        logger.error(msg)
        raise RuntimeError(msg)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with META_PATH.open() as f:
        meta = json.load(f)

    # JSON makes keys strings; convert back to int
    if "index_to_label" in meta:
        meta["index_to_label"] = {int(k): v for k, v in meta["index_to_label"].items()}
    if "label_mapping" in meta:
        meta["label_mapping"] = {k: int(v) for k, v in meta["label_mapping"].items()}

    logger.info("Model and metadata loaded.")
    return model, meta


def vectorize(features: PenguinFeatures, meta: dict) -> np.ndarray:
    """Create feature vector in the exact order used during training."""
    num_vals = {
        "bill_length_mm": features.bill_length_mm,
        "bill_depth_mm": features.bill_depth_mm,
        "flipper_length_mm": features.flipper_length_mm,
        "body_mass_g": features.body_mass_g,
        "year": features.year,
    }
    cat_vals = {"sex": features.sex.value, "island": features.island.value}

    vec: List[float] = []
    for fname in meta["ohe_feature_names"]:
        if fname in num_vals:
            vec.append(float(num_vals[fname]))
        else:
            col, cat = fname.split("_", 1)
            vec.append(1.0 if cat_vals[col] == cat else 0.0)
    return np.array(vec, dtype=float).reshape(1, -1)


# --------------- App --------------------
app = FastAPI(title="Penguins Classifier API", version="1.0.0")

model, meta = load_artifacts()


@app.get("/", tags=["health"])
def root():
    return {"message": "Hello World"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: PenguinFeatures):
    logger.info("Received request: %s", features.model_dump())

    try:
        X = vectorize(features, meta)
    except KeyError as e:
        logger.warning("Vectorize error: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid categorical value: {e}")

    try:
        probs = model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        species = meta["index_to_label"].get(idx)
        if species is None:
            logger.error("index_to_label missing idx %s", idx)
            raise HTTPException(status_code=500, detail="Internal label mapping error.")
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail="Prediction failed. See logs.")

    probs_named = {meta["index_to_label"][i]: float(p) for i, p in enumerate(probs)}

    logger.info("Prediction success: %s (%d)", species, idx)
    return {"prediction": idx, "species": species, "probabilities": probs_named}
