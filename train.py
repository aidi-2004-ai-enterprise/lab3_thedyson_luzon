"""
train.py
Train an XGBoost classifier on Seaborn's penguins dataset, evaluate it,
and save the model + preprocessing metadata for inference.

Run:
    uv run python train.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Paths
DATA_DIR = Path("app/data")
MODEL_PATH = DATA_DIR / "model.json"
META_PATH = DATA_DIR / "preprocess_meta.json"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = sns.load_dataset("penguins")

    if "year" not in df.columns:
        df["year"] = 2008  # Default year if not present

    cols_needed = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
        "body_mass_g", "sex", "island", "year", "species"
    ]
    # Only dropna on columns that actually exist (safety)
    present = [c in df.columns and c for c in cols_needed]
    df = df.dropna(subset=present)
    return df



def encode(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, Dict, LabelEncoder]:
    """Encode features and target, returning arrays and metadata."""
    # Target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    # Categorical OHE with fixed categories
    categories = [sorted(df[c].unique().tolist()) for c in cat_cols]
    ohe = OneHotEncoder(
        categories=categories,
        sparse_output=False,
        handle_unknown="ignore"
    )
    X_cat = ohe.fit_transform(df[cat_cols])
    X_num = df[num_cols].to_numpy(dtype=float)

    X = np.concatenate([X_num, X_cat], axis=1)

    feature_names = (
        num_cols +
        [f"{col}_{cat}" for col, cats in zip(cat_cols, categories) for cat in cats]
    )

    meta = {
        "numeric_features": num_cols,
        "categorical_features": {col: cats for col, cats in zip(cat_cols, categories)},
        "ohe_feature_names": feature_names,
        "label_mapping": {cls: int(idx) for idx, cls in enumerate(le.classes_)},
        "index_to_label": {int(idx): cls for idx, cls in enumerate(le.classes_)},
    }
    return X, y, meta, le


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    """Train XGBoost with parameters to reduce overfitting."""
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    cat_cols = ["sex", "island"]
    num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year"]
    target_col = "species"

    X, y, meta, label_encoder = encode(df, cat_cols, num_cols, target_col)

    # Stratified split to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)

    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    f1_tr = f1_score(y_train, y_pred_train, average="weighted")
    f1_te = f1_score(y_test, y_pred_test, average="weighted")

    print("=== Classification report (TEST) ===")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
    print(f"Weighted F1   Train: {f1_tr:.3f} | Test: {f1_te:.3f}")

    # Save model + meta
    model.save_model(str(MODEL_PATH))
    with META_PATH.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
