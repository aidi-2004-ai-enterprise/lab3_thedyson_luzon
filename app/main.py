from model import XGBoostModel_mock
from fastapi import FastAPI
from pydantic import BaseModel  
from typing import List, Dict

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

def normalize(data):
    return data

@app.get("/predict")
async def predict(features):

    normalized_data = normalize(features)

    model = XGBoostModel_mock()
    prediction = model.predict(normalized_data)
    return {"prediction": prediction}