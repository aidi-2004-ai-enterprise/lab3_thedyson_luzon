# Lab 3 – Penguins Classification (XGBoost + FastAPI)

![Demo](assets/demo.gif)

> **Author:** Thedyson Luzon 
> **Repo:** `lab3_thedyson_luzon` • **Course:** AIDI‑2004 – AI Enterprise

---

## Overview

Train an **XGBoost** model on Seaborn’s **penguins** dataset and deploy it via **FastAPI**.  
This repo meets the rubric by including:

- One‑hot encoding for `sex` & `island`, label encoding for `species`
- Stratified 80/20 split, anti‑overfitting XGBoost params (`max_depth=3`, `n_estimators=100`, etc.)
- FastAPI app with Pydantic `Enum`s (strict categorical validation)
- Graceful error handling (invalid/missing fields) + logging to file & console
- Dependency management with **uv** (preferred) or `pip` or **Docker Compose**
- Working `/docs` Swagger UI and a short demo GIF/video

---

## Project Structure
```
├── train.py
├── .github/
│ ├── workflows/
│ │ ├── ci.yml
│ └── pull_request_template.md
├── app/
│ ├── main.py
│ ├── data/
│ │ ├── model.json
│ │ └── preprocess_meta.json
│ └── logs/
├── tests/
│ └── test_app.py
├── assets/
│ └── demo.gif
├── requirements.txt
├── pyproject.toml 
├── Dockerfile
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── uv.lock
└── README.md
```
---

## Quick Start

### A) With `uv`
```bash
uv sync
uv run python train.py                      # trains & saves artifacts to app/data/
uv run uvicorn app.main:app --reload        # http://127.0.0.1:8000
```
### B) With `pip/venv`
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```
### C) With Docker Compose
```bash
docker compose up --build # API: http://127.0.0.1:8080
# Stop
docker compose down --remove-orphans
```
## Example Request
### Success
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 50.0,
    "bill_depth_mm": 15.0,
    "flipper_length_mm": 210.0,
    "body_mass_g": 4500.0,
    "year": 2008,
    "sex": "male",
    "island": "Biscoe"
  }'
```
#### Sample Response
```json
{
  "prediction": 2,
  "species": "Gentoo",
  "probabilities": {
    "Adelie": 0.01,
    "Chinstrap": 0.02,
    "Gentoo": 0.97
  }
}
```
## Invalid Categorical Value
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 50,
    "bill_depth_mm": 15,
    "flipper_length_mm": 210,
    "body_mass_g": 4500,
    "year": 2008,
    "sex": "male",
    "island": "Atlantis"
  }'
```
`Returns 422 with clear enum validation errors.`
## Missing required field
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 50,
    "bill_depth_mm": 15,
    "flipper_length_mm": 210,
    "body_mass_g": 4500,
    "year": 2008,
    "island": "Dream"
  }'
```
### Logging
`app/logs/app.log`
```bash
tail -f app/logs/app.log
```
### Testing
```bash
uv run pytest -q
# or
docker compose run --rm api pytest -q
```