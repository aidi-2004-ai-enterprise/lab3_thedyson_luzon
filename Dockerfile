FROM python:3.10-slim

WORKDIR /app

# System-level deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose API port
EXPOSE 8080

# Default command: train then serve
CMD ["bash", "-c", "python train.py && uvicorn app.main:app --host 0.0.0.0 --port 8080"]
