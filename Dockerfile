# ==========================================
# Stage 1: Build React + Vite Frontend
# ==========================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python / FastAPI + EfficientNetB0 ML Runtime
# ==========================================
FROM python:3.11-slim

# System dependencies for OpenCV, PIL, and native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as non-root user (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=torch

WORKDIR $HOME/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend, inference pipeline, models, results, and sample data
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user src/ ./src/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user results/ ./results/
COPY --chown=user:user samples/ ./samples/

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./frontend/dist

# Switch to non-root user
USER user

# Hugging Face Spaces default web port
EXPOSE 7860

# Launch Uvicorn server serving both FastAPI endpoints and React SPA
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
