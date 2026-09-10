# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies: LaTeX for PDF generation, build tools for any
# wheels that need compilation, curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && pip install sentence-transformers

# Copy source
COPY app/ ./app/
COPY run.py ./
COPY .env.example ./.env.example

# Writable data directory
RUN mkdir -p /app/data

EXPOSE 8000 8501

# Default: run the backend. Compose overrides this per-service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]