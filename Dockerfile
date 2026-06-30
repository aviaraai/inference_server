FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    libopenblas-dev \
    libomp-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY pyproject.toml .
COPY uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy application
COPY . .

ENV PYTHONPATH=/app

EXPOSE 9004

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
