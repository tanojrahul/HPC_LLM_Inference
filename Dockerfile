# GPU-enabled base image with Python
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy source
COPY . .

ENV PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    HUGGINGFACE_HUB_CACHE=/app/models \
    API_KEY=changeme \
    ASR_MODEL=openai/whisper-small \
    LLM_MODEL=Qwen/Qwen3-0.6B

EXPOSE 8000

# Use gunicorn for production, single worker to keep model in memory
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "server:create_app()", "--workers", "1", "--timeout", "600"]
