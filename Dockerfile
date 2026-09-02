# ==============================================================================
# Edge AI Vision Engine - Multi-Stage Optimized Dockerfile
# Base: Python 3.11 Slim (Debian Bookworm)
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final Production Runtime Image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime OpenCV & multimedia libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python wheels from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code
COPY config.py .
COPY engine/ ./engine/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY .env.example .env

# Expose FastAPI HTTP & WebSocket Port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start Uvicorn ASGI Server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
