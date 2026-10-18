# Multi-platform production container for StudySphere AI Backend
# Compatible with Hugging Face Spaces (Port 7860, 16GB RAM, Free ₹0)
# and Render / Docker container hosts (Port dynamic $PORT)

FROM python:3.11-slim

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    ENVIRONMENT=production \
    STRICT_SESSION_AUTH=true \
    RATE_LIMIT_ENABLED=true \
    MAX_UPLOAD_SIZE_MB=25 \
    HF_HOME=/app/.cache/huggingface

# Install system utilities (curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-privileged user for security compliance
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Step 1: Install PyTorch CPU wheel first to prevent downloading 2GB+ CUDA binaries
RUN pip install --no-cache-dir torch>=2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Pre-download embedding model weights during build to eliminate cold-start runtime latency
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Step 4: Copy application source code
COPY . .

# Step 5: Ensure non-root ownership and switch user
RUN chown -R appuser:appuser /app
USER appuser

# Expose default port (7860 for Hugging Face Spaces, overridable by host via $PORT)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-7860}/api/health || exit 1

# Start FastAPI via uvicorn binding to 0.0.0.0 and dynamically adapting to $PORT
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
