# Multi-stage production Dockerfile for AI ChatBot API & Inference Engine
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools for dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-optimized torch & project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create dedicated non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application source and baseline artifacts
COPY common/ /app/common/
COPY ChatBot/ /app/ChatBot/
COPY ChatBot-PipeLine/ /app/ChatBot-PipeLine/
COPY artifacts/ /app/artifacts/
COPY monitoring/ /app/monitoring/
COPY pytest.ini /app/pytest.ini
COPY .dvc/ /app/.dvc/
COPY dvc.yaml dvc.lock params.yaml /app/
COPY scripts/render-start.sh /app/scripts/render-start.sh

# Ensure runtime directories exist with correct non-root permissions
RUN mkdir -p /app/ChatBot-PipeLine/data/feedback \
             /app/ChatBot-PipeLine/data/versioned \
             /app/artifacts/models \
             /app/artifacts/reports/evidently \
             /app/artifacts/metrics && \
    chmod +x /app/scripts/render-start.sh && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/ChatBot:/app/ChatBot-PipeLine" \
    PORT=6655 \
    DO_NOT_TRACK=1 \
    EVIDENTLY_DISABLE_TELEMETRY=1 \
    MLFLOW_ALLOW_FILE_STORE=true

EXPOSE 6655

# Healthcheck distinguishes liveness from a hung process and respects Render's
# dynamically assigned PORT.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD-SHELL curl -f "http://localhost:${PORT:-6655}/health" || exit 1

# Graceful startup entrypoint
CMD ["/app/scripts/render-start.sh"]

