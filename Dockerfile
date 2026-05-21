# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for container security
RUN useradd --create-home --uid 1000 trader
WORKDIR /app

# Copy pre-built packages from builder stage
COPY --from=builder /install /usr/local

# Copy source — secrets NEVER copied; they arrive via K8s envFrom at runtime
COPY --chown=trader:trader . .

# Runtime defaults — all trading is paper-only by default
ENV PAPER_ONLY=true \
    APCA_PAPER=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_DIR=/app/logs/execution \
    DATA_DIR=/app/system-data

# Create runtime dirs — logs and data are mounted as PVCs in K8s
RUN mkdir -p /app/logs/execution /app/system-data /app/research_requests \
    && chown -R trader:trader /app/logs /app/system-data /app/research_requests

USER trader

# Health probe: just check the process is responsive
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default entrypoint — override STRATEGY_ID and CONFIG_PATH per K8s deployment
ENTRYPOINT ["python", "run.py"]
