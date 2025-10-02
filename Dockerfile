# syntax=docker/dockerfile:1.6

############################
# Stage 1 — build wheels
############################
FROM python:3.11-slim AS builder

# System deps needed to build some wheels (netCDF4, psycopg if ever added, etc.)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libhdf5-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Use a venv so we can copy it into the runtime image
ENV VENV=/opt/venv
RUN python -m venv $VENV
ENV PATH="$VENV/bin:$PATH" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Leverage layer caching
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel \
    && pip install -r requirements.txt

# Copy source
COPY . /app


############################
# Stage 2 — runtime image
############################
FROM python:3.11-slim AS runtime

# Runtime libs only (smaller than *-dev)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libhdf5-103-1 libnetcdf19 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy the prebuilt virtualenv and app code
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --from=builder /app /app

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Ports:
# - 8080 FastAPI (your uvicorn run inside air_quality_api.py)
# - 9308 metrics (if your app exposes it) — optional
EXPOSE 8080 9308

# Healthcheck hits FastAPI /health on 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Ensure bootstrap runs once on start (your script checks BOOTSTRAP env)
ENV BOOTSTRAP=1

# Start the app by executing your launcher script
# (your air_quality_api.py calls uvicorn.run(..., port=8080))
CMD ["python", "air_quality_api.py"]
