# =============================================================================
# MCP Developer Assistant - Multi-stage Dockerfile
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Python environment
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 mcp && \
    useradd --uid 1000 --gid mcp --shell /bin/bash --create-home mcp

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY --chown=mcp:mcp . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R mcp:mcp /app

# -----------------------------------------------------------------------------
# Stage 2: Proxy server
# -----------------------------------------------------------------------------
FROM base as proxy

USER mcp

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "proxy.auth_gateway:app", "--host", "0.0.0.0", "--port", "8001"]

# -----------------------------------------------------------------------------
# Stage 3: MCP Server
# -----------------------------------------------------------------------------
FROM base as server

USER mcp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "server.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: Development
# -----------------------------------------------------------------------------
FROM base as dev

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

USER mcp

EXPOSE 8000 8001

CMD ["bash"]
