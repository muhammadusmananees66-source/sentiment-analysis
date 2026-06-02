# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     make \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application
# COPY src/ ./src/
# COPY configs/ ./configs/
# COPY tests/ ./tests/

# # Create non-root user
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# # Health check
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Run application
# CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]






FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ make curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY deployments/docker/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

ENTRYPOINT ["serve"]