FROM python:3.12-slim

LABEL org.opencontainers.image.title="vector-novelty"
LABEL org.opencontainers.image.description="Centroid-based novelty scoring for agent populations — pure NumPy, O(n) diversity."
LABEL org.opencontainers.image.source="https://github.com/SuperInstance/vector-novelty"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Install build deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching
COPY pyproject.toml README.md ./
COPY vector_novelty/ ./vector_novelty/
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e .

# Copy tests
COPY tests/ ./tests/

# Switch to non-root user
USER appuser

# No web server — run test suite as smoke test
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -m pytest tests/ -x --tb=short || exit 1

CMD ["python3", "-m", "pytest", "tests/", "-v"]
