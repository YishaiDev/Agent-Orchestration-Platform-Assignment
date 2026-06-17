# Agent Orchestration Platform — API image.
# The uv project lives in app/; the repo root is on PYTHONPATH so `app` imports as a package.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace/app

# Install dependencies first (cached unless the lockfile changes); the app runs from source.
COPY app/pyproject.toml app/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application source (only tracked source — never .env or .venv).
COPY app/__init__.py app/main.py app/cli.py app/config.yaml ./
COPY app/src ./src
COPY app/evals ./evals

ENV PATH="/workspace/app/.venv/bin:$PATH" \
    PYTHONPATH=/workspace \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["python", "main.py"]
