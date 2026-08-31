FROM python:3.11-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY insightwellness_ai/ ./insightwellness_ai/
# Knowledge base for the RAG agent behind /chat
COPY data/ ./data/

# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "insightwellness_ai.api.app:app"]
