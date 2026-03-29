FROM python:3.11-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY insightwellness_ai/api/app.py .

COPY models/ ./models/

# Add the virtual environment to PATH so Python knows where your packages are
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["uv", "run", "flask", "--app", "app", "run", "--host=0.0.0.0", "--port=8080"]