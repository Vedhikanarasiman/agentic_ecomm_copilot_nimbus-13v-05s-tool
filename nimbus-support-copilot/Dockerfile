FROM python:3.11-slim

WORKDIR /code

# System deps for asyncpg / sentence-transformers build steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Editable-style install without -e (no source mutation needed inside the
# image) — installs the app package properly instead of relying on WORKDIR
# happening to be on sys.path.
RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
