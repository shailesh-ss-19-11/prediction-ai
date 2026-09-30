FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    PORT=5000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /data /app/logs
VOLUME ["/data"]
EXPOSE 5000

# Credentials come from .env at runtime (see docker-compose.yml), never baked in.
CMD ["python", "main.py"]
