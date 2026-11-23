FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY liveguard ./liveguard
RUN pip install --no-cache-dir ".[voice,gemini]"
VOLUME ["/config", "/data", "/models"]
EXPOSE 8099
ENTRYPOINT ["liveguard", "-c", "/config/liveguard.yaml"]
