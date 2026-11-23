FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY liveguard ./liveguard
RUN pip install --no-cache-dir ".[voice]"
VOLUME ["/config", "/data", "/models"]
EXPOSE 8099
ENTRYPOINT ["liveguard", "-c", "/config/liveguard.yaml"]
