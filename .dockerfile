FROM python:3.8-slim

# Install zstd so mise can extract .tar.zst
RUN apt-get update && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5004
CMD ["gunicorn", "-k", "aiohttp.GunicornWebWorker", "your_module:app", "--bind", "0.0.0.0:5004"]