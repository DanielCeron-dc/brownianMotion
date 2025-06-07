FROM python:3.10-slim

# set working directory
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your app code (but not venv)
COPY . .

# expose and run
EXPOSE 5004
CMD ["gunicorn", "-k", "aiohttp.GunicornWebWorker", "your_module:app", "--bind", "0.0.0.0:5004"]