FROM python:3.11-slim

WORKDIR /app

# Install essential compilation libraries for C extensions (e.g. xgboost, psutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies without editable mode flag (for container setup)
RUN sed -i '/-e ./d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

COPY . .

# Expose the Flask port
EXPOSE 5000

# CMD to launch Flask application
CMD ["python", "app.py"]
