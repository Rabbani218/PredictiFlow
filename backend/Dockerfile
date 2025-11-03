FROM python:3.9-slim

WORKDIR /app

# Install system build dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install build tools then requirements
RUN python -m pip install --upgrade pip setuptools wheel build \
	&& pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]