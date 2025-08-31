# Use a lightweight Python base image
FROM python:3.10-slim

# Prevents Python from writing .pyc files & buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements if you have, else directly install
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Expose port
EXPOSE 8000

# Start Gradio app
CMD ["python", "sop-openai.py"]
