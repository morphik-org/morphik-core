# Use Python 3.12.5 as base image
FROM python:3.12.5-slim

# Set working directory
WORKDIR /app

# Unstructured.io dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libmagic1 -y
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
# # Copy and load environment variables from .env file
# COPY .env .
# ENV $(cat .env | xargs)



# Expose port
EXPOSE 8000 443 80 20

# Run the server
# CMD ["python", "start_server.py"]
CMD ["uvicorn", "core.api:app", "--host", "127.0.0.1", "--port", "443"]
