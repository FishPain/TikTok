# Use Python 3.12 slim
FROM python:3.12-slim

# Install system deps (Pillow & OpenCV need these)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY location.py .

# Environment variables
ENV FLASK_APP=location.py
ENV FLASK_ENV=production
ENV GEMINI_TEXT="gemini-2.5-flash"
ENV GEMINI_IMAGE="gemini-2.5-flash-image-preview"

# Expose port
EXPOSE 8300

# Run service
CMD ["python", "location.py"]