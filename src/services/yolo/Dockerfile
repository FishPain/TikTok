FROM python:3.12-slim-bullseye

# Install system dependencies (OpenCV + YOLO + certs)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    wget \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Pre-download YOLO model (tiny, cached by ultralytics)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO download attempted"

# Copy service code
COPY yolo.py .

COPY models /app/models

EXPOSE 5100

ENV FLASK_APP=yolo.py
ENV FLASK_ENV=production

CMD ["python", "yolo.py"]