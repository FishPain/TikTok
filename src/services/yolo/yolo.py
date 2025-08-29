from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device)
    model.eval()
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

@app.route("/health")
def health():
    """Health check endpoint"""
    if model is not None:
        return jsonify({"status": "healthy", "model": "yolov5s", "device": device})
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503

@app.route("/detect", methods=["POST"])
def detect():
    """Object detection endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        # Load image
        image = Image.open(file.stream).convert("RGB")
        
        # Run inference
        results = model(image)
        
        # Parse results
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > 0.5:  # Confidence threshold
                detection = {
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                detections.append(detection)
        
        return jsonify({
            "detections": detections,
            "count": len(detections),
            "model": "yolov5s"
        })
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
