from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load face detection model
try:
    # Load OpenCV's pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("Face detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load face detection model: {e}")
    face_cascade = None

@app.route("/health")
def health():
    """Health check endpoint"""
    if face_cascade is not None:
        return jsonify({
            "status": "healthy",
            "model": "opencv_haarcascade",
            "service": "face_detection"
        })
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503

@app.route("/detect", methods=["POST"])
def detect_faces():
    """Face detection endpoint"""
    if face_cascade is None:
        return jsonify({"error": "Face detection model not loaded"}), 503
        
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        # Load image
        image = Image.open(file.stream).convert("RGB")
        img_array = np.array(image)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to required format
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append([float(x), float(y), float(x + w), float(y + h)])
        
        return jsonify({
            "faces": face_boxes,
            "count": len(face_boxes),
            "model": "opencv_haarcascade"
        })
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
