from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import requests
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load MobileNet model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    mobilenet = models.mobilenet_v3_large(pretrained=True).to(device)
    mobilenet.eval()
    logger.info("MobileNet model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load MobileNet model: {e}")
    mobilenet = None

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ImageNet labels
try:
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    imagenet_classes = requests.get(LABELS_URL, timeout=10).text.strip().split("\n")
    logger.info("ImageNet classes loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ImageNet classes: {e}")
    imagenet_classes = [f"class_{i}" for i in range(1000)]  # Fallback

@app.route("/health")
def health():
    """Health check endpoint"""
    if mobilenet is not None:
        return jsonify({
            "status": "healthy", 
            "model": "mobilenet_v3_large", 
            "device": device,
            "classes_loaded": len(imagenet_classes)
        })
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503

@app.route("/predict", methods=["POST"])
def predict():
    """Image classification endpoint"""
    if mobilenet is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = mobilenet(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            predictions = []
            for i in range(top5_prob.size(0)):
                predictions.append({
                    "class": imagenet_classes[top5_catid[i]],
                    "confidence": float(top5_prob[i])
                })

        return jsonify({
            "predictions": predictions,
            "model": "mobilenet_v3_large"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
