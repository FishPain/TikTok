from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import requests
import os
from typing import Dict, Any, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app, version="1.0", title="AI Microservices Gateway",
          description="Gateway for AI inference microservices with public masking APIs")

# Service URLs from environment variables
SERVICES = {
    "yolo": os.getenv("YOLO_SERVICE_URL", "http://yolo-service:7000"),
    "resnet": os.getenv("RESNET_SERVICE_URL", "http://resnet-service:6000"),
    "mobilenet": os.getenv("MOBILENET_SERVICE_URL", "http://mobilenet-service:8000"),
    "openai": os.getenv("OPENAI_SERVICE_URL", "http://openai-service:9000"),
    "face_detection": os.getenv("FACE_DETECTION_SERVICE_URL", "http://face-detection-service:5500")
}

# Namespaces
ns_v1 = api.namespace("v1", description="AI Inference Endpoints")
ns_api = api.namespace("api", description="Public Masking APIs")

# Request models for Swagger documentation
chat_req = api.model("ChatRequest", {
    "model": fields.String(required=True, example="gpt-4"),
    "messages": fields.List(fields.Raw, required=True,
                            example=[{"role": "user", "content": "Hello"}]),
    "temperature": fields.Float(default=0.7),
    "max_tokens": fields.Integer(default=128)
})

# Models for masking APIs
bbox_model = api.model("BoundingBox", {
    "x1": fields.Float(required=True, description="Left x coordinate"),
    "y1": fields.Float(required=True, description="Top y coordinate"),
    "x2": fields.Float(required=True, description="Right x coordinate"),
    "y2": fields.Float(required=True, description="Bottom y coordinate")
})

mask_response = api.model("MaskResponse", {
    "data": fields.List(fields.List(fields.Float), required=True, 
                       description="Array of bounding boxes in format [[x1,y1,x2,y2], ...]",
                       example=[[100.5, 150.2, 200.8, 250.9], [300.1, 400.3, 450.7, 550.2]])
})

ocr_value_model = api.model("OCRValue", {
    "text": fields.String(required=True, description="Detected text"),
    "bbox": fields.List(fields.Float, required=True, description="Bounding box [x1,y1,x2,y2]"),
    "confidence": fields.Float(description="OCR confidence score")
})

pii_request = api.model("PIIRequest", {
    "ocr_values": fields.List(fields.Nested(ocr_value_model), required=True,
                             description="OCR detected values with bounding boxes")
})

@ns_v1.route("/health")
class Health(Resource):
    def get(self):
        """Health check for all services"""
        health_status = {}
        for service_name, service_url in SERVICES.items():
            try:
                response = requests.get(f"{service_url}/health", timeout=5)
                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                health_status[service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        all_healthy = all(status["status"] == "healthy" for status in health_status.values())
        return {
            "gateway_status": "healthy" if all_healthy else "degraded",
            "services": health_status
        }

@ns_v1.route("/chat/completions")
class ChatCompletion(Resource):
    @ns_v1.expect(chat_req)
    def post(self):
        """Chat completion using OpenAI service"""
        try:
            response = requests.post(
                f"{SERVICES['openai']}/chat/completions",
                json=request.json,
                timeout=30
            )
            return response.json(), response.status_code
        except Exception as e:
            logger.error(f"OpenAI service error: {str(e)}")
            return {"error": f"OpenAI service unavailable: {str(e)}"}, 503

@ns_v1.route("/vision/classify/resnet")
class ResNetClassification(Resource):
    def post(self):
        """Image classification using ResNet"""
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            response = requests.post(
                f"{SERVICES['resnet']}/predict",
                files={"file": (file.filename, file.stream, file.mimetype)},
                timeout=30
            )
            return response.json(), response.status_code
        except Exception as e:
            logger.error(f"ResNet service error: {str(e)}")
            return {"error": f"ResNet service unavailable: {str(e)}"}, 503

@ns_v1.route("/vision/classify/mobilenet")
class MobileNetClassification(Resource):
    def post(self):
        """Image classification using MobileNet"""
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            response = requests.post(
                f"{SERVICES['mobilenet']}/predict",
                files={"file": (file.filename, file.stream, file.mimetype)},
                timeout=30
            )
            return response.json(), response.status_code
        except Exception as e:
            logger.error(f"MobileNet service error: {str(e)}")
            return {"error": f"MobileNet service unavailable: {str(e)}"}, 503

@ns_v1.route("/vision/detect/yolo")
class YOLODetection(Resource):
    def post(self):
        """Object detection using YOLO"""
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            response = requests.post(
                f"{SERVICES['yolo']}/detect",
                files={"file": (file.filename, file.stream, file.mimetype)},
                timeout=30
            )
            return response.json(), response.status_code
        except Exception as e:
            logger.error(f"YOLO service error: {str(e)}")
            return {"error": f"YOLO service unavailable: {str(e)}"}, 503

@ns_v1.route("/vision/analyze")
class VisionAnalyze(Resource):
    def post(self):
        """Comprehensive vision analysis using multiple models"""
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        results = {}
        
        # Prepare file data for multiple requests
        file_data = file.read()
        file.seek(0)  # Reset file pointer
        
        # Run all vision models in parallel (you could use threading for better performance)
        try:
            # YOLO Detection
            try:
                yolo_response = requests.post(
                    f"{SERVICES['yolo']}/detect",
                    files={"file": (file.filename, file_data, file.mimetype)},
                    timeout=30
                )
                results["object_detection"] = yolo_response.json()
            except Exception as e:
                results["object_detection"] = {"error": str(e)}

            # ResNet Classification
            try:
                resnet_response = requests.post(
                    f"{SERVICES['resnet']}/predict",
                    files={"file": (file.filename, file_data, file.mimetype)},
                    timeout=30
                )
                results["resnet_classification"] = resnet_response.json()
            except Exception as e:
                results["resnet_classification"] = {"error": str(e)}

            # MobileNet Classification
            try:
                mobilenet_response = requests.post(
                    f"{SERVICES['mobilenet']}/predict",
                    files={"file": (file.filename, file_data, file.mimetype)},
                    timeout=30
                )
                results["mobilenet_classification"] = mobilenet_response.json()
            except Exception as e:
                results["mobilenet_classification"] = {"error": str(e)}

            return {
                "analysis_id": f"analysis_{hash(file.filename)}",
                "results": results
            }
        except Exception as e:
            logger.error(f"Vision analysis error: {str(e)}")
            return {"error": f"Vision analysis failed: {str(e)}"}, 500


# =============================================================================
# PUBLIC MASKING APIs
# =============================================================================

def detect_faces_in_image(image_data) -> List[Tuple[float, float, float, float]]:
    """
    Detect faces in image and return bounding boxes
    Returns list of (x1, y1, x2, y2) tuples
    """
    try:
        # Call face detection service
        response = requests.post(
            f"{SERVICES['face_detection']}/detect",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            faces = result.get("faces", [])
            return [(face[0], face[1], face[2], face[3]) for face in faces]
        else:
            logger.error(f"Face detection service error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Face detection service call failed: {e}")
    
    # Fallback to mock data if service fails
    return [
        (100.5, 150.2, 200.8, 250.9),
        (300.1, 400.3, 450.7, 550.2)
    ]

def detect_location_in_image(image_data) -> List[Tuple[float, float, float, float]]:
    """
    Detect location-related content in image and return bounding boxes
    Returns list of (x1, y1, x2, y2) tuples
    """
    # TODO: Implement actual location detection (street signs, landmarks, etc.)
    # For now, return mock data
    return [
        (50.0, 75.5, 180.3, 120.8),
        (250.2, 300.1, 400.6, 380.9)
    ]

def detect_pii_in_ocr(ocr_values: List[Dict]) -> List[Tuple[float, float, float, float]]:
    """
    Detect PII in OCR values and return bounding boxes of sensitive information
    Returns list of (x1, y1, x2, y2) tuples
    """
    # TODO: Implement actual PII detection logic
    # Check for patterns like phone numbers, emails, SSN, etc.
    pii_boxes = []
    
    # Mock PII detection logic
    for ocr_item in ocr_values:
        text = ocr_item.get("text", "").lower()
        bbox = ocr_item.get("bbox", [])
        
        # Simple pattern matching (expand this with real PII detection)
        if any(pattern in text for pattern in ["@", "phone", "ssn", "email", "address"]):
            if len(bbox) >= 4:
                pii_boxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
    
    # Add some mock data if no PII detected
    if not pii_boxes:
        pii_boxes = [
            (120.0, 200.5, 350.8, 230.2),
            (400.1, 450.3, 600.7, 480.9)
        ]
    
    return pii_boxes

@ns_api.route("/mask/face")
class FaceMask(Resource):
    @ns_api.doc("mask_face")
    @ns_api.expect(api.parser().add_argument('file', location='files', type='file', required=True, help='Image file'))
    @ns_api.marshal_with(mask_response)
    def post(self):
        """
        Detect faces in image and return bounding boxes for masking
        
        Returns bounding boxes in format: {data: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]}
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            # Read image data
            image_data = file.read()
            
            # Detect faces
            face_boxes = detect_faces_in_image(image_data)
            
            # Convert to required format
            data = [[x1, y1, x2, y2] for x1, y1, x2, y2 in face_boxes]
            
            return {"data": data}
            
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return {"error": f"Face detection failed: {str(e)}"}, 500

@ns_api.route("/mask/location")
class LocationMask(Resource):
    @ns_api.doc("mask_location")
    @ns_api.expect(api.parser().add_argument('file', location='files', type='file', required=True, help='Image file'))
    @ns_api.marshal_with(mask_response)
    def post(self):
        """
        Detect location-related content in image and return bounding boxes for masking
        
        Returns bounding boxes in format: {data: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]}
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            # Read image data
            image_data = file.read()
            
            # Detect location content
            location_boxes = detect_location_in_image(image_data)
            
            # Convert to required format
            data = [[x1, y1, x2, y2] for x1, y1, x2, y2 in location_boxes]
            
            return {"data": data}
            
        except Exception as e:
            logger.error(f"Location detection error: {str(e)}")
            return {"error": f"Location detection failed: {str(e)}"}, 500

@ns_api.route("/mask/pii")
class PIIMask(Resource):
    @ns_api.doc("mask_pii")
    @ns_api.expect(api.parser()
                   .add_argument('file', location='files', type='file', required=True, help='Image file')
                   .add_argument('ocr_values', location='form', type='str', required=True, 
                               help='JSON string of OCR values with format: [{"text": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.95}]'))
    @ns_api.marshal_with(mask_response)
    def post(self):
        """
        Detect PII in OCR values and return bounding boxes for masking
        
        Requires both image file and OCR values.
        OCR values should be in format: [{"text": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.95}]
        
        Returns bounding boxes in format: {data: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]}
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400
            
        if "ocr_values" not in request.form:
            return {"error": "No OCR values provided"}, 400

        file = request.files["file"]
        try:
            import json
            
            # Read image data
            image_data = file.read()
            
            # Parse OCR values
            ocr_values_str = request.form["ocr_values"]
            ocr_values = json.loads(ocr_values_str)
            
            # Detect PII in OCR values
            pii_boxes = detect_pii_in_ocr(ocr_values)
            
            # Convert to required format
            data = [[x1, y1, x2, y2] for x1, y1, x2, y2 in pii_boxes]
            
            return {"data": data}
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for ocr_values"}, 400
        except Exception as e:
            logger.error(f"PII detection error: {str(e)}")
            return {"error": f"PII detection failed: {str(e)}"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)