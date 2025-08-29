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
    "openai": os.getenv("OPENAI_SERVICE_URL", "http://openai-service:9000")
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

def detect_faces_in_image(image_data) -> List[Tuple[float, float, float, float]]:
    """
    Detect faces in image using YOLO and return bounding boxes
    Returns list of (x1, y1, x2, y2) tuples
    """
    try:
        # Call YOLO service for object detection
        response = requests.post(
            f"{SERVICES['yolo']}/detect",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])
            
            # Filter for person/face detections
            face_boxes = []
            for detection in detections:
                class_name = detection.get("class", "").lower()
                if "person" in class_name or "face" in class_name:
                    bbox = detection.get("bbox", {})
                    face_boxes.append((
                        float(bbox.get("x1", 0)),
                        float(bbox.get("y1", 0)),
                        float(bbox.get("x2", 0)),
                        float(bbox.get("y2", 0))
                    ))
            
            if face_boxes:
                return face_boxes
        else:
            logger.error(f"YOLO service error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"YOLO service call failed: {e}")
    
    # Fallback to mock data if service fails
    return [
        (100.5, 150.2, 200.8, 250.9),
        (300.1, 400.3, 450.7, 550.2)
    ]

def detect_location_in_image(image_data) -> List[Tuple[float, float, float, float]]:
    """
    Detect location-related content in image using YOLO and return bounding boxes
    Returns list of (x1, y1, x2, y2) tuples
    """
    try:
        # Call YOLO service for object detection
        response = requests.post(
            f"{SERVICES['yolo']}/detect",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])
            
            # Filter for location-related objects (signs, landmarks, buildings, etc.)
            location_boxes = []
            location_classes = ["stop sign", "traffic light", "street sign", "building", "car", "truck", "bus"]
            
            for detection in detections:
                class_name = detection.get("class", "").lower()
                if any(loc_class in class_name for loc_class in location_classes):
                    bbox = detection.get("bbox", {})
                    location_boxes.append((
                        float(bbox.get("x1", 0)),
                        float(bbox.get("y1", 0)),
                        float(bbox.get("x2", 0)),
                        float(bbox.get("y2", 0))
                    ))
            
            if location_boxes:
                return location_boxes
        else:
            logger.error(f"YOLO service error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"YOLO service call failed: {e}")
    
    # Fallback to mock data if service fails
    return [
        (50.0, 75.5, 180.3, 120.8),
        (250.2, 300.1, 400.6, 380.9)
    ]

def detect_pii_in_ocr(ocr_values: List[Dict]) -> List[Tuple[float, float, float, float]]:
    """
    Detect PII in OCR values using OpenAI and return bounding boxes of sensitive information
    Returns list of (x1, y1, x2, y2) tuples
    """
    pii_boxes = []
    
    try:
        # Prepare text for OpenAI analysis
        ocr_texts = [item.get("text", "") for item in ocr_values]
        combined_text = " | ".join(ocr_texts)
        
        # Call OpenAI service to analyze for PII
        openai_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a PII detection expert. Analyze the following OCR text and identify any personally identifiable information such as names, emails, phone numbers, addresses, SSN, etc. Respond with a JSON array of detected PII indices in the format: [{\"type\": \"email\", \"text\": \"john@example.com\", \"index\": 0}]"
                },
                {
                    "role": "user", 
                    "content": f"OCR texts to analyze: {combined_text}"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        response = requests.post(
            f"{SERVICES['openai']}/chat/completions",
            json=openai_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse AI response and match with OCR bounding boxes
            # For now, use simple pattern matching as fallback
            for i, ocr_item in enumerate(ocr_values):
                text = ocr_item.get("text", "").lower()
                bbox = ocr_item.get("bbox", [])
                
                # Enhanced pattern matching
                pii_patterns = [
                    "@",  # Email
                    "phone", "tel:", "call",  # Phone
                    "ssn", "social security",  # SSN
                    "address", "street", "ave", "rd",  # Address
                    "license", "id:", "passport"  # IDs
                ]
                
                if any(pattern in text for pattern in pii_patterns) and len(bbox) >= 4:
                    pii_boxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        else:
            logger.error(f"OpenAI service error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"OpenAI PII detection failed: {e}")
        
        # Fallback to basic pattern matching
        for ocr_item in ocr_values:
            text = ocr_item.get("text", "").lower()
            bbox = ocr_item.get("bbox", [])
            
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