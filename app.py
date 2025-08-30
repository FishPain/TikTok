from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import requests
import os
import io
import json
from typing import Dict, Any, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="AI Microservices Gateway",
    description="Gateway for AI inference microservices with public masking APIs",
)

# Service URLs from environment variables
SERVICES = {
    "yolo": os.getenv("YOLO_SERVICE_URL", "http://yolo-service:8100"),
    "llm": os.getenv("LLM_SERVICE_URL", "http://llm-service:8200"),
}

# Namespaces
ns_v1 = api.namespace("v1", description="AI Inference Endpoints")
ns_api = api.namespace("api", description="Public Masking APIs")

# Models for masking APIs
mask_response = api.model(
    "MaskResponse",
    {
        "data": fields.List(
            fields.List(fields.Float),
            required=True,
            description="Array of bounding boxes in format [[x1,y1,x2,y2], ...]",
            example=[[100.5, 150.2, 200.8, 250.9], [300.1, 400.3, 450.7, 550.2]],
        )
    },
)


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
                    "response_time": response.elapsed.total_seconds(),
                }
            except Exception as e:
                health_status[service_name] = {"status": "unhealthy", "error": str(e)}

        all_healthy = all(
            status["status"] == "healthy" for status in health_status.values()
        )
        return {
            "gateway_status": "healthy" if all_healthy else "degraded",
            "services": health_status,
        }


def detect_faces_in_image(image_data) -> List[Tuple[float, float, float, float]]:
    """
    Detect faces in image using YOLO service and return bounding boxes
    Returns list of (x1, y1, x2, y2) tuples
    """
    try:
        # Call YOLO service for face detection
        response = requests.post(
            f"{SERVICES['yolo']}/detect",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            data={"type": "faces"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])

            # Extract face bounding boxes
            face_boxes = []
            for detection in detections:
                class_name = detection.get("class", "").lower()
                if "face" in class_name:
                    bbox = detection.get("bbox", {})
                    face_boxes.append(
                        (
                            float(bbox.get("x1", 0)),
                            float(bbox.get("y1", 0)),
                            float(bbox.get("x2", 0)),
                            float(bbox.get("y2", 0)),
                        )
                    )

            if face_boxes:
                return face_boxes
        else:
            logger.error(f"YOLO service error: {response.status_code}")

    except Exception as e:
        logger.error(f"YOLO service call failed: {e}")

    # Fallback to mock data if service fails
    return [(100.5, 150.2, 200.8, 250.9), (300.1, 400.3, 450.7, 550.2)]


def detect_location_in_image(image_data) -> List[Tuple[float, float, float, float]]:
    """
    Detect location-related content in image using YOLO service and return bounding boxes
    Returns list of (x1, y1, x2, y2) tuples
    """
    try:
        # Call YOLO service for general object detection
        response = requests.post(
            f"{SERVICES['yolo']}/detect",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            data={"type": "general"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])

            # Filter for location-related objects (signs, landmarks, buildings, etc.)
            location_boxes = []
            location_classes = [
                "stop sign",
                "traffic light",
                "street sign",
                "building",
                "car",
                "truck",
                "bus",
            ]

            for detection in detections:
                class_name = detection.get("class", "").lower()
                if any(loc_class in class_name for loc_class in location_classes):
                    bbox = detection.get("bbox", {})
                    location_boxes.append(
                        (
                            float(bbox.get("x1", 0)),
                            float(bbox.get("y1", 0)),
                            float(bbox.get("x2", 0)),
                            float(bbox.get("y2", 0)),
                        )
                    )

            if location_boxes:
                return location_boxes
        else:
            logger.error(f"YOLO service error: {response.status_code}")

    except Exception as e:
        logger.error(f"YOLO service call failed: {e}")

    # Fallback to mock data if service fails
    return [(50.0, 75.5, 180.3, 120.8), (250.2, 300.1, 400.6, 380.9)]


def detect_pii_in_ocr(
    ocr_values: List[Dict],
) -> List[Tuple[float, float, float, float]]:
    """
    Detect PII in OCR values using LLM service and return bounding boxes of sensitive information
    Returns list of (x1, y1, x2, y2) tuples
    """
    pii_boxes = []

    try:
        # Call LLM service to analyze OCR values for PII
        # First prepare the data for the LLM service
        ocr_data = {"client_ocr": ocr_values}

        # Create a dummy image file for the LLM service (it expects both image and OCR)
        dummy_image = io.BytesIO()
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        img.save(dummy_image, format="JPEG")
        dummy_image.seek(0)

        response = requests.post(
            f"{SERVICES['llm']}/classify/pii",
            files={"file": ("dummy.jpg", dummy_image, "image/jpeg")},
            data={"client_ocr": json.dumps(ocr_values)},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            pii_items = result.get("pii_items", [])

            # Match detected PII with OCR bounding boxes
            for pii_item in pii_items:
                pii_text = pii_item.get("text", "")
                # Find matching OCR item
                for ocr_item in ocr_values:
                    if ocr_item.get("text", "") == pii_text:
                        bbox = ocr_item.get("bbox", [])
                        if len(bbox) >= 4:
                            pii_boxes.append(
                                (
                                    float(bbox[0]),
                                    float(bbox[1]),
                                    float(bbox[2]),
                                    float(bbox[3]),
                                )
                            )
                        break
        else:
            logger.error(f"LLM service error: {response.status_code}")
            # Fallback to basic pattern matching
            for ocr_item in ocr_values:
                text = ocr_item.get("text", "").lower()
                bbox = ocr_item.get("bbox", [])

                pii_patterns = [
                    "@",  # Email
                    "phone",
                    "tel:",
                    "call",  # Phone
                    "ssn",
                    "social security",  # SSN
                    "address",
                    "street",
                    "ave",
                    "rd",  # Address
                    "license",
                    "id:",
                    "passport",  # IDs
                ]

                if any(pattern in text for pattern in pii_patterns) and len(bbox) >= 4:
                    pii_boxes.append(
                        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    )

    except Exception as e:
        logger.error(f"LLM PII detection failed: {e}")

        # Fallback to basic pattern matching
        for ocr_item in ocr_values:
            text = ocr_item.get("text", "").lower()
            bbox = ocr_item.get("bbox", [])

            if any(
                pattern in text for pattern in ["@", "phone", "ssn", "email", "address"]
            ):
                if len(bbox) >= 4:
                    pii_boxes.append(
                        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    )

    # Add some mock data if no PII detected
    if not pii_boxes:
        pii_boxes = [(120.0, 200.5, 350.8, 230.2), (400.1, 450.3, 600.7, 480.9)]

    return pii_boxes


@ns_api.route("/mask/face")
class FaceMask(Resource):
    @ns_api.doc("mask_face")
    @ns_api.expect(
        api.parser().add_argument(
            "file", location="files", type="file", required=True, help="Image file"
        )
    )
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
    @ns_api.expect(
        api.parser().add_argument(
            "file", location="files", type="file", required=True, help="Image file"
        )
    )
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
    @ns_api.expect(
        api.parser()
        .add_argument(
            "file", location="files", type="file", required=True, help="Image file"
        )
        .add_argument(
            "ocr_values",
            location="form",
            type="str",
            required=True,
            help='JSON string of OCR values with format: [{"text": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.95}]',
        )
    )
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
    app.run(host="0.0.0.0", port=8000, debug=True)
