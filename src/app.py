from flask import Flask, request
from flask_restx import Api, Resource, fields
import requests
import logging
import os
import json
from functools import wraps
from helper import (
    detect_faces_in_image,
    detect_location_in_image,
    detect_pii_in_ocr,
    build_privacy_masks,
)


# Service URLs from environment variables
SERVICES = {
    "yolo": os.getenv("YOLO_SERVICE_URL", "http://yolo-service:8100"),
    "llm": os.getenv("LLM_SERVICE_URL", "http://llm-service:8200"),
    "location": os.getenv("LOCATION_SERVICE_URL", "http://location-service:8300"),
}

# API Key for authentication
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "IAMASECRET")


def require_api_key(f):
    """Decorator to check for valid API key in x-api-key header"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != API_SECRET_KEY:
            return {"error": "Unauthorized - Invalid or missing API key"}, 401
        return f(*args, **kwargs)

    return decorated_function


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


authorizations = {"apikey": {"type": "apiKey", "in": "header", "name": "x-api-key"}}

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="PII Classification Service",
    description="Classifies OCR text and image vulnerabilities",
    authorizations=authorizations,
    security="apikey",  # <-- requires api key by default
)

# Namespaces
ns_v1 = api.namespace("v1", description="AI Inference Endpoints")
ns_api = api.namespace("api", description="Public Masking APIs")

# Models for masking APIs - Standardized structure
mask_item_model = api.model(
    "MaskItem",
    {
        "coordinate": fields.String(
            required=True,
            description='Coordinate string in the form "(x1, x2, y1, y2)"',
            example="(100.5, 200.8, 150.2, 250.9)",
        ),
        "reason": fields.String(
            required=True,
            description="Reason for masking (may be empty string)",
            example="visible face",
        ),
    },
)

mask_response = api.model(
    "MaskResponse",
    {
        "mask": fields.List(
            fields.Nested(mask_item_model),
            required=True,
            description="List of mask items with coordinates and reasons",
        )
    },
)

mask_item_str_model = api.model(
    "MaskItemStr",
    {
        "coordinate": fields.String(
            required=True,
            description='Coordinate string in the form "(x1, x2, y1, y2)"',
            example="(100.5, 200.8, 150.2, 250.9)",
        ),
        "reason": fields.String(
            required=True,
            description="Reason for masking (may be empty string)",
            example="visible face",
        ),
    },
)

mask_group_model = api.model(
    "MaskGroup",
    {
        "mask": fields.List(
            fields.Nested(mask_item_str_model),
            required=True,
            description="List of mask items for this vulnerability type",
        )
    },
)

privacy_masks_response = api.model(
    "PrivacyMasksResponse",
    {
        "face": fields.Nested(
            mask_group_model,
            allow_null=True,
            description="Face masks or null",
        ),
        "location": fields.String(
            required=False,
            description="Regenerated image URL or base64 string (or null)",
        ),
        "pii": fields.Nested(
            mask_group_model,
            allow_null=True,
            description="PII masks or null",
        ),
    },
)


@ns_v1.route("/health")
class Health(Resource):
    @require_api_key
    def get(self):
        """Health check for all services"""
        health_status = {}
        for service_name, service_url in SERVICES.items():
            try:
                response = requests.get(
                    f"{service_url}/health",
                    timeout=5,
                    headers={"x-api-key": API_SECRET_KEY},
                )
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


@ns_api.route("/mask/face")
class FaceMask(Resource):
    @ns_api.doc("mask_face")
    @ns_api.expect(
        api.parser().add_argument(
            "file", location="files", type="file", required=True, help="Image file"
        )
    )
    @ns_api.marshal_with(mask_response)
    @require_api_key
    def post(self):
        """
        Detect faces in image and return structured mask data

        Returns structured mask data with coordinates and reasons in format:
        {mask: [{coordinate: "(x1, x2, y1, y2)", reason: "visible face"}, ...]}
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            # Read image data
            image_data = file.read()

            # Detect faces
            face_boxes = detect_faces_in_image(image_data)

            # Convert to standardized format using helper function
            from helper import _format_masks

            result = _format_masks(face_boxes, reason="visible face")

            # Return consistent structure (empty mask array if no detections)
            return result or {"mask": []}

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
    @require_api_key
    def post(self):
        """
        Detect location-related content in image and return structured mask data

        Returns structured mask data with coordinates and reasons in format:
        {mask: [{coordinate: "(x1, x2, y1, y2)", reason: "location-revealing object"}, ...]}
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        try:
            # Read image data
            image_data = file.read()

            # Detect location content
            generated_image = detect_location_in_image(image_data)
            return {"image": generated_image}

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
            "ocr",
            location="form",
            type="str",
            required=True,
            help='JSON string of OCR values with format: {"text": ["word1", "word2"], "bbox": [[x1,y1,x2,y2], [x1,y1,x2,y2]]}',
        )
    )
    @ns_api.marshal_with(mask_response)
    @require_api_key
    def post(self):
        """
        Detect PII in OCR values and return structured mask data

        Requires both image file and OCR values.
        OCR values should be in format: {"text": ["word1", "word2"], "bbox": [[x1,y1,x2,y2], [x1,y1,x2,y2]]}

        Returns structured mask data with coordinates and reasons in format:
        {mask: [{coordinate: "(x1, x2, y1, y2)", reason: "potential personal info"}, ...]}
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        if "ocr" not in request.form:
            return {"error": "No OCR values provided"}, 400

        file = request.files["file"]
        ocr_values = request.form["ocr"]

        try:
            # Read image data
            image_data = file.read()

            # Parse OCR values
            try:
                ocr_data = json.loads(ocr_values)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format for ocr values"}, 400

            # Detect PII in OCR values
            pii_boxes = detect_pii_in_ocr(image_data, ocr_data)

            # Convert to standardized format using helper function
            from helper import _format_masks

            result = _format_masks(pii_boxes, reason="potential personal info")

            # Return consistent structure (empty mask array if no detections)
            return result or {"mask": []}

        except Exception as e:
            logger.error(f"PII detection error: {str(e)}")
            return {"error": f"PII detection failed: {str(e)}"}, 500


@ns_api.route("/mask/all")
class PrivacyPipeline(Resource):
    @ns_api.doc("mask_all")
    @ns_api.expect(
        api.parser().add_argument(
            "file", location="files", type="file", required=True, help="Image file"
        )
    )
    @ns_api.marshal_with(privacy_masks_response, skip_none=False)
    @require_api_key
    def post(self):
        """
        Detect all privacy-related content and return masks in structured form.

        This endpoint combines the results of /mask/face, /mask/location, and /mask/pii
        into a single response with the same structure format:

        Each category will be null if no vulnerabilities of that type are detected,
        or will contain a mask array with coordinate and reason for each detection.
        """
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]

        try:
            image_data = file.read()

            # Build structured masks via your orchestrator
            data = build_privacy_masks(image_data)

            # `data` must be a dict with keys face/location/pii -> {mask:[{coordinate,reason}]} or None
            return data

        except Exception as e:
            logger.error(f"Privacy detection error: {str(e)}")
            return {"error": f"Privacy detection failed: {str(e)}"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
