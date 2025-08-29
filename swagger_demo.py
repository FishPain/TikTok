"""
Swagger Configuration Demo
This file shows all the Swagger/OpenAPI configurations used in the main app.py
"""

from flask import Flask
from flask_restx import Api, Resource, fields

# Flask-RESTX automatically generates Swagger documentation
app = Flask(__name__)

# API configuration with metadata for Swagger
api = Api(
    app, 
    version="1.0", 
    title="AI Microservices Gateway",
    description="Gateway for AI inference microservices with public masking APIs",
    doc="/",  # Swagger UI served at root
    contact="support@example.com",
    license="MIT",
    license_url="https://opensource.org/licenses/MIT"
)

# Namespaces organize endpoints in Swagger UI
ns_v1 = api.namespace("v1", description="AI Inference Endpoints")
ns_api = api.namespace("api", description="Public Masking APIs")

# Request/Response models for Swagger documentation
chat_req = api.model("ChatRequest", {
    "model": fields.String(
        required=True, 
        example="gpt-4",
        description="OpenAI model to use"
    ),
    "messages": fields.List(
        fields.Raw, 
        required=True,
        example=[{"role": "user", "content": "Hello"}],
        description="Chat messages array"
    ),
    "temperature": fields.Float(
        default=0.7,
        description="Randomness in response (0-1)"
    ),
    "max_tokens": fields.Integer(
        default=128,
        description="Maximum tokens in response"
    )
})

# Bounding box model
bbox_model = api.model("BoundingBox", {
    "x1": fields.Float(required=True, description="Left x coordinate"),
    "y1": fields.Float(required=True, description="Top y coordinate"), 
    "x2": fields.Float(required=True, description="Right x coordinate"),
    "y2": fields.Float(required=True, description="Bottom y coordinate")
})

# Mask response model
mask_response = api.model("MaskResponse", {
    "data": fields.List(
        fields.List(fields.Float), 
        required=True,
        description="Array of bounding boxes in format [[x1,y1,x2,y2], ...]",
        example=[[100.5, 150.2, 200.8, 250.9], [300.1, 400.3, 450.7, 550.2]]
    )
})

# OCR value model for PII endpoint
ocr_value_model = api.model("OCRValue", {
    "text": fields.String(
        required=True, 
        description="Detected text",
        example="john.doe@email.com"
    ),
    "bbox": fields.List(
        fields.Float, 
        required=True, 
        description="Bounding box [x1,y1,x2,y2]",
        example=[100, 200, 300, 220]
    ),
    "confidence": fields.Float(
        description="OCR confidence score",
        example=0.95
    )
})

# Example endpoint with full Swagger documentation
@ns_api.route("/mask/face")
class FaceMask(Resource):
    @ns_api.doc(
        "mask_face",
        description="Detect faces in uploaded image and return bounding boxes for masking purposes"
    )
    @ns_api.expect(
        api.parser().add_argument(
            'file', 
            location='files', 
            type='file', 
            required=True, 
            help='Image file (JPEG, PNG supported)'
        )
    )
    @ns_api.marshal_with(mask_response)
    @ns_api.response(200, "Success", mask_response)
    @ns_api.response(400, "Bad Request - No file provided")
    @ns_api.response(500, "Internal Server Error")
    def post(self):
        """
        Face Detection for Masking
        
        Analyzes uploaded image to detect human faces and returns bounding box coordinates
        for each detected face. Coordinates are in format [x1, y1, x2, y2] where:
        - x1, y1: top-left corner
        - x2, y2: bottom-right corner
        
        Use these coordinates to apply masks or blur effects to protect privacy.
        """
        return {"data": [[100.5, 150.2, 200.8, 250.9]]}

if __name__ == "__main__":
    print("Swagger Demo Configuration")
    print("=" * 50)
    print(f"API Title: {api.title}")
    print(f"API Version: {api.version}")
    print(f"API Description: {api.description}")
    print(f"Swagger UI URL: http://localhost:5000{api.doc}")
    print(f"OpenAPI JSON: http://localhost:5000/swagger.json")
    print()
    print("Features enabled:")
    print("✓ Interactive Swagger UI")
    print("✓ Request/Response models")
    print("✓ File upload support")
    print("✓ Organized namespaces")
    print("✓ Detailed documentation")
    print("✓ Example data")
    print("✓ Error response codes")
