# TikTok Tech Jam

This repository contains code for the TikTok Tech Jam project.

## Installation

1. Install dependencies:
    Create your venv and install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Running the Python File

To run the main Python file:
# Age Detection REST API Service

A Flask-based REST API service that provides age detection and selective face blurring functionality. This service converts the original Streamlit interface into a proper REST API while maintaining all the core functionality.

## Features

- **Face Detection**: Combines Haar cascade and YOLO models for robust face detection
- **Age Estimation**: Uses OpenCV DNN models to estimate age from detected faces
- **Minor Protection**: Automatically identifies and blurs faces of minors (under 18)
- **Additional Privacy**: Optional blurring of text, license plates, and QR codes
- **REST API**: Clean HTTP endpoints for integration with other services

## Endpoints

### GET `/health`
Health check endpoint that reports service status and available models.

### GET `/`
API documentation with endpoint descriptions and parameters.

### POST `/detect/faces`
Detect faces in an uploaded image.
- **Input**: Multipart form with `file` (image)
- **Parameters**: `yolo_conf` (optional, default: 0.4)
- **Output**: JSON with face bounding boxes

### POST `/detect/age`
Detect age for all faces in an uploaded image.
- **Input**: Multipart form with `file` (image)
- **Parameters**: `yolo_conf` (optional, default: 0.4)
- **Output**: JSON with age analysis for each face

### POST `/process/blur-minors`
Blur only faces identified as minors (under 18).
- **Input**: Multipart form with `file` (image)
- **Parameters**: 
  - `yolo_conf` (optional, default: 0.4)
  - `return_image` (optional, default: true)
- **Output**: JSON with processing results and optionally base64-encoded blurred image

### POST `/process/privacy-full`
Complete privacy processing with minor face blurring and optional text/plate/QR blurring.
- **Input**: Multipart form with `file` (image)
- **Parameters**:
  - `yolo_conf` (optional, default: 0.4)
  - `blur_text` (optional, default: false)
  - `blur_plates` (optional, default: false)
  - `blur_qr` (optional, default: false)
  - `return_image` (optional, default: true)
- **Output**: JSON with comprehensive processing results

## Models Required

### Essential Models (automatically downloaded):
- **YOLOv8n**: General object detection (`yolov8n.pt`)
- **Haar Cascade**: Face detection (built into OpenCV)
- **EasyOCR**: Text detection (downloaded automatically)

### Optional Models (provide manually):
- **Age Detection Model**: 
  - `age_deploy.prototxt` - Model architecture
  - `age_net.caffemodel` - Pre-trained weights
  - Place in `/app/models/` directory
- **YOLO Face Model**: `yolov8n-face.pt` (optional, for better face detection)

## Installation

### Docker (Recommended)

1. **Build and run the service:**
   ```bash
   docker compose up age-detection-service --build
   ```

2. **Service will be available at:** `http://localhost:5100`

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the service:**
   ```bash
   python age_detection_service.py
   ```

## Testing

Run the test script to verify all endpoints:

```bash
python test_age_api.py
```

## Usage Examples

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:5100/health

# Detect faces
curl -X POST -F 'file=@image.jpg' http://localhost:5100/detect/faces

# Detect age
curl -X POST -F 'file=@image.jpg' -F 'yolo_conf=0.4' http://localhost:5100/detect/age

# Blur minors only
curl -X POST -F 'file=@image.jpg' -F 'return_image=true' http://localhost:5100/process/blur-minors

# Full privacy processing
curl -X POST -F 'file=@image.jpg' 
     -F 'blur_text=true' 
     -F 'blur_plates=true' 
     -F 'blur_qr=true' 
     http://localhost:5100/process/privacy-full
```

### Python Example

```python
import requests

# Upload and process image
with open('family_photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5100/process/blur-minors',
        files={'file': f},
        data={'yolo_conf': 0.4, 'return_image': 'true'}
    )

result = response.json()
print(f"Minors blurred: {result['minors_blurred']}")

# Save processed image
if 'blurred_image' in result:
    import base64
    image_data = base64.b64decode(result['blurred_image'])
    with open('processed_photo.jpg', 'wb') as f:
        f.write(image_data)
```

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `FLASK_DEBUG`: Set to `False` for production

### Model Paths
- Models are expected in `/app/models/` directory
- Volumes can be mounted to provide custom models

### Performance Tuning
- Adjust `yolo_conf` parameter for detection sensitivity
- Smaller values detect more faces but may have false positives
- Larger values are more conservative but may miss faces

## Architecture

The service maintains the same core functionality as the original Streamlit application:

1. **Face Detection**: Uses both Haar cascade and YOLO for robust detection
2. **Age Classification**: OpenCV DNN model classifies faces into age buckets
3. **Minor Identification**: Probabilistic approach to identify faces under 18
4. **Selective Blurring**: Only blurs faces classified as minors
5. **Additional Privacy**: Optional detection and blurring of text, plates, QR codes

## Integration Notes

- Service runs on port 5100 by default
- All endpoints accept standard multipart form uploads
- Images can be returned as base64-encoded strings
- JSON responses include detailed detection metadata
- Service includes comprehensive error handling and logging

## Original Streamlit Conversion

This REST API service provides the same functionality as the original Streamlit application but in a service-oriented architecture suitable for:
- Microservice deployments
- API integrations
- Batch processing
- Mobile app backends
- Web service integration

## Params to change if needed:

### 1. in ```is_minor``` function 

``` p_minor_threshold ``` = how much probability mass must be in the “under 18” buckets to count as a minor.

If p_minor_threshold = 0.60 → classified as minor ✅

If p_minor_threshold = 0.70 → classified as adult ❌ (since 0.63 < 0.70)

I used 0.40 to blur uncertain cases (safer but more false positive, some adults could be blurred)