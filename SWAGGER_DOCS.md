# AI Microservices API - Swagger Documentation

## Swagger UI Features

The application automatically generates comprehensive Swagger documentation using Flask-RESTX. When you visit `http://localhost:5000`, you'll see:

### 1. **Interactive API Documentation**
- **Try it out** functionality for all endpoints
- **Request/Response examples** with real data
- **Model schemas** with validation rules
- **Authentication** information where applicable

### 2. **Organized Namespaces**

#### **v1 Namespace - Internal AI Services**
```
GET /v1/health - Service health monitoring
POST /v1/chat/completions - OpenAI chat completions
POST /v1/vision/classify/resnet - ResNet image classification
POST /v1/vision/classify/mobilenet - MobileNet image classification  
POST /v1/vision/detect/yolo - YOLO object detection
POST /v1/vision/analyze - Multi-model vision analysis
```

#### **api Namespace - Public Masking APIs**
```
POST /api/mask/face - Face detection for masking
POST /api/mask/location - Location content detection
POST /api/mask/pii - PII detection in OCR data
```

### 3. **Request/Response Models**

#### **Chat Request Model**
```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 150
}
```

#### **Mask Response Model** 
```json
{
  "data": [
    [100.5, 150.2, 200.8, 250.9],
    [300.1, 400.3, 450.7, 550.2]
  ]
}
```

#### **OCR Value Model** (for PII endpoint)
```json
{
  "text": "john.doe@email.com",
  "bbox": [100, 200, 300, 220],
  "confidence": 0.95
}
```

### 4. **File Upload Documentation**
- **Face Masking**: Upload image file
- **Location Masking**: Upload image file
- **PII Masking**: Upload image file + OCR values as form data

### 5. **Response Examples**

#### **Health Check Response**
```json
{
  "gateway_status": "healthy",
  "services": {
    "yolo": {"status": "healthy", "response_time": 0.123},
    "resnet": {"status": "healthy", "response_time": 0.089},
    "mobilenet": {"status": "healthy", "response_time": 0.105},
    "openai": {"status": "healthy", "response_time": 0.234},
    "face_detection": {"status": "healthy", "response_time": 0.067}
  }
}
```

#### **Face Detection Response**
```json
{
  "data": [
    [45.2, 78.1, 145.8, 178.9],
    [245.1, 123.4, 345.7, 223.6]
  ]
}
```

### 6. **Error Handling Documentation**
- **400 Bad Request**: Invalid input data
- **503 Service Unavailable**: Backend service down
- **500 Internal Server Error**: Processing error

## How to Access Swagger UI

1. **Start the services**:
   ```bash
   docker compose up --build
   ```

2. **Visit the Swagger UI**:
   ```
   http://localhost:5000
   ```

3. **Explore the documentation**:
   - Click on any endpoint to see details
   - Use "Try it out" to test endpoints
   - View request/response schemas
   - See example data

## OpenAPI Specification

The application generates a complete OpenAPI 3.0 specification that includes:
- **Paths**: All available endpoints
- **Components**: Reusable schemas and models
- **Tags**: Organized by service type
- **Security**: Authentication requirements
- **Examples**: Real request/response examples

You can also access the raw OpenAPI JSON at:
```
http://localhost:5000/swagger.json
```

## Testing with Swagger UI

The Swagger UI allows you to:

1. **Test Health Endpoint**: Click "Try it out" → "Execute"
2. **Test Chat Completion**: Provide JSON request body
3. **Test Image Endpoints**: Upload actual image files
4. **Test Masking APIs**: Upload images and provide OCR data

All responses are displayed directly in the UI with proper formatting and syntax highlighting.
