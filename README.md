# AI Microservices Inference Pipeline

A streamlined microservice architecture for AI-powered masking APIs with YOLO object detection and OpenAI integration.

## Architecture Overview

This project implements a focused microservice-based AI inference pipeline optimized for masking applications:

### Services

1. **API Gateway** (`api-gateway`) - Port 5000
   - Main entry point for all AI inference requests
   - Routes requests to appropriate microservices
   - Provides health monitoring and service orchestration
   - Built with Flask and Flask-RESTX (Swagger documentation)

2. **YOLO Object Detection Service** (`yolo-service`) - Port 7000
   - YOLOv5 model for object detection
   - Powers face detection and location masking
   - Returns bounding boxes, class labels, and confidence scores

3. **OpenAI Service** (`openai-service`) - Port 9000
   - Proxy service for OpenAI API calls
   - Handles chat completions and PII detection
   - Manages API key and error handling

## Project Structure

```
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile                  # API Gateway container
├── requirements.txt            # API Gateway dependencies
├── README.md                   # This file
└── src/
    ├── app.py                  # API Gateway application
    └── services/
        ├── yolo/
        │   ├── Dockerfile.yolo
        │   ├── requirements.txt
        │   └── yolo.py
        └── openai/
            ├── Dockerfile.openai
            ├── requirements.txt
            └── openai_service.py
```
        ├── resnet/
        │   ├── Dockerfile.resnet
        │   ├── requirements.txt
        │   └── resnet.py
        ├── mobilenet/
        │   ├── Dockerfile.mobilenet
        │   ├── requirements.txt
        │   └── mobilenet.py
        └── openai/
            ├── Dockerfile.openai
            ├── requirements.txt
            └── openai_service.py
```

## API Endpoints

### Health Check
- `GET /v1/health` - Check health status of all services

### Chat Completions
- `POST /v1/chat/completions` - OpenAI chat completions

### 🔒 Public Masking APIs
- `POST /api/mask/face` - Detect faces using YOLO and return bounding boxes for masking
- `POST /api/mask/location` - Detect location content using YOLO and return bounding boxes for masking  
- `POST /api/mask/pii` - Detect PII using OpenAI analysis and return bounding boxes for masking

## Setup and Deployment

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (for OpenAI service)

### Environment Variables
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Quick Start
1. Clone the repository
2. Set up environment variables
3. Build and run all services:
```bash
docker-compose up --build
```

### Individual Service Development
To run a specific service for development:
```bash
# Build specific service
docker-compose build yolo-service

# Run specific service
docker-compose up yolo-service
```

## Usage Examples

### Health Check
```bash
curl -X GET http://localhost:5000/v1/health
```

### Public Masking APIs

#### Face Masking
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/mask/face
```
Response format:
```json
{
  "data": [
    [100.5, 150.2, 200.8, 250.9],
    [300.1, 400.3, 450.7, 550.2]
  ]
}
```

#### Location Masking
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/mask/location
```
Response format:
```json
{
  "data": [
    [50.0, 75.5, 180.3, 120.8],
    [250.2, 300.1, 400.6, 380.9]
  ]
}
```

#### PII Masking
```bash
curl -X POST \
  -F "file=@image.jpg" \
  -F 'ocr_values=[{"text": "john@email.com", "bbox": [100, 200, 300, 220], "confidence": 0.95}]' \
  http://localhost:5000/api/mask/pii
```
Response format:
```json
{
  "data": [
    [120.0, 200.5, 350.8, 230.2],
    [400.1, 450.3, 600.7, 480.9]
  ]
}
```

### Chat Completion
```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

## API Documentation

Once the services are running, you can access the interactive API documentation at:
- **Swagger UI**: http://localhost:5000/

## Service Details

### YOLO Service
- **Model**: YOLOv5s (small version for faster inference)
- **Input**: Image files (JPEG, PNG)
- **Output**: Object detections with bounding boxes and confidence scores
- **Confidence Threshold**: 0.5 (configurable)
- **Use Cases**: Face detection, location masking (signs, vehicles, buildings)

### OpenAI Service
- **Models**: All OpenAI models (gpt-3.5-turbo, gpt-4, etc.)
- **Input**: Chat messages in OpenAI format
- **Output**: Generated responses with usage statistics
- **Use Cases**: Chat completions, PII detection in OCR text

## Monitoring and Logging

- All services include structured logging
- Health check endpoints for monitoring
- Error handling with appropriate HTTP status codes
- Request/response logging for debugging

## Scaling and Performance

- Each service runs independently and can be scaled separately
- GPU support available for vision models (CUDA)
- Stateless design for horizontal scaling
- Docker networking for efficient inter-service communication

## Security Considerations

- API keys managed through environment variables
- Service isolation through Docker containers
- Internal network communication between services
- Input validation and error handling

## Development

### Adding New Models
1. Create a new service directory under `src/services/`
2. Implement Flask application with health check and prediction endpoints
3. Add Dockerfile and requirements.txt
4. Update docker-compose.yml
5. Add routes to API gateway

### Testing
Each service can be tested independently or through the API gateway.

## Troubleshooting

### Common Issues
1. **Model loading failures**: Check GPU/CPU compatibility and model downloads
2. **Service communication**: Verify Docker network configuration
3. **API key issues**: Ensure OpenAI API key is properly set in environment

### Logs
Check service logs:
```bash
docker compose logs yolo-service
docker compose logs openai-service
docker compose logs api-gateway
```

## License

This project is open source and available under the [MIT License](LICENSE).