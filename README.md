# TikTok AI Microservices

A microservices architecture for AI-powered content masking with face detection, location detection, and PII (Personally Identifiable Information) detection.

## Architecture

This project consists of three main services:

1. **API Gateway** (`app.py`) - Main REST API with Swagger documentation
2. **YOLO Service** (`services/yolo/`) - Computer vision service for object and face detection
3. **LLM Service** (`services/llm/`) - Language model service for PII detection in OCR text

## Features

- **Face Masking**: Detect faces in images and return bounding boxes for masking
- **Location Masking**: Detect location-related objects (signs, vehicles, etc.) for privacy
- **PII Detection**: Analyze OCR text using AI to identify personally identifiable information
- **Microservice Architecture**: Each service runs independently in Docker containers
- **Health Monitoring**: Built-in health checks for all services
- **Swagger Documentation**: Interactive API documentation

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API Key

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/TikTok
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

3. **Start all services:**
   ```bash
   ./start.sh
   ```

   Or manually:
   ```bash
   docker compose up --build
   ```

### Service Endpoints

Once running, you can access:

- **API Gateway**: http://localhost:5000 (Swagger UI available)
- **YOLO Service**: http://localhost:7000 
- **LLM Service**: http://localhost:9000

## API Endpoints

### Public Masking APIs

#### Face Masking
```http
POST /api/mask/face
Content-Type: multipart/form-data

file: <image_file>
```
Returns face bounding boxes in format: `{data: [[x1,y1,x2,y2], ...]}`

#### Location Masking  
```http
POST /api/mask/location
Content-Type: multipart/form-data

file: <image_file>
```
Returns location object bounding boxes in format: `{data: [[x1,y1,x2,y2], ...]}`

#### PII Masking
```http
POST /api/mask/pii
Content-Type: multipart/form-data

file: <image_file>
ocr_values: <json_string>
```
OCR values format:
```json
[
  {
    "text": "john@example.com",
    "bbox": [100, 200, 300, 220],
    "confidence": 0.95
  }
]
```
Returns PII bounding boxes in format: `{data: [[x1,y1,x2,y2], ...]}`

### Health Check
```http
GET /v1/health
```
Returns health status of all services.

## Development

### Project Structure
```
├── app.py                 # API Gateway
├── docker-compose.yml     # Service orchestration
├── Dockerfile            # Gateway container
├── requirements.txt      # Gateway dependencies
├── services/
│   ├── yolo/            # Computer vision service
│   │   ├── yolo.py      # YOLO service implementation
│   │   ├── Dockerfile   # YOLO container
│   │   ├── requirements.txt
│   │   └── models/      # AI models
│   └── llm/             # Language model service
│       ├── pii.py       # PII detection service
│       ├── Dockerfile   # LLM container
│       └── requirements.txt
```

### Service Communication

Services communicate via HTTP within the Docker network:
- API Gateway → YOLO Service: `http://yolo-service:7000`
- API Gateway → LLM Service: `http://llm-service:9000`

### Environment Variables

- `OPENAI_API_KEY`: Required for LLM service
- `YOLO_SERVICE_URL`: Internal YOLO service URL
- `LLM_SERVICE_URL`: Internal LLM service URL

### Useful Commands

```bash
# View logs
docker compose logs -f

# Restart specific service
docker compose restart yolo-service

# Stop all services
docker compose down

# Rebuild and restart
docker compose up --build

# Check service health
curl http://localhost:5000/v1/health
```

## Troubleshooting

1. **OpenAI API Key Issues**: Ensure your API key is properly set in `.env`
2. **Service Not Starting**: Check logs with `docker-compose logs <service-name>`
3. **Port Conflicts**: Ensure ports 5000, 7000, and 9000 are available
4. **Model Loading**: YOLO models are downloaded automatically on first run

## Contributing

1. Each service should be independently deployable
2. Use environment variables for configuration
3. Include health check endpoints
4. Follow REST API conventions
5. Add proper error handling and logging

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

4. **Age Detection Service** (`age-detection-service`) - Port 5100
   - Combines YOLO and age classification models
   - Selective face blurring for minor protection
   - Advanced privacy features with text/plate/QR detection

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
        ├── openai/
        │   ├── Dockerfile.openai
        │   ├── requirements.txt
        │   └── openai_service.py
        └── age/
            ├── Dockerfile
            ├── requirements.txt
            ├── age_detection_service.py
            ├── test_age_api.py
            └── README.md
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