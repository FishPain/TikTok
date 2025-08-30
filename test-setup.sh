#!/bin/bash

# Test the microservices setup without actually running them

echo "🧪 Testing TikTok AI Microservices Setup"
echo "======================================="

# Check if Docker images were built successfully
echo "🐳 Checking Docker images..."

images=(
    "tiktok-api-gateway"
    "tiktok-yolo-service" 
    "tiktok-llm-service"
)

for image in "${images[@]}"; do
    if docker image inspect "$image" > /dev/null 2>&1; then
        echo "✅ $image - Image built successfully"
    else
        echo "❌ $image - Image not found"
    fi
done

echo ""
echo "🔍 Checking configuration files..."

# Check if docker-compose.yml is valid
if docker compose config > /dev/null 2>&1; then
    echo "✅ docker-compose.yml - Valid configuration"
else
    echo "❌ docker-compose.yml - Invalid configuration"
fi

# Check if all Dockerfiles exist
dockerfiles=(
    "Dockerfile"
    "services/yolo/Dockerfile"
    "services/llm/Dockerfile"
)

for dockerfile in "${dockerfiles[@]}"; do
    if [ -f "$dockerfile" ]; then
        echo "✅ $dockerfile - Exists"
    else
        echo "❌ $dockerfile - Missing"
    fi
done

# Check if all Python files exist
python_files=(
    "app.py"
    "services/yolo/yolo.py"
    "services/llm/pii.py"
)

for py_file in "${python_files[@]}"; do
    if [ -f "$py_file" ]; then
        echo "✅ $py_file - Exists"
    else
        echo "❌ $py_file - Missing"
    fi
done

# Check if requirements files exist
req_files=(
    "requirements.txt"
    "services/yolo/requirements.txt"
    "services/llm/requirements.txt"
)

for req_file in "${req_files[@]}"; do
    if [ -f "$req_file" ]; then
        echo "✅ $req_file - Exists"
    else
        echo "❌ $req_file - Missing"
    fi
done

echo ""
echo "📋 Service Configuration Summary:"
echo "================================="
echo "• API Gateway (app.py): Port 5000"
echo "• YOLO Service: Port 7000"
echo "• LLM Service: Port 9000"
echo ""
echo "🌐 Network: ai-network (Docker bridge)"
echo ""
echo "📚 API Endpoints:"
echo "• Face Masking: POST /api/mask/face"
echo "• Location Masking: POST /api/mask/location"  
echo "• PII Masking: POST /api/mask/pii"
echo "• Health Check: GET /v1/health"
echo ""
echo "✨ Setup verification complete!"
echo ""
echo "🚀 To start services:"
echo "   ./start.sh"
echo ""
echo "🛠️  To start manually:"
echo "   docker compose up -d"
