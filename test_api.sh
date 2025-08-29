#!/bin/bash

# AI Microservices Test Script
# This script tests all the endpoints of the AI microservices

BASE_URL="http://localhost:5000"
V1_URL="$BASE_URL/v1"
API_URL="$BASE_URL/api"

echo "🧪 Testing AI Microservices Pipeline"
echo "===================================="

# Test health check
echo "1. Testing Health Check..."
curl -s -X GET "$V1_URL/health" | python3 -m json.tool
echo -e "\n"

# Test OpenAI chat (requires API key)
echo "2. Testing OpenAI Chat Completion..."
curl -s -X POST "$V1_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "temperature": 0.7,
    "max_tokens": 50
  }' | python3 -m json.tool
echo -e "\n"

# Test masking endpoints with mock data
echo "3. Testing Public Masking APIs..."
echo "================================"

# Create a small test image file (1x1 pixel PNG)
echo "Creating test image..."
echo -n -e '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01ur\xdd|\x00\x00\x00\x00IEND\xaeB`\x82' > test_image.png

echo "3a. Testing Face Masking API..."
curl -s -X POST "$API_URL/mask/face" \
  -F "file=@test_image.png" | python3 -m json.tool
echo -e "\n"

echo "3b. Testing Location Masking API..."
curl -s -X POST "$API_URL/mask/location" \
  -F "file=@test_image.png" | python3 -m json.tool
echo -e "\n"

echo "3c. Testing PII Masking API..."
OCR_DATA='[{"text": "john.doe@email.com", "bbox": [100, 200, 300, 220], "confidence": 0.95}, {"text": "Phone: 555-1234", "bbox": [100, 250, 250, 270], "confidence": 0.88}]'
curl -s -X POST "$API_URL/mask/pii" \
  -F "file=@test_image.png" \
  -F "ocr_values=$OCR_DATA" | python3 -m json.tool
echo -e "\n"

# Clean up test image
rm -f test_image.png

echo "4. Available Endpoints Summary:"
echo "   Core Services:"
echo "   - GET  $V1_URL/health (Service health check)"
echo "   - POST $V1_URL/chat/completions (OpenAI chat)"
echo ""
echo "   Masking APIs:"
echo "   - POST $API_URL/mask/face (Face detection via YOLO)"
echo "   - POST $API_URL/mask/location (Location detection via YOLO)"
echo "   - POST $API_URL/mask/pii (PII detection via OpenAI)"
echo ""

echo "✅ Test script completed!"
echo "📖 Visit http://localhost:5000 for Swagger API documentation"
echo ""
echo "🔒 Public Masking API Endpoints:"
echo "   - POST $API_URL/mask/face"
echo "   - POST $API_URL/mask/location" 
echo "   - POST $API_URL/mask/pii"
