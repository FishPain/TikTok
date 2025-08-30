#!/bin/bash

# Startup script for TikTok AI Microservices

echo "🚀 Starting TikTok AI Microservices..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from template..."
    cp .env.template .env
    echo "📝 Please edit .env file with your OpenAI API key before running again."
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "^OPENAI_API_KEY=sk-" .env; then
    echo "⚠️  OpenAI API key not properly set in .env file."
    echo "📝 Please add your OpenAI API key to the .env file."
    exit 1
fi

echo "✅ Environment variables configured"

# Build and start services
echo "🏗️  Building Docker images..."
docker compose build

echo "🚀 Starting all services..."
docker compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

# Check API Gateway
if curl -f http://localhost:8000/v1/health > /dev/null 2>&1; then
    echo "✅ API Gateway is healthy (port 8000)"
else
    echo "❌ API Gateway is not responding (port 8000)"
fi

# Check YOLO Service
if curl -f http://localhost:8100/health > /dev/null 2>&1; then
    echo "✅ YOLO Service is healthy (port 8100)"
else
    echo "❌ YOLO Service is not responding (port 8100)"
fi

# Check LLM Service
if curl -f http://localhost:8200/health > /dev/null 2>&1; then
    echo "✅ LLM Service is healthy (port 8200)"
else
    echo "❌ LLM Service is not responding (port 8200)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📊 Service URLs:"
echo "  • API Gateway (Swagger UI): http://localhost:8000"
echo "  • YOLO Service: http://localhost:8100"
echo "  • LLM Service: http://localhost:8200"
echo ""
echo "🔧 Useful commands:"
echo "  • View logs: docker compose logs -f"
echo "  • Stop services: docker compose down"
echo "  • Restart services: docker compose restart"
echo ""
