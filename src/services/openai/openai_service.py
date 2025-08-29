from flask import Flask, request, jsonify
import openai
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/health")
def health():
    """Health check endpoint"""
    if openai.api_key:
        return jsonify({
            "status": "healthy",
            "service": "openai",
            "api_key_configured": bool(openai.api_key)
        })
    else:
        return jsonify({
            "status": "unhealthy", 
            "error": "OpenAI API key not configured"
        }), 503

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI chat completions endpoint"""
    if not openai.api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 503
        
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    try:
        response = openai.ChatCompletion.create(
            model=data.get("model", "gpt-3.5-turbo"),
            messages=data.get("messages", []),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 150),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0)
        )
        
        return jsonify({
            "choices": response["choices"],
            "usage": response["usage"],
            "model": response["model"]
        })
        
    except openai.error.AuthenticationError:
        logger.error("OpenAI authentication error")
        return jsonify({"error": "Invalid OpenAI API key"}), 401
    except openai.error.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        return jsonify({"error": "Rate limit exceeded"}), 429
    except openai.error.InvalidRequestError as e:
        logger.error(f"Invalid OpenAI request: {e}")
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def list_models():
    """List available OpenAI models"""
    if not openai.api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 503
        
    try:
        models = openai.Model.list()
        return jsonify({
            "models": [model["id"] for model in models["data"]]
        })
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
