import os
import base64
import logging
import boto3
from flask import Flask, request, jsonify
from PIL import Image
from google import genai
from google.genai import types
from uuid import uuid4
import re
from functools import wraps


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("privacy-edit-api")

app = Flask(__name__)

# API Key for authentication
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "IAMASECRET")


def require_api_key(f):
    """Decorator to check for valid API key in x-api-key header"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != API_SECRET_KEY:
            return jsonify({"error": "Unauthorized - Invalid or missing API key"}), 401
        return f(*args, **kwargs)

    return decorated_function


# Instantiate Gemini client (expects GOOGLE_API_KEY in env if needed by your setup)
client = genai.Client()

# -----------------------------
# Analysis Prompt (2-step flow)
# -----------------------------
ANALYSIS_PROMPT = """
You are a privacy-protection assistant. Analyze the given image carefully and provide:

1. **Scene Description**
   - Summarize the core environment (architecture, natural elements, objects, atmosphere).
   - Focus on the general context (e.g., urban street, residential block, park) without overfitting to location-specific identifiers.

2. **Privacy-Sensitive Details**
   - Identify elements that could compromise privacy: faces, clothing details, street names, block/unit numbers, logos, license plates, national flags, or other symbols that might enable precise geolocation.

3. **Refined Editing Prompt**
   - Generate a detailed image-editing prompt that removes or anonymizes the identified sensitive elements.
   - Use gentle, seamless techniques so anonymization blends naturally into the scene (avoid crude patches, solid black bars, or heavy blur).
   - Examples of acceptable techniques:
     • Replace PII text on signs with abstract shapes or colors that mimic typography.
     • Transform flags into simple, non-identifiable colored banners.
     • Re-illustrate faces or people as simplified, generic characters.
     • Repaint logos or plates with neutral, non-distracting textures.
     • Abstract or replace unique architectural elements with generic counterparts of similar scale and materiality.
   - Ensure the masking feels integrated into the visual style, not like censorship.

4. **Realism & Cohesion (Mandatory)**
   - The final image must appear realistic and natural, even if stylized.
   - Preserve consistent colors, lighting, materials, perspective, edge sharpness, grain/noise, and shadows so edits are not obvious.

5. **Step-by-Step Edit Plan (Mandatory, Extremely Clear)**
   - Provide a numbered list of concrete edits. Each item MUST follow this structure:
     • **What:** Element & short description (e.g., “Signboard at top-left”, “Car license plate on white van”).
     • **Where:** Location reference (e.g., “top-left near lamp post” or approximate bbox if helpful).
     • **Original → Replacement:** Explicit before/after text when applicable (e.g., “change number 123 → 634”), or “unknown/unreadable → abstract lines”.
     • **Method:** Specific technique (e.g., “repaint with neutral matte texture matching surrounding color; maintain font weight and kerning; add slight motion blur to match depth of field”).
     • **Blend Notes:** How to preserve realism (e.g., “match lighting direction, keep perspective, add subtle shadow, keep edge softness”).
   - Use non-identifying, generic replacements (randomized numerals/letters/shapes) that look plausible but cannot be traced.
   - If the original text/symbols are unreadable, state “unreadable” and propose a generic replacement that fits the scene.

**Your output must include exactly three sections:**
- **Scene Summary:** (3–6 sentences)
- **Privacy Risks:** (bullet list)
- **Refined Image-Editing Prompt:** A cohesive prompt that includes the numbered Step-by-Step Edit Plan above, ready to pass to an image editor.
"""


# -----------------------------
# Utilities
# -----------------------------
def detect_content_type_from_data_url(b64_string: str) -> str:
    match = re.match(r"^data:(.*?);base64,", b64_string)
    if match:
        return match.group(1)  # e.g. "image/png"
    return "application/octet-stream"


def upload_base64_to_s3(b64_string: str) -> str:
    """
    Uploads a base64 string to an S3 bucket and returns a URL.

    Args:
        b64_string (str): The base64 string (can include data URL prefix).

    Returns:
        str: URL to download the file
    """
    content_type = "image/jpeg"

    # Strip off data URL prefix if present
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[-1]

    # Decode into bytes
    file_bytes = base64.b64decode(b64_string)

    # Build key
    upload_prefix = os.environ.get("S3_UPLOAD_PATH", "uploads/")
    if not upload_prefix.endswith("/"):
        upload_prefix += "/"
    ext = content_type.split("/")[-1] if "/" in content_type else "bin"
    upload_key = f"{upload_prefix}{uuid4().hex}.{ext}"

    # Upload
    bucket = os.environ.get("BUCKET_NAME", "2025tiktoktechjam2025")
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=upload_key,
        Body=file_bytes,
        ContentType=content_type,
    )

    logger.info(f"✅ Uploaded {upload_key} ({content_type}) to bucket {bucket}")

    # Return the public S3 URL
    region = boto3.session.Session().region_name or "us-east-1"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{upload_key}"


def _read_image_to_pil(file_storage):
    try:
        return Image.open(file_storage.stream).convert("RGBA")
    except Exception as e:
        raise ValueError(f"Failed to read image: {e}")


def _extract_first_text_part(generation):
    """
    Safely extract the first text part from a Gemini response (if present).
    """
    try:
        for cand in generation.candidates:
            for part in cand.content.parts:
                if getattr(part, "text", None):
                    return part.text
        return None
    except Exception:
        return None


def _extract_first_inline_image(generation):
    """
    Return raw image bytes (PNG/JPEG) from the first inline_data part (if any).
    """
    try:
        for cand in generation.candidates:
            for part in cand.content.parts:
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    return bytes(part.inline_data.data)
        return None
    except Exception:
        return None


# -----------------------------
# API Endpoint
# -----------------------------
@app.route("/health", methods=["GET"])
@require_api_key
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "location-service"})


@app.route("/location/hide", methods=["POST"])
@require_api_key
def edit_image():
    """
    Multipart form-data:
      - image: file (required)
      - temperature: float (optional; default 0.8)
      - analysis_model: str (optional; default "gemini-2.5-flash")
      - image_model: str (optional; default "gemini-2.5-flash-image-preview")
    """
    try:
        temperature = 0.2
        analysis_model = os.environ.get("GEMINI_TEXT", "gemini-2.5-flash")
        image_model = os.environ.get("GEMINI_IMAGE", "gemini-2.5-flash-image-preview")

        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' file in form-data"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "Empty filename for 'image'"}), 400

        # Read input image
        pil_img = _read_image_to_pil(image_file)

        # ----- Step 1: Analyze & produce refined prompt -----
        analysis_generation = client.models.generate_content(
            model=analysis_model,
            contents=[ANALYSIS_PROMPT, pil_img],
            config=types.GenerateContentConfig(temperature=temperature),
        )

        refined_prompt = _extract_first_text_part(analysis_generation)
        if not refined_prompt:
            logger.error("No text extracted from analysis step.")
            return (
                jsonify({"error": "Analysis failed to produce a refined prompt."}),
                500,
            )

        # ----- Step 2: Generate edited image -----
        gen = client.models.generate_content(
            model=image_model,
            contents=[refined_prompt, pil_img],
            config=types.GenerateContentConfig(temperature=temperature),
        )

        img_bytes = _extract_first_inline_image(gen)
        if not img_bytes:
            # Sometimes the model returns text instead (diagnostics)
            maybe_text = _extract_first_text_part(gen)
            logger.error("No inline image found in generation step.")
            return (
                jsonify(
                    {
                        "error": "Image generation did not return an image.",
                        "refined_prompt": refined_prompt,
                        "model_output_text": maybe_text or "No additional text.",
                    }
                ),
                502,
            )

        # Encode as base64 for JSON
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        path = upload_base64_to_s3(
            image_b64,
        )
        return (
            jsonify(
                {
                    "refined_prompt": refined_prompt,
                    "image_path": path,
                    "analysis_model": analysis_model,
                    "image_model": image_model,
                    "temperature": temperature,
                }
            ),
            200,
        )

    except ValueError as ve:
        logger.exception("Bad request.")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Unhandled error.")
        return jsonify({"error": f"Internal server error: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8300, debug=True)
