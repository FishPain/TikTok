import os, io, json, base64, logging, re
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
from functools import wraps

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pii_service")

# --- Constants ---
SYSTEM_PROMPT = """You are a privacy detection classifier operating under GDPR-aligned principles.

PURPOSE LIMITATION
- Sole purpose: classify each OCR item as PII or NON-PII using the image’s context to support a user-side redaction workflow.
- Do not identify people, profile them, link to external data, or infer attributes beyond this classification task.

DATA MINIMIZATION
- Consider ONLY the provided client_ocr list (each item is a single "text" string) and the image context.
- You MUST return the OCR text **exactly as provided**, character-for-character.
- Do not merge, split, normalize, reorder, reformat, or synthesize new items.
- One OCR string = one possible pii_items entry if classified as PII.

CLASSIFICATION CRITERIA
Treat as **PII** when the OCR text clearly refers to a private individual:
- Personal full names.
- Government IDs (e.g., NRIC/FIN/passport).
- Personal phone numbers.
- Residential/home addresses.
- Personal emails.
- Bank/credit card/account numbers.

Treat as **NON-PII** when the OCR text is clearly business/venue/public information:
- Business/venue names, addresses, phone numbers.
- Generic corporate emails (info@, sales@).
- Ratings, prices, menus, opening hours.

Contextual rules:
- If text appears in a business listing → NON-PII.
- If text appears on an ID card, form, bill, or personal doc → PII.
- If uncertain, prefer NON-PII.

SPECIAL CATEGORIES (GDPR Art. 9)
- If text directly reveals sensitive categories (health, religion, political views, etc.), classify as PII with type "Other" and mention “special-category” in the reason.

STRICT OUTPUT RULES
- Output MUST follow this JSON schema exactly:
{
  "pii_items": [
    {
      "text": "exact OCR text as provided in client_ocr (no edits)",
      "type": "Name|Phone|Address|Email|ID|Account|Other",
      "reason": "brief explanation"
    }
  ]
}
- "text" must match one OCR item exactly, not a concatenation or reformatted version.
- Include one pii_items entry per OCR text classified as PII.
- Exclude OCR items classified as NON-PII.

FEW-SHOT EXAMPLES
Example 1: OCR tokens for a date of birth
Input: ["10", "june", "2020"]
Output:
{
  "pii_items": [
    {"text": "10", "type": "Other", "reason": "Part of a date of birth"},
    {"text": "june", "type": "Other", "reason": "Part of a date of birth"},
    {"text": "2020", "type": "Other", "reason": "Part of a date of birth"}
  ]
}

Example 2: OCR tokens for a residential address
Input: ["123", "Main", "Street", "Singapore"]
Output:
{
  "pii_items": [
    {"text": "123", "type": "Address", "reason": "Part of a residential address"},
    {"text": "Main", "type": "Address", "reason": "Part of a residential address"},
    {"text": "Street", "type": "Address", "reason": "Part of a residential address"},
    {"text": "Singapore", "type": "Address", "reason": "Part of a residential address"}
  ]
}

Example 3: OCR tokens for a personal name
Input: ["Tan", "Kai"]
Output:
{
  "pii_items": [
    {"text": "Tan", "type": "Name", "reason": "Part of a personal full name"},
    {"text": "Kai", "type": "Name", "reason": "Part of a personal full name"}
  ]
}

Example 4: OCR tokens for an ID number
Input: ["S1234567D"]
Output:
{
  "pii_items": [
    {"text": "S1234567D", "type": "ID", "reason": "Singapore NRIC/ID number"}
  ]
}
"""

# --- NEW: Vulnerability prompt (face / location / PII) ---
VULN_PROMPT = """You are a privacy risk classifier for images.

Task: From the image (and optional OCR context), decide which of the following vulnerability labels apply:
- "face": a clearly visible human face (frontal or near-frontal; recognizable).
- "location": elements that could reveal a precise physical location (e.g., readable street or unit numbers, unique storefront/school names, license plates tied to a place, apartment/condo names with identifiable surroundings, maps/GPS screenshots, postal labels).
- "PII": personal information visible in the image that could identify a private individual (e.g., full name, personal phone, personal email, home address, government IDs, bank/credit card/account numbers). Do NOT extract or return any text—just assess presence.

Rules:
- Prefer fewer false positives; if uncertain, omit the label.
- Do NOT transcribe, quote, or infer identities.
- Return ONLY the JSON below, nothing else.

Return strictly:
{ "vulnerabilities": ["face", "location", "PII"] }

If none apply, return:
{ "vulnerabilities": [] }
"""


# --- Structured output models ---
class PIIItem(BaseModel):
    text: str = Field(..., description="Exact OCR text as provided in client_ocr")
    type: Literal["Name", "Phone", "Address", "Email", "ID", "Account", "Other"]
    reason: str


class PIIOutput(BaseModel):
    pii_items: List[PIIItem] = Field(default_factory=list)


# --- NEW: Vulnerability output model ---
class VulnerabilityOutput(BaseModel):
    vulnerabilities: List[Literal["face", "location", "PII"]] = Field(
        default_factory=list
    )


# --- Helpers ---
def normalize_image(file_stream) -> Image.Image:
    im = Image.open(file_stream)
    im = ImageOps.exif_transpose(im)
    im.thumbnail((1024, 1024))
    return im.convert("RGB")


def image_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# --- LLM ---
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20"), temperature=0)

# --- Flask App ---
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


@app.route("/health", methods=["GET"])
@require_api_key
def health():
    return jsonify({"status": "ok", "model": llm.model_name})


@app.route("/classify/pii", methods=["POST"])
@require_api_key
def classify_pii():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' (image)"}), 400

    if "ocr" not in request.form:
        return jsonify({"error": "Missing 'ocr' (JSON list of strings)"}), 400

    file = request.files["file"]
    ocr_raw = request.form["ocr"]

    # --- Validate image ---
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({"error": "Only jpg and png images are supported"}), 400

    try:
        # Parse OCR list of strings
        try:
            ocr_items = json.loads(ocr_raw)
        except Exception:
            return jsonify({"error": "OCR must be valid JSON list of strings"}), 400

        if not isinstance(ocr_items, list) or not all(
            isinstance(x, str) for x in ocr_items
        ):
            return jsonify({"error": "'ocr' must be a JSON list of strings"}), 400

        # Process image
        im = normalize_image(file.stream)
        img_data_url = image_to_data_url(im)

        # Build prompt input
        user_text = "client_ocr list:\n" + json.dumps(ocr_items, ensure_ascii=False)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            HumanMessage(
                content=[
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ]
            ),
        ]

        # Call LLM
        response = llm.invoke(messages)

        raw = response.content.strip()
        try:
            result_dict = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON: {raw}")

        # Validate with schema
        result = PIIOutput(**result_dict)
        return jsonify(result.dict())

    except Exception as e:
        logger.exception("Classification failed")
        return jsonify({"error": str(e)}), 500


@app.route("/classify/vulnerability", methods=["POST"])
@require_api_key
def classify_vulnerability():
    """
    POST /classify/vulnerability
    Form-data:
      - file: image (jpg/png)

    Returns:
      { "vulnerabilities": ["face", "location", "PII"] }
      or
      { "vulnerabilities": [] }
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' (image)"}), 400

    file = request.files["file"]

    if not (file.filename.lower().endswith((".jpg", ".jpeg", ".png"))):
        return jsonify({"error": "Only jpg and png images are supported"}), 400

    try:
        im = normalize_image(file.stream)
        img_data_url = image_to_data_url(im)

        # Build prompt payload
        content = [{"type": "image_url", "image_url": {"url": img_data_url}}]

        messages = [
            {"role": "system", "content": VULN_PROMPT},
            HumanMessage(content=content),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()
        try:
            result_dict = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON: {raw}")

        result = VulnerabilityOutput(**result_dict)
        return jsonify(result.model_dump())

    except Exception as e:
        logger.exception("Vulnerability classification failed")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
@require_api_key
def index():
    """API documentation"""
    return jsonify(
        {
            "service": "PII Classification Service",
            "version": "1.3",
            "description": "Classifies OCR text elements from an image as PII or NON-PII and detects image privacy vulnerabilities.",
            "endpoints": {
                "GET /health": {
                    "description": "Health check endpoint",
                    "response": {"status": "ok", "model": "<LLM model name>"},
                },
                "POST /classify/pii": {
                    "description": "Classify OCR items in an image as PII or NON-PII",
                    "content_type": "multipart/form-data",
                    "form_fields": {
                        "file": "Required. Image file (jpg or png).",
                        "ocr": "Required. OCR JSON file containing list of objects with keys: "
                        "{'text': str, 'bbox': [x1,y1,x2,y2]}.",
                    },
                    "response": {
                        "pii_items": [
                            {
                                "text": "<OCR text>",
                                "type": "Name | Phone | Address | Email | ID | Account | Other",
                                "reason": "Short GDPR-aware reason",
                            }
                        ]
                    },
                    "example_curl": (
                        "curl -X POST http://localhost:8200/classify/pii "
                        '-F "file=@test.jpg" '
                        '-F "ocr=@client_ocr.json"'
                    ),
                },
                "POST /classify/vulnerability": {
                    "description": "Identify if the image shows a human face, reveals precise location, and/or contains visible PII.",
                    "content_type": "multipart/form-data",
                    "form_fields": {
                        "file": "Required. Image file (jpg or png).",
                        "ocr": "Optional. OCR JSON list to improve detection of 'PII' and 'location'.",
                    },
                    "response": {"vulnerabilities": ["face", "location", "PII"]},
                    "example_curl": (
                        "curl -X POST http://localhost:8200/classify/vulnerability "
                        '-F "file=@test.jpg" '
                        '-F "ocr=@client_ocr.json"'
                    ),
                },
            },
            "notes": [
                "The service requires OPENAI_API_KEY as an environment variable.",
                "Supported image formats: jpg, png.",
                "For /classify/pii, OCR JSON must be a valid array of objects with at least 'text' and 'bbox'.",
                "For /classify/vulnerability, OCR is optional but can improve 'PII' and 'location' signals.",
            ],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8200, debug=True)
