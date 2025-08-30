import os, io, json, base64, logging, re
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pii_service")

# --- Constants ---
SYSTEM_PROMPT = """You are a privacy detection classifier operating under GDPR-aligned principles.

PURPOSE LIMITATION
- Sole purpose: classify each OCR item as PII or NON-PII using the image’s context to support a user-side redaction workflow.
- Do not identify people, profile them, link to external data, or infer attributes beyond this classification task.

DATA MINIMIZATION
- Consider ONLY the provided client_ocr list (each item: {"text": str, "bbox": [x1,y1,x2,y2]}) and the image context.
- Do not invent new strings or alter text. Return only what the output schema requires.

CLASSIFICATION CRITERIA
Treat as **PII** when the text clearly refers to a private individual:
- Personal full names.
- Government IDs (e.g., NRIC/FIN/passport).
- Personal phone numbers.
- Residential/home addresses.
- Personal emails.
- Bank/credit card/account numbers.

Treat as **NON-PII** when the text is clearly business/venue/public information:
- Business/venue names, addresses, phone numbers.
- Generic corporate emails (info@, sales@).
- Ratings, prices, menus, opening hours.

Contextual rules:
- If text appears in a business listing → NON-PII.
- If text appears on an ID card, form, bill, or personal doc → PII.
- If uncertain, prefer NON-PII.

SPECIAL CATEGORIES (GDPR Art. 9)
- If text directly reveals sensitive categories (health, religion, political views, etc.), classify as PII with type "Other" and mention “special-category” in the reason.

Return output strictly in this JSON schema:
{
  "pii_items": [
    {
      "text": "exact OCR text",
      "type": "Name|Phone|Address|Email|ID|Account|Other",
      "reason": "brief explanation"
    }
  ]
}
"""


# --- Structured output models ---
class PIIItem(BaseModel):
    text: str = Field(..., description="Exact OCR text as provided in client_ocr")
    type: Literal["Name", "Phone", "Address", "Email", "ID", "Account", "Other"]
    reason: str


class PIIOutput(BaseModel):
    pii_items: List[PIIItem] = Field(default_factory=list)


# --- Helpers ---
def normalize_image(file_stream) -> Image.Image:
    im = Image.open(file_stream)
    im = ImageOps.exif_transpose(im)
    # resize safeguard
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": llm.model_name})


@app.route("/classify/pii", methods=["POST"])
def classify_pii():
    """
    POST /classify/pii
    Form-data:
      - file: image (jpg/png)
      - ocr: OCR JSON (as file upload)
    """
    if "file" not in request.files or "ocr" not in request.files:
        return jsonify({"error": "Missing 'file' (image) or 'ocr' (JSON)"}), 400

    file = request.files["file"]
    ocr_file = request.files["ocr"]

    # --- Validate image type ---
    if not (file.filename.lower().endswith((".jpg", ".jpeg", ".png"))):
        return jsonify({"error": "Only jpg and png images are supported"}), 400

    try:
        # Load image
        im = normalize_image(file.stream)
        img_data_url = image_to_data_url(im)

        # Load OCR JSON
        try:
            ocr_items = json.load(ocr_file)
        except Exception:
            return jsonify({"error": "OCR must be valid JSON"}), 400

        user_text = "client_ocr list:\n" + json.dumps(ocr_items, ensure_ascii=False)

        # Build messages
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

        # Parse JSON
        raw = response.content.strip()
        try:
            result_dict = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON: {raw}")

        result = PIIOutput(**result_dict)
        return jsonify(result.dict())

    except Exception as e:
        logger.exception("Classification failed")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """API documentation"""
    return jsonify(
        {
            "service": "PII Classification Service",
            "version": "1.2",
            "description": "Classifies OCR text elements from an image as PII or NON-PII "
            "using a GDPR-aligned structured output model (via OpenAI multimodal).",
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
                        "curl -X POST http://localhost:5200/classify/pii "
                        '-F "file=@test.jpg" '
                        '-F "ocr=@client_ocr.json"'
                    ),
                },
            },
            "notes": [
                "The service requires OPENAI_API_KEY as an environment variable.",
                "Supported image formats: jpg, png.",
                "OCR JSON must be valid JSON array of objects with at least 'text' and 'bbox'.",
            ],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
