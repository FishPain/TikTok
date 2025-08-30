# run_gpt_classify_lc.py
# Usage:
#   python run_gpt_classify_lc.py --image path/to/img.jpg --ocr client_ocr.json --out pii_result.json
#
# Requirements:
#   pip install langchain langchain-openai pillow pydantic
#   export OPENAI_API_KEY=...

import os, io, json, base64, argparse
from PIL import Image, ImageOps
from typing import List, Literal, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

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
"""

# --- Structured output models ---

class PIIItem(BaseModel):
    text: str = Field(..., description="Exact OCR text as provided in client_ocr")
    type: Literal["Name", "Phone", "Address", "Email", "ID", "Account", "Other"] = Field(..., description="Type of PII detected")
    reason: str = Field(..., description="Short GDPR-aware reason for classification")

class PIIOutput(BaseModel):
    pii_items: List[PIIItem] = Field(default_factory=list, description="List of PII items detected")

# --- Helpers ---

def normalize_image(path: str) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")

def image_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--ocr", required=True, help="Path to OCR JSON file")
    ap.add_argument("--out", default="pii_result.json", help="Output JSON path")
    ap.add_argument("--model", default="gpt-4o-2024-11-20", help="OpenAI model to use")
    args = ap.parse_args()

    # Load inputs
    im = normalize_image(args.image)
    img_data_url = image_to_data_url(im)
    with open(args.ocr, "r", encoding="utf-8") as f:
        ocr_items = json.load(f)

    # Build prompt
    user_text = (
        "client_ocr list (index = position in this array):\n"
        + json.dumps(ocr_items, ensure_ascii=False)
    )

    parser = PydanticOutputParser(pydantic_object=PIIOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nReturn output in the required JSON schema only."),
        ("user", [
            {"type": "text", "text": "{user_text}"},
            {"type": "image_url", "image_url": {"url": "{img_url}"}}
        ])
    ])

    chain = prompt | ChatOpenAI(model=args.model, temperature=0) | parser

    try:
        result: PIIOutput = chain.invoke({"user_text": user_text, "img_url": img_data_url})
    except Exception as e:
        print(f"[ERROR] Parsing failed: {e}")
        result = PIIOutput(pii_items=[])

    # Save
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result.dict(), f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out}")
    print(json.dumps(result.dict(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()