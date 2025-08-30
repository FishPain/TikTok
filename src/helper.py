import requests
import json
import os
import boto3
import io
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Tuple, Optional, Set, Union
import logging
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SERVICES = {
    "yolo": os.getenv("YOLO_SERVICE_URL", "http://yolo-service:8100"),
    "llm": os.getenv("LLM_SERVICE_URL", "http://llm-service:8200"),
    "location": os.getenv("LOCATION_SERVICE_URL", "http://location-service:8300"),
}

# API Key for service-to-service communication
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "IAMASECRET")

# Common headers for all service requests
HEADERS = {"x-api-key": API_SECRET_KEY}

# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_UPLOAD_PATH = os.getenv("S3_UPLOAD_PATH", "privacy-masks")

# Initialize S3 client if credentials are available
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and BUCKET_NAME:
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION,
        )
        logger.info("S3 client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize S3 client: {e}")
        s3_client = None
else:
    logger.warning("S3 credentials not configured")


def apply_masks_to_image(
    image_data: bytes,
    face_boxes: List[Tuple[float, float, float, float]],
    pii_boxes: List[Tuple[float, float, float, float]] = None,
) -> bytes:
    """
    Apply blur masks to the image for face and PII detection areas.
    Returns the processed image as bytes.
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_data))

        # Create a copy to work with
        masked_image = image.copy()

        # Apply face masks (blur)
        if face_boxes:
            masked_image = _apply_blur_masks(masked_image, face_boxes)

        # Apply PII masks (black rectangles)
        if pii_boxes:
            masked_image = _apply_black_masks(masked_image, pii_boxes)

        # Convert back to bytes
        output_buffer = io.BytesIO()
        masked_image.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()

    except Exception as e:
        logger.error(f"Error applying masks to image: {e}")
        return image_data  # Return original if masking fails


def _apply_blur_masks(
    image: Image.Image, boxes: List[Tuple[float, float, float, float]]
) -> Image.Image:
    """Apply blur effect to specified regions"""
    from PIL import ImageFilter

    for x1, y1, x2, y2 in boxes:
        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                f"Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2}), skipping"
            )
            continue

        # Ensure coordinates are within image bounds
        width, height = image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))

        # Skip if region is too small
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        try:
            # Extract the region
            region = image.crop((x1, y1, x2, y2))

            # Apply blur
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=15))

            # Paste back
            image.paste(blurred_region, (x1, y1))
        except Exception as e:
            logger.warning(f"Failed to blur region ({x1}, {y1}, {x2}, {y2}): {e}")

    return image


def _apply_black_masks(
    image: Image.Image, boxes: List[Tuple[float, float, float, float]]
) -> Image.Image:
    """Apply black rectangles to specified regions"""
    draw = ImageDraw.Draw(image)

    for x1, y1, x2, y2 in boxes:
        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                f"Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2}), skipping"
            )
            continue

        # Ensure coordinates are within image bounds
        width, height = image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))

        # Skip if region is too small
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue

        try:
            # Draw black rectangle
            draw.rectangle([x1, y1, x2, y2], fill="black")
        except Exception as e:
            logger.warning(f"Failed to draw black mask ({x1}, {y1}, {x2}, {y2}): {e}")

    return image


def upload_to_s3(image_data: bytes, filename: str = None) -> Optional[str]:
    """
    Upload image to S3 and return the public URL.
    Returns None if S3 is not configured or upload fails.
    """
    if not s3_client or not BUCKET_NAME:
        logger.warning("S3 not configured, cannot upload image")
        return None

    try:
        # Generate unique filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"masked_image_{timestamp}_{unique_id}.jpg"

        # Construct S3 key
        s3_key = f"{S3_UPLOAD_PATH}/{filename}" if S3_UPLOAD_PATH else filename

        # Upload to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=image_data,
            ContentType="image/jpeg",
            ACL="public-read",  # Make publicly accessible
        )

        # Return public URL
        s3_url = f"https://{BUCKET_NAME}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Image uploaded to S3: {s3_url}")
        return s3_url

    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return None


# --- existing helpers (unchanged) ---
def detect_faces_in_image(image_data: bytes) -> List[Tuple[float, float, float, float]]:
    try:
        response = requests.post(
            f"{SERVICES['yolo']}/detect/age",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            data={"type": "faces"},
            headers=HEADERS,
            timeout=180,
        )
        if response.status_code == 200:
            results = response.json()
            detections = results.get("faces", [])
            face_boxes = []
            for det in detections:
                # logger.info(f"YOLO detection: {det}")
                if det.get("is_minor", False) is True:
                    bbox = det.get("bbox", {})
                    face_boxes.append(
                        (
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        )
                    )
            return face_boxes
        else:
            logger.error(f"YOLO service error: {response.status_code}")
    except Exception as e:
        logger.error(f"YOLO service call failed: {e}")
    return []


def detect_location_in_image(
    image_data: bytes,
) -> List[Tuple[float, float, float, float]]:
    try:
        response = requests.post(
            f"{SERVICES['location']}/location/hide",
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"type": "general"},
            headers=HEADERS,
            timeout=180,
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("image_path", "")
        else:
            logger.error(f"Location service error: {response.status_code}")
    except Exception as e:
        logger.error(f"Location service call failed: {e}")
    return ""


def detect_pii_in_ocr(
    image_data: bytes,
    ocr_values: Union[Dict[str, Any], str],
) -> List[Tuple[float, float, float, float]]:
    """
    Calls LLM /classify/pii and maps returned PII texts back to OCR bboxes.
    Accepts OCR as dict with text/bbox keys, or JSON string.
    Returns list of (x1, y1, x2, y2) tuples.
    """

    # Handle different input formats
    if isinstance(ocr_values, str):
        try:
            ocr_data = json.loads(ocr_values)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for OCR values")
            return []
    else:
        ocr_data = ocr_values

    if isinstance(ocr_data, dict):
        ocr_text = ocr_data.get("text", [])
    else:
        # Fallback for legacy format
        ocr_text = []

    # Encode list as JSON string for form field
    form = {"ocr": json.dumps(ocr_text, ensure_ascii=False)}

    files = {"file": ("image.jpg", image_data, "image/jpeg")}

    resp = requests.post(
        f"{SERVICES['llm']}/classify/pii",
        files=files,
        data=form,
        headers=HEADERS,
        timeout=180,
    )

    if resp.status_code != 200:
        logger.error(f"LLM service error: {resp.status_code} - {resp.text}")
        return []

    try:
        payload = resp.json()
        pii_items = payload.get("pii_items", [])
        return pii_to_bbox_list(pii_items, ocr_data)

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return []


def pii_to_bbox_list(
    pii_items: List[Dict[str, Any]], ocr_result: Dict[str, Any]
) -> List[Tuple[float, float, float, float]]:
    """
    Map detected PII items back to OCR bboxes and return as list of tuples.
    Returns list of (x1, y1, x2, y2) tuples.
    """
    ocr_texts = ocr_result.get("text", [])
    ocr_bboxes = ocr_result.get("bbox", [])

    text_to_boxes: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for t, box in zip(ocr_texts, ocr_bboxes):
        if len(box) >= 4:
            # Convert from [x, y, width, height] to [x1, y1, x2, y2]
            x, y, width, height = box[:4]
            x1, y1, x2, y2 = x, y, x + width, y + height
            text_to_boxes.setdefault(t, []).append(
                (float(x1), float(y1), float(x2), float(y2))
            )

    # Collect all boxes from PII items
    all_boxes: List[Tuple[float, float, float, float]] = []
    for pii in pii_items:
        t = pii.get("text")
        if not t:
            continue
        for box in text_to_boxes.get(t, []):
            all_boxes.append(box)

    return _unique_boxes(all_boxes)


def pii_to_masks(
    pii_items: List[Dict[str, Any]],
    ocr_result: Dict[str, Any],
    image_data: bytes = None,
) -> Optional[Dict[str, Any]]:
    """
    Map detected PII items back to OCR bboxes and convert into mask format.
    Returns S3 URL of masked image or coordinate format as fallback.
    """
    boxes = pii_to_bbox_list(pii_items, ocr_result)
    reason = "potential personal info"
    return _format_masks(boxes, reason=reason, image_data=image_data, mask_type="pii")


# --- ask LLM which vulnerability types exist ---
def get_vulnerabilities(image_data: bytes) -> List[str]:
    """
    Call /classify/vulnerability. Returns list like ["face", "location", "PII"] (may be empty).
    """
    files = {"file": ("image.jpg", image_data, "image/jpeg")}

    try:
        resp = requests.post(
            f"{SERVICES['llm']}/classify/vulnerability",
            files=files,
            headers=HEADERS,
            timeout=180,
        )
        if resp.status_code == 200:
            payload = resp.json()
            return payload.get("vulnerabilities", []) or []
        else:
            logger.error(
                f"/classify/vulnerability error: {resp.status_code} - {resp.text}"
            )
    except Exception as e:
        logger.error(f"/classify/vulnerability failed: {e}")
    return []


def _unique_boxes(
    boxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    """Deduplicate exact duplicate boxes while preserving order."""
    seen: Set[Tuple[float, float, float, float]] = set()
    out: List[Tuple[float, float, float, float]] = []
    for b in boxes:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _format_masks(
    boxes: List[Tuple[float, float, float, float]],
    reason: str = "",
    image_data: bytes = None,
    mask_type: str = "face",
) -> Optional[Dict[str, Any]]:
    """
    Apply masks to image and upload to S3, returning S3 URL instead of coordinates.
    Returns None if no boxes or if processing fails.
    """
    boxes = _unique_boxes(boxes)
    if not boxes or not image_data:
        return None

    try:
        # Apply appropriate masks based on type
        if mask_type == "face":
            masked_image_data = apply_masks_to_image(image_data, boxes)
        elif mask_type == "pii":
            masked_image_data = apply_masks_to_image(image_data, [], boxes)
        else:
            masked_image_data = apply_masks_to_image(image_data, boxes)

        # Upload to S3
        s3_url = upload_to_s3(masked_image_data)

        if s3_url and mask_type != "face":
            return s3_url
        else:
            return ""

    except Exception as e:
        logger.error(f"Error processing masks: {e}")
        # Fallback to coordinate format
        return {
            "mask": [
                {"coordinate": f"({x1}, {y1}, {x2}, {y2})", "reason": reason}
                for (x1, y1, x2, y2) in boxes
            ]
        }


def extract_ocr_values(image_data: bytes) -> Optional[List[Dict[str, Any]]]:
    """
    Extract OCR values from the image using the OCR service.
    """
    files = {"file": ("image.jpg", image_data, "image/jpeg")}
    try:
        resp = requests.post(
            f"{SERVICES['yolo']}/process/text",
            files=files,
            headers=HEADERS,
            timeout=180,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.error(f"/process/text error: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"/process/text failed: {e}")
    return None


def build_privacy_masks(
    image_data: bytes,
) -> Dict[str, Any]:
    """
    1) Query /classify/vulnerability to know which types are present.
    2) Run the corresponding detectors.
    3) Return the consolidated structure with null for types not identified (or identified but no boxes).
    """
    vulnerabilities = get_vulnerabilities(image_data)

    # Initialize result to null for all three types
    result: Dict[str, Any] = {
        "face": None,
        "location": None,
        "pii": None,
    }

    logger.info(f"Identified vulnerabilities: {vulnerabilities}")

    # FACE
    if "face" in vulnerabilities:
        face_boxes = detect_faces_in_image(image_data)
        logger.info(f"Detected face boxes: {face_boxes}")
        result["face"] = _format_masks(
            face_boxes, reason="visible face", image_data=image_data, mask_type="face"
        )

    # LOCATION
    if "location" in vulnerabilities:
        generated_image = detect_location_in_image(image_data)
        logger.info(f"Generated image with location hidden: {generated_image}")
        result["location"] = generated_image

    # PII
    if "PII" in vulnerabilities:
        ocr_values = extract_ocr_values(image_data)
        if ocr_values:
            logger.info(f"Extracted OCR values: {ocr_values}")
            pii_boxes = detect_pii_in_ocr(image_data, ocr_values)
            logger.info(f"Detected PII boxes: {pii_boxes}")
            result["pii"] = _format_masks(
                pii_boxes,
                reason="potential personal info",
                image_data=image_data,
                mask_type="pii",
            )
        else:
            # Identified but no OCR provided → cannot localize; keep as null
            result["pii"] = None

    return result


def build_privacy_masks_as_urls(
    image_data: bytes,
) -> Dict[str, Optional[str]]:
    """
    1) Query /classify/vulnerability to know which types are present.
    2) Run the corresponding detectors.
    3) Return S3 URLs for masked images, or null for types not detected.
    """
    vulnerabilities = get_vulnerabilities(image_data)

    # Initialize result to null for all three types
    result: Dict[str, Optional[str]] = {
        "face": None,
        "location": None,
        "pii": None,
    }

    logger.info(f"Identified vulnerabilities: {vulnerabilities}")

    # FACE
    if "face" in vulnerabilities:
        face_boxes = detect_faces_in_image(image_data)
        logger.info(f"Detected face boxes: {face_boxes}")
        if face_boxes:
            # Apply face masks and get S3 URL
            masked_image_data = apply_masks_to_image(image_data, face_boxes)
            # result["face"] = upload_to_s3(masked_image_data)
            result["face"] = ""

    # LOCATION
    if "location" in vulnerabilities:
        generated_image = detect_location_in_image(image_data)
        logger.info(f"Generated image with location hidden: {generated_image}")
        result["location"] = generated_image

    # PII
    if "PII" in vulnerabilities:
        ocr_values = extract_ocr_values(image_data)
        if ocr_values:
            logger.info(f"Extracted OCR values: {ocr_values}")
            pii_boxes = detect_pii_in_ocr(image_data, ocr_values)
            logger.info(f"Detected PII boxes: {pii_boxes}")
            if pii_boxes:
                # Apply PII masks and get S3 URL
                masked_image_data = apply_masks_to_image(image_data, [], pii_boxes)
                result["pii"] = upload_to_s3(masked_image_data)

    return result


def process_face_masks(image_data: bytes) -> Optional[Dict[str, Any]]:
    """
    Process face detection and return masked image URL or coordinates.
    """
    face_boxes = detect_faces_in_image(image_data)
    if not face_boxes:
        return None
    return _format_masks(
        face_boxes, reason="visible face", image_data=image_data, mask_type="face"
    )


def process_pii_masks(
    image_data: bytes, ocr_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Process PII detection and return masked image URL or coordinates.
    """
    pii_boxes = detect_pii_in_ocr(image_data, ocr_data)
    if not pii_boxes:
        return None
    return _format_masks(
        pii_boxes,
        reason="potential personal info",
        image_data=image_data,
        mask_type="pii",
    )
