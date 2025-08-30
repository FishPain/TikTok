import requests
import json
import os
from typing import Dict, Any, List, Tuple, Optional, Set, Union
import logging

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
            result = response.json()
            detections = result.get("detections", [])
            face_boxes = []
            for det in detections:
                class_name = det.get("class", "").lower()
                if "face" in class_name:
                    bbox = det.get("bbox", {})
                    face_boxes.append(
                        (
                            float(bbox.get("x1", 0)),
                            float(bbox.get("y1", 0)),
                            float(bbox.get("x2", 0)),
                            float(bbox.get("y2", 0)),
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
            text_to_boxes.setdefault(t, []).append(tuple(box[:4]))

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
    pii_items: List[Dict[str, Any]], ocr_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Map detected PII items back to OCR bboxes and convert into mask format.
    Returns something like:
    {
      "mask": [
        {"coordinate": "(x1, x2, y1, y2)", "reason": "..."},
        ...
      ]
    }
    or None if nothing matched.
    """
    boxes = pii_to_bbox_list(pii_items, ocr_result)
    reason = "potential personal info"
    return _format_masks(boxes, reason=reason)


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
    boxes: List[Tuple[float, float, float, float]], reason: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Convert list of (x1, y1, x2, y2) into the required structure.
    Return None if no boxes (so caller can set the key to null).
    Required string order is "(x1, x2, y1, y2)" per your spec.
    """
    boxes = _unique_boxes(boxes)
    if not boxes:
        return None
    return {
        "mask": [
            {"coordinate": f"({x1}, {x2}, {y1}, {y2})", "reason": reason}
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
        result["face"] = _format_masks(face_boxes, reason="visible face")

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
            result["pii"] = _format_masks(pii_boxes, reason="potential personal info")
        else:
            # Identified but no OCR provided → cannot localize; keep as null
            result["pii"] = None

    return result
