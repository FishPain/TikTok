from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
from typing import List, Tuple
import os
import base64
import logging
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


# ---- Age model (OpenCV DNN) ----
AGE_PROTOTXT = "/app/models/age_deploy.prototxt"
AGE_MODEL = "/app/models/age_net.caffemodel"
AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(21-24)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


age_net = None
if os.path.exists(AGE_PROTOTXT) and os.path.exists(AGE_MODEL):
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)
        logger.info("Age detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load age model: {e}")
        age_net = None
else:
    logger.warning("Age model files not found")

# --- Hacks / compatibility ---
# ssl._create_default_https_context = ssl._create_unverified_context

# --- Global models (loaded once) ---
# Haar face
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# EasyOCR (text)
try:
    reader = easyocr.Reader(["en"], download_enabled=True)
    logger.info("EasyOCR model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load EasyOCR: {e}")
    reader = None

# YOLO general (tiny)
yolo_general = None
try:
    yolo_general = YOLO("/app/models/yolov8n.pt")
    logger.info("YOLO general model loaded successfully")
except Exception as e:
    logger.warning(f"YOLO general model not available: {e}")

# YOLO face-specific (preferred if present)
yolo_face = None
for face_ckpt in ["/app/models/yolov8n-face.pt", "yolov8n-face.pt"]:
    try:
        yolo_face = YOLO(face_ckpt)
        logger.info(f"YOLO face model loaded from {face_ckpt}")
        break
    except Exception:
        continue


# --- Utils ---
def _xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def _clip_box(x, y, w, h, W, H):
    x = max(0, x)
    y = max(0, y)
    if w < 0 or h < 0:
        return 0, 0, 0, 0
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    return int(x), int(y), int(w), int(h)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a2x, b2x), min(a2y, b2y)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_boxes(
    boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """Simple NMS-like merge by area (keep largest)"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


# --- Core: Face detection (Haar + YOLO) ---
def detect_faces(
    image_bgr: np.ndarray, yolo_conf: float = 0.25
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces using BOTH Haar cascade and YOLO.
    Returns list of (x, y, w, h) in pixel coords.
    """
    H, W = image_bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    # 1) Haar cascade
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    haar = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for x, y, w, h in haar:
        x, y, w, h = _clip_box(x, y, w, h, W, H)
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    # 2) YOLO
    try:
        if yolo_face is not None:
            # True face model
            results = yolo_face.predict(
                image_bgr, conf=yolo_conf, verbose=False, show=False
            )
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    x, y, w, h = _xyxy_to_xywh(xyxy)
                    x, y, w, h = _clip_box(x, y, w, h, W, H)
                    if w > 0 and h > 0:
                        boxes.append((x, y, w, h))
        elif yolo_general is not None:
            # Approximate face as the upper portion of person detections
            results = yolo_general.predict(
                image_bgr, conf=yolo_conf, verbose=False, show=False
            )
            for r in results:
                names = r.names if hasattr(r, "names") else {}
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name == "person":
                        xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                        px, py, pw, ph = _xyxy_to_xywh(xyxy)
                        # Heuristic: head region ~ top 35% of the person bbox, narrower width
                        head_h = int(ph * 0.35)
                        head_w = int(pw * 0.5)
                        cx = px + pw // 2
                        hx = int(cx - head_w // 2)
                        hy = py
                        hx, hy, head_w, head_h = _clip_box(hx, hy, head_w, head_h, W, H)
                        if head_w > 0 and head_h > 0:
                            boxes.append((hx, hy, head_w, head_h))
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")

    # Merge overlaps from both methods
    boxes = _merge_boxes(boxes, iou_thresh=0.3)
    return boxes


# --- Age helpers ---
def _age_softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.clip(ex.sum(), 1e-8, None)


def estimate_age_probs(face_bgr: np.ndarray):
    """
    Returns: (bucket_probs: np.ndarray[8], expected_age: float)
    """
    if age_net is None or face_bgr is None or face_bgr.size == 0:
        return None, None

    blob = cv2.dnn.blobFromImage(
        cv2.resize(face_bgr, (227, 227)),
        1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
    )
    age_net.setInput(blob)
    preds = age_net.forward().flatten()
    probs = _age_softmax(preds)

    # Midpoints for rough expectation
    mids = np.array([1, 5, 10, 17.5, 22.5, 28.5, 40.5, 50.5], dtype=float)
    expected_age = float((probs * mids).sum())
    return probs, expected_age


def prob_minor(probs: np.ndarray):
    """Sum of buckets that mostly lie under 18"""
    if probs is None:
        return None
    p_0_2, p_4_6, p_8_12, p_15_20 = probs[0], probs[1], probs[2], probs[3]
    return float(p_0_2 + p_4_6 + p_8_12 + 0.7 * p_15_20)


def is_minor(
    face_bgr: np.ndarray,
    p_minor_threshold: float = 0.40,
    hard_age_cut: float = 18.0,
    min_face_size_px: int = 40,
):
    """
    Returns True if the face is classified as under 18.
    """
    if face_bgr is None:
        return False
    if min(face_bgr.shape[0], face_bgr.shape[1]) < min_face_size_px:
        return False

    probs, expected_age = estimate_age_probs(face_bgr)
    if probs is None or expected_age is None:
        return False

    p_m = prob_minor(probs)
    if (p_m is not None and p_m >= p_minor_threshold) or (expected_age < hard_age_cut):
        return True
    return False


# --- Other detectors ---
def detect_text(image_rgb: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
    """
    Detect text regions with EasyOCR and return list of (x1, y1, w, h, text).
    """
    if reader is None:
        return []

    results = reader.readtext(image_rgb)
    boxes: List[Tuple[int, int, int, int, str]] = []
    H, W = image_rgb.shape[:2]

    for r in results:
        pts, text = r[0], r[1]
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(W, max(xs)), min(H, max(ys))

        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w > 0 and h > 0:
            boxes.append((x1, y1, w, h, text))

    # If you still want merging, pass only coords to _merge_boxes
    merged = _merge_boxes([b[:4] for b in boxes], iou_thresh=0.2)

    # Attach text back to merged boxes (simple heuristic: keep original texts of boxes that overlap)
    merged_with_text: List[Tuple[int, int, int, int, str]] = []
    for mx1, my1, mw, mh in merged:
        # find first matching text from original boxes
        match = next(
            (
                t
                for (x1, y1, w, h, t) in boxes
                if abs(x1 - mx1) < 5 and abs(y1 - my1) < 5
            ),
            "",
        )
        merged_with_text.append((mx1, my1, mw, mh, match))

    return merged_with_text


def detect_license_plates(image_bgr: np.ndarray):
    if reader is None:
        return []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb)
    boxes = []
    for box, text, conf in results:
        t = "".join(ch for ch in text if ch.isalnum()).upper()
        if (
            5 <= len(t) <= 10
            and any(c.isdigit() for c in t)
            and any(c.isalpha() for c in t)
        ):
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return _merge_boxes(boxes, iou_thresh=0.2)


def detect_qr_codes(image_bgr):
    detector = cv2.QRCodeDetector()
    boxes = []
    points = None
    try:
        res = detector.detectAndDecodeMulti(image_bgr)
        if isinstance(res, tuple):
            if len(res) == 3:
                decoded_info, points, _ = res
            elif len(res) == 4:
                _, decoded_info, points, _ = res
            else:
                points = None
        else:
            points = None
    except Exception:
        try:
            ok, pts = detector.detectMulti(image_bgr)
            points = pts if ok else None
        except Exception:
            points = None

    if points is not None and len(points) > 0:
        for quad in points:
            xs = quad[:, 0].astype(int).tolist()
            ys = quad[:, 1].astype(int).tolist()
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def blur_regions(
    image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    out = image_bgr.copy()
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        roi = out[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        k = max(51, (w // 3) | 1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        out[y : y + h, x : x + w] = blurred
    return out


def selective_blur_faces_minors_only(image_bgr, face_boxes):
    """Blur ONLY minors' faces"""
    out = image_bgr.copy()
    minor_faces = []
    for x, y, w, h in face_boxes:
        crop = out[y : y + h, x : x + w]
        if is_minor(crop):
            k = max(51, (w // 3) | 1)
            out[y : y + h, x : x + w] = cv2.GaussianBlur(crop, (k, k), 0)
            minor_faces.append((x, y, w, h))
    return out, minor_faces


def image_to_base64(image_bgr):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode(".jpg", image_bgr)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64


def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# --- REST Endpoints ---


@app.route("/health")
@require_api_key
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "models": {
            "age_net": age_net is not None,
            "face_cascade": True,
            "easyocr": reader is not None,
            "yolo_general": yolo_general is not None,
            "yolo_face": yolo_face is not None,
        },
    }
    return jsonify(status)


@app.route("/detect/age", methods=["POST"])
@require_api_key
def detect_age_endpoint():
    """Detect age for faces in image"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .jpg, .jpeg, .png files are allowed"}), 400
    yolo_conf = float(request.form.get("yolo_conf", 0.4))

    try:
        # Load image
        image = Image.open(file.stream).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = detect_faces(image_bgr, yolo_conf=yolo_conf)

        # Analyze each face for age
        face_analysis = []
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = image_bgr[y : y + h, x : x + w]
            probs, expected_age = estimate_age_probs(face_crop)
            p_minor = prob_minor(probs) if probs is not None else None
            is_minor_result = is_minor(face_crop)

            face_info = {
                "face_id": i,
                "bbox": [x, y, w, h],
                "expected_age": expected_age,
                "is_minor": is_minor_result,
                "minor_probability": p_minor,
                "age_buckets": AGE_BUCKETS,
                "age_probabilities": probs.tolist() if probs is not None else None,
            }
            face_analysis.append(face_info)

        return jsonify(
            {
                "faces": face_analysis,
                "total_faces": len(faces),
                "minors_detected": sum(1 for f in face_analysis if f["is_minor"]),
            }
        )

    except Exception as e:
        logger.error(f"Age detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect", methods=["POST"])
@require_api_key
def detect_objects():
    """Detect only location-related objects (ignore faces)"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .jpg, .jpeg, .png files are allowed"}), 400

    try:
        yolo_conf = float(request.form.get("conf", 0.1))

        # Load image
        image = Image.open(file.stream).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        H, W = image_bgr.shape[:2]

        detections = []

        if yolo_general is not None:
            results = yolo_general.predict(
                image_bgr, conf=yolo_conf, verbose=False, show=False
            )
            logger.info(f"YOLO general detection results: {results}")
            for r in results:
                names = getattr(r, "names", {}) or {}
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue

                for b in r.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    confidence = float(b.conf[0].item()) if b.conf is not None else 0.0
                    if confidence < yolo_conf:
                        continue

                    # Filter to only location-related classes
                    location_classes = {
                        "stop sign",
                        "traffic light",
                        "street sign",
                        "building",
                        "car",
                        "truck",
                        "bus",
                        "bicycle",
                        "motorcycle",
                    }
                    if cls_name.lower() not in location_classes:
                        continue

                    xyxy = (
                        b.xyxy[0].cpu().numpy().astype(float).tolist()
                        if hasattr(b, "xyxy")
                        else [0, 0, 0, 0]
                    )
                    x1, y1, x2, y2 = xyxy

                    detections.append(
                        {
                            "class": cls_name,
                            "confidence": confidence,
                            "bbox": {
                                "x1": max(0.0, float(x1)),
                                "y1": max(0.0, float(y1)),
                                "x2": min(float(x2), float(W)),
                                "y2": min(float(y2), float(H)),
                            },
                        }
                    )

        return jsonify(
            {
                "detections": detections,
                "total_detections": len(detections),
                "image_dimensions": {"width": W, "height": H},
            }
        )

    except Exception as e:
        logger.exception("Object detection error")
        return jsonify({"error": str(e)}), 500


@app.route("/process/text", methods=["POST"])
@require_api_key
def get_text_endpoint():
    """Detect text regions"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .jpg, .jpeg, .png files are allowed"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        image_rgb = np.array(image)
        text_boxes = detect_text(image_rgb)

        return jsonify(
            {
                "type": "text",
                "total": len(text_boxes),
                "bbox": [list(b[:4]) for b in text_boxes],
                "text": [b[4] for b in text_boxes],
            }
        )

    except Exception as e:
        logger.error(f"Text detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/process/blur-plates", methods=["POST"])
@require_api_key
def blur_plates_endpoint():
    """Detect license plates"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .jpg, .jpeg, .png files are allowed"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        plates = detect_license_plates(image_bgr)

        return jsonify(
            {
                "type": "license_plate",
                "total": len(plates),
                "boxes": [list(b) for b in plates],
            }
        )

    except Exception as e:
        logger.error(f"License plate detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/process/blur-qr", methods=["POST"])
@require_api_key
def blur_qr_endpoint():
    """Detect QR codes"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .jpg, .jpeg, .png files are allowed"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        qrs = detect_qr_codes(image_bgr)

        return jsonify(
            {"type": "qr_code", "total": len(qrs), "boxes": [list(b) for b in qrs]}
        )

    except Exception as e:
        logger.error(f"QR detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
@require_api_key
def index():
    """API documentation"""
    return jsonify(
        {
            "service": "Privacy Gallery API",
            "version": "1.1",
            "description": "REST API for privacy-related detection tasks",
            "endpoints": {
                "GET /health": "Service health check",
                "POST /detect/age": "Detect age for faces in image",
                "POST /process/blur-text": "Detect text regions",
                "POST /process/blur-plates": "Detect license plates",
                "POST /process/blur-qr": "Detect QR codes",
            },
            "notes": "All endpoints return only metadata (no images). Bounding boxes are [x,y,w,h].",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, debug=True)
