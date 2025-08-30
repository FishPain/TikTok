# privacy_gallery_fixed.py (updated for "blur minors only")
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import ssl
from typing import List, Tuple
import os

# ---- Age model (OpenCV DNN) ----
AGE_PROTOTXT = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)", "(38-43)", "(48-53)"]

age_net = None
if os.path.exists(AGE_PROTOTXT) and os.path.exists(AGE_MODEL):
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)
    except Exception:
        age_net = None

# --- Hacks / compatibility ---
# Hack for SSL errors (hackathon-safe)
ssl._create_default_https_context = ssl._create_unverified_context

# --- Global models (loaded once) ---
# Haar face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# EasyOCR (text)
reader = easyocr.Reader(["en"], download_enabled=True)

# YOLO general (tiny)
# NOTE: this is the standard COCO model. It does NOT include a "face" class, but includes "person".
# We'll use it as a fallback to approximate a face region from the upper part of each detected person box
# when a dedicated YOLO face model isn't available.
yolo_general = None
try:
    yolo_general = YOLO("/model/yolov8n.pt")
except Exception:
    yolo_general = None

# YOLO face-specific (preferred if present). Users can drop "yolov8n-face.pt" besides the script.
yolo_face = None
for face_ckpt in ["/model/yolov8n-face.pt"]:
    try:
        yolo_face = YOLO(face_ckpt)
        break
    except Exception:
        continue

# --- Utils ---
def _xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

def _clip_box(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
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

def _merge_boxes(boxes: List[Tuple[int,int,int,int]], iou_thresh: float = 0.3) -> List[Tuple[int,int,int,int]]:
    """Simple NMS-like merge by area (keep largest)"""
    if not boxes:
        return []
    # sort by area desc
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept

# --- Core: Face detection (Haar + YOLO) ---
def detect_faces(image_bgr: np.ndarray, yolo_conf: float = 0.25) -> List[Tuple[int,int,int,int]]:
    """
    Detect faces using BOTH Haar cascade and YOLO.
    - If a YOLO face model is available (e.g., yolov8n-face.pt), use it.
    - Otherwise, fall back to approximating face regions from the 'person' detections of the COCO model.
    - Always union with Haar cascade detections for robustness.
    Returns list of (x, y, w, h) in pixel coords.
    """
    H, W = image_bgr.shape[:2]
    boxes: List[Tuple[int,int,int,int]] = []

    # 1) Haar cascade
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in haar:
        x, y, w, h = _clip_box(x, y, w, h, W, H)
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    # 2) YOLO
    try:
        if yolo_face is not None:
            # True face model
            results = yolo_face.predict(image_bgr, conf=yolo_conf, verbose=False)
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
            results = yolo_general.predict(image_bgr, conf=yolo_conf, verbose=False)
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
    except Exception:
        # YOLO errors shouldn't break the app; just ignore
        pass

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
    Expected age uses bucket midpoints: [1,5,10,17.5,22.5,28.5,40.5,50.5] (rough)
    """
    if age_net is None or face_bgr is None or face_bgr.size == 0:
        return None, None

    blob = cv2.dnn.blobFromImage(cv2.resize(face_bgr, (227, 227)), 1.0,
                                 (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward().flatten()  # shape (8,)
    probs = _age_softmax(preds)

    # Midpoints for rough expectation
    mids = np.array([1, 5, 10, 17.5, 22.5, 28.5, 40.5, 50.5], dtype=float)
    expected_age = float((probs * mids).sum())
    return probs, expected_age

def prob_minor(probs: np.ndarray):
    """Sum of buckets that mostly lie under 18 (0-2, 4-6, 8-12, 15-20 with partial credit)."""
    if probs is None: 
        return None
    # Count full for first three buckets, and partial for 15-20 since it straddles 18
    p_0_2, p_4_6, p_8_12, p_15_20 = probs[0], probs[1], probs[2], probs[3]
    # Weight 15-20 bucket ~70% as 'minor' (tunable)
    return float(p_0_2 + p_4_6 + p_8_12 + 0.7 * p_15_20)

def is_minor(face_bgr: np.ndarray,
             p_minor_threshold: float = 0.40,
             hard_age_cut: float = 18.0,
             min_face_size_px: int = 40):
    """
    Returns True if the face is classified as under 18.
    - If the model is unavailable or uncertain -> return False (do NOT blur).
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
def detect_text(image_rgb: np.ndarray):
    # easyocr returns list of [box, text, confidence]; convert boxes to x,y,w,h
    results = reader.readtext(image_rgb)
    boxes = []
    H, W = image_rgb.shape[:2]
    for r in results:
        pts = r[0]  # 4 points
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w > 0 and h > 0:
            boxes.append((x1, y1, w, h))
    return _merge_boxes(boxes, iou_thresh=0.2)

def detect_license_plates(image_bgr: np.ndarray):
    # Lightweight heuristic: OCR for short high-contrast strings (kept simple for demo)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb)
    boxes = []
    for box, text, conf in results:
        t = "".join(ch for ch in text if ch.isalnum()).upper()
        # crude plate-like filter
        if 5 <= len(t) <= 10 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
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
        # Fallback for other OpenCV builds
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

def blur_regions(image_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = image_bgr.copy()
    for (x, y, w, h) in boxes:
        if w <= 0 or h <= 0:
            continue
        roi = out[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        # Stronger blur: scale kernel size with region size
        k = max(51, (w // 3) | 1)   # was (w // 7), now much larger
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        out[y:y+h, x:x+w] = blurred
    return out

# --- UI ---
def main():
    st.set_page_config(page_title="Privacy Gallery", layout="wide")
    st.title("Privacy Gallery (Haar + YOLO Faces) — Blur Minors Only")

    with st.sidebar:
        st.markdown("### Detection Settings")
        use_yolo = st.checkbox("Use YOLO for faces (if available)", value=True)
        yolo_conf = st.slider("YOLO confidence", 0.05, 0.9, 0.4, 0.05)
        st.markdown("---")
        run_text = st.checkbox("Detect text", value=False)
        run_plate = st.checkbox("Detect license-like text", value=False)
        run_qr = st.checkbox("Detect QR codes", value=False)

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Faces
        faces = detect_faces(image_cv, yolo_conf=yolo_conf) if use_yolo else _merge_boxes(
            [tuple(map(int, b)) for b in face_cascade.detectMultiScale(
                cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(40,40)
            )]
        )

        # Additional signals
        text_boxes = detect_text(np.array(image)) if run_text else []
        plates = detect_license_plates(image_cv) if run_plate else []
        qrs = detect_qr_codes(image_cv) if run_qr else []

        # Blur ONLY minors' faces
        def selective_blur_faces_minors_only(image_bgr, face_boxes):
            out = image_bgr.copy()
            for (x, y, w, h) in face_boxes:
                crop = out[y:y+h, x:x+w]
                if is_minor(crop):
                    k = max(51, (w // 3) | 1)
                    out[y:y+h, x:x+w] = cv2.GaussianBlur(crop, (k, k), 0)
            return out

        blurred = selective_blur_faces_minors_only(image_cv, faces)

        # Optional: blur other regions if toggled
        for regs in (text_boxes, plates, qrs):
            blurred = blur_regions(blurred, regs)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), caption="Privacy-Protected", use_container_width=True)

        with st.expander("Debug: boxes"):
            st.write({"faces": faces, "text": text_boxes, "plates": plates, "qrs": qrs})

if __name__ == "__main__":
    main()
