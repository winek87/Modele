import os
import sys
import json
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify, abort
from functools import wraps
from hailo_platform import VDevice, HEF, ConfigureParams, InferVStreams, FormatType, InputVStreamParams, OutputVStreamParams, HailoStreamInterface

# ==========================================
# KONFIGURACJA
# ==========================================
DEBUG_MODE = True  # Zapisuje pliki debug_*.jpg na dysku

API_KEY = os.environ.get("API_KEY", "tajne_haslo_123")
DETECTOR_HEF = os.environ.get("DETECTOR_HEF", "yolov8n_relu6_face_kpts--640x640_quant_hailort_hailo8_1.hef")
RECOGNIZER_HEF = os.environ.get("RECOGNIZER_HEF", "Buffalo_L.hef")

# Parametry
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
YOLO_INPUT_SIZE = (640, 640)
ARCFACE_INPUT_SIZE = (112, 112) # Standard ArcFace
NUM_LANDMARKS = 68
MAX_IMG_SIZE = 50 * 1024 * 1024 

app = Flask(__name__)
target = None
detector = None
recognizer = None

# --- IMPORTY HEIC/RAW ---
try:
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIC = True
except ImportError:
    HAS_HEIC = False
    print("[WARN] Brak pillow-heif (HEIC off)")

try:
    import rawpy
    HAS_RAW = True
except ImportError:
    HAS_RAW = False
    print("[WARN] Brak rawpy (RAW off)")

# ==========================================
# 1. SMART DECODE (Zawsze zwraca BGR)
# ==========================================
def smart_decode(file_bytes, filename):
    ext = filename.lower().split('.')[-1]
    
    if HAS_HEIC and ext in ['heic', 'heif']:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except: pass

    if HAS_RAW and ext in ['arw', 'dng', 'cr2', 'nef', 'orf', 'rw2']:
        try:
            with rawpy.imread(io.BytesIO(file_bytes)) as raw:
                return cv2.cvtColor(raw.postprocess(), cv2.COLOR_RGB2BGR)
        except: pass

    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ==========================================
# 2. YOLO LOGIKA
# ==========================================
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def dfl_decode(box_tensor):
    n, c = box_tensor.shape
    box_tensor = box_tensor.reshape(n, 4, 16)
    probs = softmax(box_tensor, axis=-1)
    bins = np.arange(16, dtype=np.float32)
    return np.sum(probs * bins, axis=-1)

def nms(boxes, scores, kpts, iou_thresh):
    if len(boxes) == 0: return [], [], []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return boxes[keep], scores[keep], kpts[keep]

def resize_with_pad(image, target_size=YOLO_INPUT_SIZE):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(image, (nw, nh))
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    pad_w = (target_size[1] - nw) // 2
    pad_h = (target_size[0] - nh) // 2
    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = img_resized
    return canvas, scale, pad_w, pad_h

def decode_outputs(results, input_shape=YOLO_INPUT_SIZE):
    strides = [8, 16, 32]
    tensors = {8: {}, 16: {}, 32: {}}
    for _, t in results.items():
        if len(t.shape) == 4: t = t[0]
        stride = input_shape[0] // t.shape[0]
        if stride not in tensors: continue
        if t.shape[-1] == 1: tensors[stride]['score'] = t
        elif t.shape[-1] == 64: tensors[stride]['box'] = t
        elif t.shape[-1] > 64: tensors[stride]['kpts'] = t

    all_boxes, all_scores, all_kpts = [], [], []
    for stride in strides:
        data = tensors[stride]
        if 'score' not in data: continue
        scores = 1 / (1 + np.exp(-data['score'].reshape(-1)))
        mask = scores > CONF_THRESHOLD
        if not np.any(mask): continue
        
        anchors = np.stack(np.meshgrid(np.arange(data['score'].shape[1]), np.arange(data['score'].shape[0])), axis=-1).astype(np.float32) + 0.5
        anchors = anchors.reshape(-1, 2)[mask]
        dist = dfl_decode(data['box'].reshape(-1, 64)[mask])
        
        x1 = (anchors[:,0] - dist[:,0]) * stride
        y1 = (anchors[:,1] - dist[:,1]) * stride
        x2 = (anchors[:,0] + dist[:,2]) * stride
        y2 = (anchors[:,1] + dist[:,3]) * stride
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
        all_scores.append(scores[mask])
        
        if 'kpts' in data:
            raw_kpts = data['kpts'].reshape(-1, data['kpts'].shape[-1])[mask]
            kpts = raw_kpts.reshape(-1, NUM_LANDMARKS, raw_kpts.shape[1]//NUM_LANDMARKS)
            kx = (kpts[..., 0] * 2 + anchors[:, 0:1]) * stride
            ky = (kpts[..., 1] * 2 + anchors[:, 1:2]) * stride
            all_kpts.append(np.stack([kx, ky], axis=-1))

    if not all_boxes: return [], [], []
    return np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_kpts)

# ==========================================
# 3. ALIGNMENT (Punkty dla 112x112)
# ==========================================
def align_face(img, landmarks):
    ref_pts = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    src_pts = np.array([[p['x'], p['y']] for p in landmarks], dtype=np.float32)
    
    if len(src_pts) != 5: return cv2.resize(img, ARCFACE_INPUT_SIZE)
    tform = cv2.estimateAffinePartial2D(src_pts, ref_pts)[0]
    return cv2.warpAffine(img, tform, ARCFACE_INPUT_SIZE) if tform is not None else cv2.resize(img, ARCFACE_INPUT_SIZE)

# ==========================================
# 4. KLASY HAILO
# ==========================================
class HailoYoloDetector:
    def __init__(self, device, hef_path):
        print(f"[INIT] Detektor: {hef_path}")
        self.hef = HEF(hef_path)
        self.group = device.configure(self.hef, ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe))[0]
        self.params = self.group.create_params()
        self.in_params = InputVStreamParams.make(self.group, format_type=FormatType.UINT8)
        self.out_params = OutputVStreamParams.make(self.group, format_type=FormatType.FLOAT32)
        self.input_name = self.hef.get_input_vstream_infos()[0].name

    def infer(self, image):
        input_img, scale, pw, ph = resize_with_pad(image, YOLO_INPUT_SIZE)
        # BGR -> RGB dla modelu
        input_data = np.expand_dims(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), axis=0)
        
        with self.group.activate(self.params):
            with InferVStreams(self.group, self.in_params, self.out_params) as pipe:
                res = pipe.infer({self.input_name: input_data})
        
        boxes, scores, kpts = decode_outputs(res, YOLO_INPUT_SIZE)
        final = []
        if len(boxes) > 0:
            boxes, scores, kpts = nms(boxes, scores, kpts, IOU_THRESHOLD)
            for box, score, kp in zip(boxes, scores, kpts):
                real_box = [int((box[0]-pw)/scale), int((box[1]-ph)/scale), int((box[2]-pw)/scale), int((box[3]-ph)/scale)]
                real_kpts = [[(p[0]-pw)/scale, (p[1]-ph)/scale] for p in kp]
                
                lm5 = []
                if len(real_kpts) >= 68:
                    for idx in [36, 45, 30, 48, 54]: lm5.append({"x": float(real_kpts[idx][0]), "y": float(real_kpts[idx][1])})
                else:
                    for p in real_kpts[:5]: lm5.append({"x": float(p[0]), "y": float(p[1])})
                
                final.append({"box": real_box, "confidence": float(score), "landmarks": lm5})
        return final

class HailoRecognizer:
    def __init__(self, device, hef_path):
        print(f"[INIT] Recognizer: {hef_path}")
        self.hef = HEF(hef_path)
        self.group = device.configure(self.hef, ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe))[0]
        self.params = self.group.create_params()
        self.in_params = InputVStreamParams.make(self.group, format_type=FormatType.UINT8)
        self.out_params = OutputVStreamParams.make(self.group, format_type=FormatType.FLOAT32)
        self.input_name = self.hef.get_input_vstream_infos()[0].name

    def infer(self, face_img):
        # 1. Resize do 112x112
        face_input = cv2.resize(face_img, ARCFACE_INPUT_SIZE)
        
        # 2. BGR -> RGB (Kluczowe dla poprawnych wektorów)
        face_rgb = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        
        # 3. DEBUG: Zapisz to co wchodzi do sieci (jako BGR żeby człowiek widział naturalne kolory)
        if DEBUG_MODE:
            cv2.imwrite("debug_aligned.jpg", cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

        # 4. Inferencja
        with self.group.activate(self.params):
            with InferVStreams(self.group, self.in_params, self.out_params) as pipe:
                res = pipe.infer({self.input_name: np.expand_dims(face_rgb, axis=0)})
        
        # 5. Normalizacja
        raw = list(res.values())[0].flatten()
        norm = np.linalg.norm(raw)
        return (raw / norm).tolist() if norm > 0 else raw.tolist()

# ==========================================
# 5. ENDPOINTY
# ==========================================
def require_appkey(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('x-api-key') == API_KEY: return f(*args, **kwargs)
        return abort(401)
    return decorated

try:
    print("----------------------------------------------------------------")
    print(f"[SYSTEM] START HAILO SERVER (Final Fix v6)")
    target = VDevice()
    detector = HailoYoloDetector(target, DETECTOR_HEF)
    recognizer = HailoRecognizer(target, RECOGNIZER_HEF)
except Exception as e:
    print(f"[CRITICAL] {e}")
    sys.exit(1)

@app.route("/detect", methods=["POST"])
@require_appkey
def detect_route():
    if 'file' not in request.files: abort(400)
    f = request.files['file']
    img = smart_decode(f.read(), f.filename)
    if img is None: abort(415)

    if DEBUG_MODE: cv2.imwrite("debug_input_raw.jpg", img)

    faces = detector.infer(img)
    final = []
    for face in faces:
        try:
            aligned = align_face(img, face['landmarks'])
            desc = recognizer.infer(aligned)
            final.append({
                "detection_confidence": face['confidence'],
                "left": face['box'][0], "top": face['box'][1],
                "right": face['box'][2], "bottom": face['box'][3],
                "landmarks": face['landmarks'],
                "descriptor": desc
            })
        except: continue
    
    print(f"[LOG] {f.filename}: Znaleziono {len(final)} twarzy.")
    return jsonify({"filename": f.filename, "faces-count": len(final), "faces": final})

@app.route("/compute", methods=["POST"])
@require_appkey
def compute_route():
    if 'file' not in request.files: abort(400)
    f = request.files['file']
    face_json = json.loads(request.form.get("face"))
    img = smart_decode(f.read(), f.filename)
    if img is None: abort(415)
    
    t, b, l, r = int(face_json['top']), int(face_json['bottom']), int(face_json['left']), int(face_json['right'])
    h, w = img.shape[:2]
    
    # Dodajemy margines dla alignera
    m = 20
    ct, cb = max(0, t-m), min(h, b+m)
    cl, cr = max(0, l-m), min(w, r+m)
    crop = img[ct:cb, cl:cr]
    
    if crop.size == 0: abort(400)

    # Próba realign na cropie
    sub_faces = detector.infer(crop)
    if sub_faces:
        best = max(sub_faces, key=lambda x: x['confidence'])
        aligned = align_face(crop, best['landmarks'])
        desc = recognizer.infer(aligned)
    else:
        # Fallback
        desc = recognizer.infer(cv2.resize(crop, ARCFACE_INPUT_SIZE))

    w_box, h_box = r-l, b-t
    face_json['landmarks'] = [
        {"x": float(l+w_box*0.3), "y": float(t+h_box*0.35)}, {"x": float(l+w_box*0.7), "y": float(t+h_box*0.35)},
        {"x": float(l+w_box*0.5), "y": float(t+h_box*0.5)},
        {"x": float(l+w_box*0.3), "y": float(t+h_box*0.7)}, {"x": float(l+w_box*0.7), "y": float(t+h_box*0.7)}
    ]
    face_json['descriptor'] = desc
    return jsonify({"filename": f.filename, "face": face_json})

@app.route("/open", methods=["GET"])
@require_appkey
def open_route(): return jsonify({"preferred_mimetype": "image/jpeg", "maximum_area": MAX_IMG_SIZE})

@app.route("/health", methods=["GET"])
def health(): return "ok"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
