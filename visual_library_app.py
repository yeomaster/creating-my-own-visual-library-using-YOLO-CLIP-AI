import os, argparse, time, json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import speech_recognition as sr

# -----------------------
# Paths / constants
# -----------------------
LIB_ROOT = Path("visual_library")
FACE_DIR = LIB_ROOT / "faces"
OBJ_DIR  = LIB_ROOT / "objects"
INDEX_VEC = LIB_ROOT / "library_vectors.npy"
INDEX_META = LIB_ROOT / "library_meta.json"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path):
    if not d.exists(): return []
    exts = {".jpg",".jpeg",".png"}
    return [p for p in d.iterdir() if p.suffix.lower() in exts]

# speech recognition part:
import speech_recognition as sr

def listen_for_command(timeout=5):
    """Listen for a short voice command and return it as text (lowercase)."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(" Say a command (e.g., 'add mug' or 'add alice')...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=5)
            command = r.recognize_google(audio, language="en-US").lower().strip()
            print(f"You said: {command}")
            return command
        except sr.WaitTimeoutError:
            print(" No speech detected.")
        except sr.UnknownValueError:
            print(" Could not understand audio.")
        except sr.RequestError as e:
            print(f" Speech API error: {e}")
    return None

# -----------------------
# Voice hotkeys for recognize()
# -----------------------
def start_voice_hotkeys(callback, keywords=("manual","classify","center","last","stop listening")):
    """
    Listens in background. When a keyword is heard, calls:
        callback(keyword, full_text=<recognized text>)
    Returns a stop() function to stop listening.
    """
    import speech_recognition as sr
    r = sr.Recognizer()
    mic = sr.Microphone()
    stopper = {"fn": None}

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.5)

    def _cb(recognizer, audio):
        try:
            txt = recognizer.recognize_google(audio, language="en-US").lower().strip()
            # print("Voice:", txt)  # debug
            for kw in keywords:
                if kw in txt:
                    callback(kw, full_text=txt)
                    break
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[Voice hotkeys] Speech API error: {e}")

    stopper["fn"] = r.listen_in_background(mic, _cb, phrase_time_limit=4)
    return stopper["fn"]




# -----------------------
# Face tools (InsightFace)
# -----------------------
class FaceTools:
    def __init__(self, det_size=(640,640), ctx_id=-1):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_s")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, frame_bgr):
        # returns list of (x1,y1,x2,y2, score, embedding or None)
        faces = self.app.get(frame_bgr)
        out = []
        for f in faces:
            x1,y1,x2,y2 = map(int, f.bbox)
            emb = None
            if hasattr(f, "normed_embedding"):
                emb = f.normed_embedding
            out.append((x1,y1,x2,y2, float(getattr(f, "det_score", 1.0)), emb))
        return out

    def embed_crop(self, crop_bgr):
        det = self.app.get(crop_bgr)
        if not det: return None
        # largest
        f = max(det, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return f.normed_embedding  # already L2-normalized (dim=512)

# -----------------------
# Object tools (YOLO + CLIP)
# -----------------------
class ItemDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        # class names map
        self.names = self.model.model.names if hasattr(self.model.model,"names") else {}
        self.conf = conf

    def detect(self, frame_bgr):
        # returns list of (x1,y1,x2,y2, conf, cls_name)
        res = self.model.predict(frame_bgr, verbose=False, conf=self.conf, imgsz=640)[0]
        boxes = []
        for b in res.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            cls_id = int(b.cls[0].item())
            name = self.names.get(cls_id, str(cls_id))
            conf = float(b.conf[0].item())
            boxes.append((x1,y1,x2,y2, conf, name))
        return boxes

class ClipEmbedder:
    # unified image embedder for items (generic objects)
    def __init__(self, model="clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
    def embed_pil(self, pil_img: Image.Image):
        vec = self.model.encode([pil_img], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec  # L2 or cosine-normalized already
    def embed_bgr_crop(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.embed_pil(pil)

# -----------------------
# Index build & search
# -----------------------
def build_index():
    ensure_dir(LIB_ROOT)
    face_tools = FaceTools()
    clip = ClipEmbedder()

    vecs = []
    meta = []

    # Faces ------------- (now inside build_index)
    if FACE_DIR.exists():
        for label in sorted([p.name for p in FACE_DIR.iterdir() if p.is_dir()]):
            for imgp in list_images(FACE_DIR / label):
                img = cv2.imdecode(np.fromfile(str(imgp), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                emb = face_tools.embed_crop(img)
                if emb is None:
                    # Retry: upscale (helps if the crop is small/tight)
                    h, w = img.shape[:2]
                    scale = 2 if max(h, w) < 300 else 1.5
                    up = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                    emb = face_tools.embed_crop(up)
                if emb is None:
                    # Last try: pad a bit around edges
                    up = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_REPLICATE)
                    emb = face_tools.embed_crop(up)
                if emb is None:
                    continue
                vecs.append(emb)
                meta.append({"type": "face", "label": label, "path": str(imgp)})

    # Items -------------
    if OBJ_DIR.exists():
        for label in sorted([p.name for p in OBJ_DIR.iterdir() if p.is_dir()]):
            for imgp in list_images(OBJ_DIR / label):
                img = cv2.imdecode(np.fromfile(str(imgp), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                emb = clip.embed_bgr_crop(img)
                vecs.append(emb)
                meta.append({"type": "item", "label": label, "path": str(imgp)})

    if not vecs:
        print("[Index] No images found to index.")
        return

    V = np.vstack(vecs).astype(np.float32)
    np.save(str(INDEX_VEC), V)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Index] Saved {len(meta)} vectors → {INDEX_VEC} & {INDEX_META}")



def load_index():
    if not (INDEX_VEC.exists() and INDEX_META.exists()):
        return None, None
    V = np.load(str(INDEX_VEC))
    with open(INDEX_META, "r", encoding="utf-8") as f:
        M = json.load(f)
    # ensure row-wise normalized for cosine sim
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    Vn = V / norms
    return Vn.astype(np.float32), M

def nn_search(Vn, q, topk=1):
    qn = q / (np.linalg.norm(q) + 1e-9)
    sims = Vn @ qn
    if topk == 1:
        idx = int(np.argmax(sims))
        return [(idx, float(sims[idx]))]
    idxs = np.argpartition(-sims, topk)[:topk]
    idxs = idxs[np.argsort(-sims[idxs])]
    return [(int(i), float(sims[i])) for i in idxs]

# this is for manually adding items
def select_roi(frame, title="Select Object"):
    # Draw a box with the mouse, press ENTER/SPACE to confirm, or C to cancel
    r = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)



# -----------------------
# Enroll (capture crops)
# -----------------------
def enroll(mode, label, shots, cooldown, cam, yolo_model="yolov8n.pt", item_class=None, manual=False):
    assert mode in {"face","item"}
    out_dir = (FACE_DIR if mode=="face" else OBJ_DIR) / label
    ensure_dir(out_dir)
    start_idx = len(list_images(out_dir))

    face_tools = FaceTools() if mode == "face" else None
    item_det   = None if (mode == "item" and manual) else (ItemDetector(yolo_model) if mode == "item" else None)

    # Open camera (try reliable backends on Windows)
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved, cd = 0, 0
    print(f"[Enroll {mode}] Label='{label}', saving {shots} crops to {out_dir}")
    print("Tip: vary angle/lighting/distance.")

    # If manual item mode: select ROI once and reuse it (press 'r' to reselect)
    manual_box = None
    if mode == "item" and manual:
        ok, frame0 = cap.read()
        if not ok:
            print("ERROR: No frame from camera."); cap.release(); return
        box = select_roi(frame0, "Select the new item")
        if box is None:
            print("Canceled."); cap.release(); return
        manual_box = box
        print(f"[Manual ROI] Using box: {manual_box}")

    while saved < shots:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()

        # Build candidate boxes
        boxes = []
        if mode == "face":
            det = face_tools.detect(frame)
            boxes = [(x1, y1, x2, y2, s) for (x1, y1, x2, y2, s, _) in det]
        elif manual:
            boxes = [(*manual_box, 1.0)]  # single fixed box
        else:
            det = item_det.detect(frame)
            for (x1, y1, x2, y2, conf, name) in det:
                if (item_class is None) or (name.lower() == item_class.lower()):
                    boxes.append((x1, y1, x2, y2, conf))

        if boxes:
            # largest (or the manual one)
            x1, y1, x2, y2, _ = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, f"{label}: {saved}/{shots}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if cd == 0:
                # add ~20% margin for robustness
                H, W = frame.shape[:2]
                mx = int(0.05 * (x2 - x1)); my = int(0.05 * (y2 - y1)) # reduce enroll margin from 20% to 5%, make the checking of library items a little more sure
                # mx = int(0.2 * (x2 - x1)); my = int(0.2 * (y2 - y1))
                xx1 = max(0, x1 - mx); yy1 = max(0, y1 - my)
                xx2 = min(W, x2 + mx); yy2 = min(H, y2 + my)
                crop = frame[yy1:yy2, xx1:xx2]

                if crop.size > 0:
                    path = out_dir / f"img_{start_idx + saved:03d}.jpg"
                    ok2, buf = cv2.imencode(".jpg", crop)
                    if ok2:
                        buf.tofile(str(path))
                        saved += 1
                        cd = cooldown
        else:
            cv2.putText(disp, "No target found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if cd > 0:
            cd -= 1

        cv2.imshow("Enroll", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and manual:
            # re-select ROI anytime in manual mode
            box = select_roi(disp, "Re-select the item")
            if box is not None:
                manual_box = box
                print(f"[Manual ROI] Updated box: {manual_box}")
        if key == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    print(f"[Enroll] Saved {saved} images in {out_dir}")

# -----------------------
# Fallback: grid proposals + CLIP matching
# -----------------------
def _prepare_item_index(Vn, meta):
    """Return (V_items, meta_items, idx_items) filtered to items only."""
    idx_items = [i for i, m in enumerate(meta) if m.get("type") == "item"]
    if not idx_items:
        return None, None, None
    V_items = Vn[idx_items]
    meta_items = [meta[i] for i in idx_items]
    return V_items, meta_items, np.asarray(idx_items, dtype=np.int32)

def _cosine_top1(Vn, q):
    """Top-1 cosine similarity (assumes Vn rows & q are L2-normalized)."""
    qn = q / (np.linalg.norm(q) + 1e-9)
    sims = Vn @ qn
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])

def _iou(a, b):
    """IoU for boxes (x1,y1,x2,y2)."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def _nms(boxes, scores, iou_thresh=0.4):
    """Simple NMS. boxes: [(x1,y1,x2,y2)], scores: [float]"""
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        suppress = []
        for j_idx, j in enumerate(rest):
            if _iou(boxes[i], boxes[int(j)]) >= iou_thresh:
                suppress.append(j_idx)
        if suppress:
            mask = np.ones(len(rest), dtype=bool)
            mask[suppress] = False
            rest = rest[mask]
        order = rest
    return keep

def grid_scan_classify_items(frame_bgr, clip_embedder, V_items, M_items,
                             sim_thresh=0.32, tile=160, stride=120, margin=0.2,
                             max_tiles=40, topk_per_tile=1, nms_iou=0.4):
    """
    Slide a coarse grid; for each tile, expand by `margin`, embed, match to item library.
    Returns: list of (x1,y1,x2,y2,label,sim)
    """
    H, W = frame_bgr.shape[:2]
    boxes, labels, sims = [], [], []

    # Build grid (cap number of tiles to keep it snappy)
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))
    coords = [(x, y) for y in ys for x in xs]
    if len(coords) > max_tiles:
        # crude sub-sample for speed
        step = max(1, len(coords) // max_tiles)
        coords = coords[::step]

    for (x, y) in coords:
        x1, y1, x2, y2 = x, y, min(W, x + tile), min(H, y + tile)
        # expand by margin
        mx = int(margin * (x2 - x1)); my = int(margin * (y2 - y1))
        xx1 = max(0, x1 - mx); yy1 = max(0, y1 - my)
        xx2 = min(W, x2 + mx); yy2 = min(H, y2 + my)

        crop = frame_bgr[yy1:yy2, xx1:xx2]
        if crop.size == 0: 
            continue

        emb = clip_embedder.embed_bgr_crop(crop)
        idx, sim = _cosine_top1(V_items, emb)
        if sim >= sim_thresh:
            boxes.append((xx1, yy1, xx2, yy2))
            labels.append(M_items[idx]["label"])
            sims.append(sim)

    # NMS to remove overlapping duplicates
    if boxes:
        keep_idx = _nms(boxes, sims, iou_thresh=nms_iou)
        return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], labels[i], sims[i]) for i in keep_idx]
    return []




# -----------------------
# Live recognition
# -----------------------
def recognize(cam, yolo_model="yolov8n.pt", face_thresh=0.35, item_thresh=0.28):
    # Detectors/embedders
    face_tools = FaceTools()
    item_det   = ItemDetector(yolo_model)
    clip       = ClipEmbedder()

    # Load index (vectors + meta)
    Vn, M = load_index()
    if Vn is None:
        print("No index found. Run: python visual_library_app.py index")
        return

    # Items-only view for the grid fallback
    V_items, M_items, idx_items = _prepare_item_index(Vn, M)

    # --- Voice hotkeys state ---
    voice_trigger = {"mode": None}   # "manual" | "center" | "last"
    last_manual_box = None
    voice_should_stop = False

    def _on_voice(keyword, full_text=None):
        nonlocal voice_should_stop
        if keyword in ("manual", "classify"):
            voice_trigger["mode"] = "manual"
            print("[Voice] Manual classify requested.")
        elif keyword == "center":
            voice_trigger["mode"] = "center"
            print("[Voice] Center classify requested.")
        elif keyword == "last":
            voice_trigger["mode"] = "last"
            print("[Voice] Reuse last ROI requested.")
        elif keyword == "stop listening":
            voice_should_stop = True
            print("[Voice] Stop listening requested.")

    stop_voice = None
    try:
        stop_voice = start_voice_hotkeys(_on_voice)
        print("[Recognize] Voice hotkeys: say 'manual', 'center', 'last', or 'stop listening'.")
    except Exception as e:
        print(f"[Voice hotkeys] Disabled ({e}).")

    # Open camera (robust backends on Windows)
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); 
        if stop_voice is not None:
            try: stop_voice(wait_for_stop=False)
            except Exception: pass
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Recognize] Press 'm' to manually classify a region, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()

        # ---- Faces (InsightFace) ----
        fdet = face_tools.detect(frame)
        for (x1,y1,x2,y2,score,emb) in fdet:
            label = "face: unknown"; color = (0,0,255)
            if emb is not None:
                (idx, sim) = nn_search(Vn, emb, 1)[0]
                meta = M[idx]
                if meta["type"] == "face" and sim >= face_thresh:
                    label = f"{meta['label']}  {sim:.2f}"
                    color = (0,255,0)
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- Items (YOLO auto boxes) ----
        idet = item_det.detect(frame)
        for (x1,y1,x2,y2,conf,cls_name) in idet:
            # add ~20% margin for more stable embeddings
            H, W = frame.shape[:2]
            mx = int(0.2 * (x2 - x1)); my = int(0.2 * (y2 - y1))
            xx1 = max(0, x1 - mx); yy1 = max(0, y1 - my)
            xx2 = min(W, x2 + mx); yy2 = min(H, y2 + my)
            crop = frame[yy1:yy2, xx1:xx2]

            label = f"{cls_name}: unknown"; color = (0,0,255)
            if crop.size > 0:
                emb = clip.embed_bgr_crop(crop)
                (idx, sim) = nn_search(Vn, emb, 1)[0]
                meta = M[idx]
                if meta["type"] == "item" and sim >= item_thresh:
                    label = f"{meta['label']}  {sim:.2f}"
                    color = (0,255,0)
                else:
                    label = f"{cls_name} ({conf:.2f})"
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # >>> Grid fallback when YOLO finds no items
        if V_items is not None and len(idet) == 0:
            grid_hits = grid_scan_classify_items(
                frame, clip, V_items, M_items,
                sim_thresh=max(0.30, item_thresh - 0.02),  # a hair under item_thresh
                tile=160, stride=120, margin=0.20,
                max_tiles=40, topk_per_tile=1, nms_iou=0.45
            )
            for (gx1, gy1, gx2, gy2, glabel, gsim) in grid_hits:
                color = (0, 255, 0)
                cv2.rectangle(disp, (gx1, gy1), (gx2, gy2), color, 2)
                cv2.putText(disp, f"{glabel}  {gsim:.2f}", (gx1, max(20, gy1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Recognize", disp)
        key = cv2.waitKey(1) & 0xFF

        # --- Keyboard 'm' manual classify (existing behavior) ---
        if key == ord('m'):
            box = select_roi(disp, "Select region to classify")
            if box is not None:
                x1,y1,x2,y2 = box
                last_manual_box = box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    det = face_tools.detect(crop)
                    emb = det[0][5] if (det and det[0][5] is not None) else clip.embed_bgr_crop(crop)
                    (idx, sim) = nn_search(Vn, emb, 1)[0]
                    meta = M[idx]
                    color = (0,255,0)
                    label = f"{meta['label']}  {sim:.2f}"
                    cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(disp, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imshow("Recognize", disp)
                    cv2.waitKey(500)

        # --- Voice-driven manual classify actions ---
        # Stop listening if requested
        if voice_should_stop and stop_voice is not None:
            try:
                stop_voice(wait_for_stop=False)
            except Exception:
                pass
            stop_voice = None
            voice_should_stop = False
            print("[Voice] Listener stopped.")

        # Handle voice triggers
        if voice_trigger["mode"] == "manual":
            box = select_roi(disp, "Voice: select region to classify")
            if box is not None:
                x1,y1,x2,y2 = box
                last_manual_box = box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    det = face_tools.detect(crop)
                    emb = det[0][5] if (det and det[0][5] is not None) else clip.embed_bgr_crop(crop)
                    (idx, sim) = nn_search(Vn, emb, 1)[0]
                    meta = M[idx]
                    color = (0,255,0)
                    label = f"{meta['label']}  {sim:.2f}"
                    cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(disp, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imshow("Recognize", disp)
                    cv2.waitKey(300)
            voice_trigger["mode"] = None

        elif voice_trigger["mode"] == "center":
            H, W = frame.shape[:2]
            s = min(H, W) // 3  # center box ~1/3 of min dimension
            x1 = (W - s) // 2; y1 = (H - s) // 2; x2 = x1 + s; y2 = y1 + s
            last_manual_box = (x1, y1, x2, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                det = face_tools.detect(crop)
                emb = det[0][5] if (det and det[0][5] is not None) else clip.embed_bgr_crop(crop)
                (idx, sim) = nn_search(Vn, emb, 1)[0]
                meta = M[idx]
                color = (0,255,0)
                label = f"{meta['label']}  {sim:.2f}"
                cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                cv2.putText(disp, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Recognize", disp)
                cv2.waitKey(300)
            voice_trigger["mode"] = None

        elif voice_trigger["mode"] == "last" and last_manual_box is not None:
            x1, y1, x2, y2 = last_manual_box
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                det = face_tools.detect(crop)
                emb = det[0][5] if (det and det[0][5] is not None) else clip.embed_bgr_crop(crop)
                (idx, sim) = nn_search(Vn, emb, 1)[0]
                meta = M[idx]
                color = (0,255,0)
                label = f"{meta['label']}  {sim:.2f}"
                cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                cv2.putText(disp, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Recognize", disp)
                cv2.waitKey(300)
            voice_trigger["mode"] = None

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if stop_voice is not None:
        try:
            stop_voice(wait_for_stop=False)
        except Exception:
            pass




# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Visual Library: enroll, index, recognize.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # enroll
    ep = sub.add_parser("enroll", help="Capture crops into the library")
    ep.add_argument("--mode", choices=["face","item"], required=True)
    ep.add_argument("--label", required=True)
    ep.add_argument("--shots", type=int, default=20)
    ep.add_argument("--cooldown", type=int, default=6)
    ep.add_argument("--cam", type=int, default=0)
    ep.add_argument("--yolo", default="yolov8n.pt")
    ep.add_argument("--item-class", default=None, help="Only enroll this YOLO class (e.g., 'bottle')")
    ep.add_argument("--manual", action="store_true", help="Manual ROI selection for items (no YOLO)")


    # index
    sub.add_parser("index", help="Build embedding index from library")

    # recognize
    rp = sub.add_parser("recognize", help="Live recognition with bounding boxes")
    rp.add_argument("--cam", type=int, default=0)
    rp.add_argument("--yolo", default="yolov8n.pt")
    rp.add_argument("--face-thresh", type=float, default=0.35)
    rp.add_argument("--item-thresh", type=float, default=0.28)

    # voice
    sub.add_parser("voice", help="Voice-triggered enrollment mode")

    args = ap.parse_args()
    ensure_dir(FACE_DIR); ensure_dir(OBJ_DIR)

    if args.cmd == "enroll":
        enroll(args.mode, args.label, args.shots, args.cooldown, args.cam, args.yolo, args.item_class, manual=args.manual)
    elif args.cmd == "index":
        build_index()
    elif args.cmd == "recognize":
        recognize(args.cam, args.yolo, args.face_thresh, args.item_thresh)
    elif args.cmd == "voice":
        print("[Voice mode] Say 'add mug' or 'add alice'. Say 'stop' to quit.")
        while True:
            cmd = listen_for_command()
            if not cmd:
                continue

            if cmd.startswith("add "):
                label = cmd.split("add ", 1)[1].strip()
                if not label:
                    print("No label detected.")
                    continue

                # Optional: guess if it's a face or item
                if any(k in label for k in ["person", "face", "me", "myself", "alice", "bob"]):
                    mode = "face"
                else:
                    mode = "item"

                print(f" Detected command: add '{label}' (mode={mode})")
                enroll(mode, label, shots=20, cooldown=6, cam=0)
                print(" Done enrolling. You can say another command.")

            elif any(k in cmd for k in ["stop", "exit", "quit"]):
                print(" Exiting voice mode.")
                break

    

if __name__ == "__main__":
    main()


# notes about this code/ how it works:
# when detecting faces the program uses InsightFace library's built in face detector, as this is more accurate that YOLO
# for other items, it uses YOLOv8
# for each time it detects something, it captures the LARGEST bounding box the program detects, takes a picture of that item, and then stores it in a library
# we use the following code to detect/remember/take picture of faces:
# python visual_library_app.py enroll --mode face --label (name of person)
# we use the following code to detect/remember/take picture of items:
# python visual_library_app.py enroll --mode item --label (name of item)
# takes 20 pictures of item/face
# after we take picture of all the items we want we can now INDEX our images
# indexing is when the code/script goes through every saved image in visual_library/ and computes a numerical signature (embedding) for each
# these numerical signature are then changed into vectors, which is then sent to Open AI CLIP
# once sent to clip it is then compiled into a matrix and sent back. this matrix acts as a searchable vector database for our program, essentially acting as our virtual libary
# indexing code:
# python visual_library_app.py index
# now we use both Face detector (InsightFace) and object detector (YOLO) both run per frame
# but instead, we use the library we just created via index as the things the computer creates bounding boxses around
# to do this we finally run this code:
# python visual_library_app.py recognize

# this is for adding a item manually into our library:
# python visual_library_app.py enroll --mode item --label (item YOLO cant recognise) --manual  
# once we enter the camera drag and drop a box in the screen where you want the items pictures to be taken and then press enter
# press c to cancel at any time
# the program will take 20 pictures and store them in the library
# then we rebuild the library and run the program again

# code run simplification:
# python visual_library_app.py enroll --mode face --label (name of person)
# python visual_library_app.py enroll --mode item --label (name of item)
# python visual_library_app.py enroll --mode item --label (item YOLO cant recognise) --manual   
# python visual_library_app.py index
# python visual_library_app.py recognize --face-thresh 0.35 --item-thresh 0.7        

# activating via voice
# python visual_library_app.py voice
