"""Model wrappers: all detectors share one context. YOLO26 with YOLOv8 fallback.

PPE comes from YOLOE (Ultralytics' open-vocabulary model) driven by text
prompts — fully pretrained, no custom training. Class prompts are mapped to
the canonical helmet/vest/gloves/boots names the pipeline expects. If YOLOE
or its text encoder is unavailable (e.g. offline), fall back to the legacy
custom-trained best-4.pt.
"""
import contextlib
import os
from pathlib import Path

from ultralytics import YOLO, YOLOWorld

ROOT = Path(__file__).resolve().parent.parent


def _p(rel):
    """Model path anchored to the repo root, so the pipeline runs from any cwd."""
    q = ROOT / rel
    return str(q) if q.exists() else rel

# Person tracking: YOLO26s + BoT-SORT with the pretrained yolo26n-reid encoder.
# Chosen by benchmark_tracking.py on the construction video: detects ~8.7 of ~9
# workers per frame (yolo26n saw only ~5), 25 raw IDs -> 10 merged workers,
# stable through occlusion, ~13.5 tracking fps on MPS.
TRACKER = str(Path(__file__).resolve().parent.parent / "config" / "botsort_reid_construction.yaml")

PPE_PROMPTS = ["hard hat", "high-visibility vest", "gloves", "boots"]
PPE_NAME_MAP = {"hard hat": "helmet", "high-visibility vest": "vest",
                "gloves": "gloves", "boots": "boots"}

# Per-class confidence floors. The open-vocabulary PPE model is calibrated
# very differently per prompt: measured over 84 worker crops from the sample
# footage, "hard hat" scores a median 0.75 while "high-visibility vest"
# scores a median 0.065. A single flat threshold therefore either floods
# helmets with noise or accepts every vest. These are applied AFTER the model
# runs at a low floor, so each class gets its own operating point.
PPE_CONF = {"helmet": 0.30, "vest": 0.10, "gloves": 0.30, "boots": 0.30}
PPE_MODEL_FLOOR = 0.05   # model-level conf; per-class floors filter above it

# open-vocabulary machinery list — broad on purpose: sites have trucks, vans
# and pickups, not just excavators. Pipeline-side size + temporal
# confirmation filters keep false hits out at the lower conf threshold.
EQUIP_PROMPTS = ["excavator", "bulldozer", "dump truck", "wheel loader",
                 "forklift", "crane", "truck", "pickup truck", "van", "car",
                 "bus", "tractor"]


@contextlib.contextmanager
def _in_root():
    """Run inside the repo root.

    The open-vocabulary text encoders (YOLOE's mobileclip_blt.ts, YOLO-World's
    CLIP weights) are resolved by ultralytics against the *current working
    directory*, not against the model path. Without this, running the pipeline
    from anywhere other than the repo root silently re-downloads 572 MB of
    encoder into the caller's cwd on every run.
    """
    prev = os.getcwd()
    try:
        os.chdir(ROOT)
        yield
    finally:
        with contextlib.suppress(OSError):
            os.chdir(prev)


def _load(name, fallback=None):
    try:
        return YOLO(name)
    except Exception:
        return YOLO(fallback) if fallback else None


def _load_ppe():
    """Pretrained open-vocabulary PPE detector (YOLOE, text-prompt classes).
    Returns (model, name_map); legacy best-4.pt fallback needs no mapping."""
    try:
        from ultralytics import YOLOE
        m = YOLOE(_p("yoloe-11s-seg.pt"))
        with _in_root():          # text encoder is resolved against cwd
            m.set_classes(PPE_PROMPTS, m.get_text_pe(PPE_PROMPTS))
        return m, PPE_NAME_MAP
    except Exception:
        return YOLO(_p("models/best-4.pt")), None


class Detectors:
    def __init__(self, device=None):
        self.person = _load(_p("models/yolo26s.pt"), _p("models/yolo26n.pt"))
        self.ppe, self._ppe_map = _load_ppe()
        self.pose = _load(_p("models/yolo26n-pose.pt"), _p("models/yolov8n-pose.pt"))
        self.equip = YOLOWorld(_p("models/yolov8x-worldv2.pt"))
        try:
            with _in_root():      # CLIP text encoder is resolved against cwd
                self.equip.set_classes(EQUIP_PROMPTS + ["person"])
        except Exception as e:
            # set_classes needs the CLIP text encoder (network on first use);
            # offline/proxied environments fall back to the checkpoint's
            # default COCO classes — trucks/cars/buses still match
            print(f"warning: equipment prompts not set ({e}); "
                  f"using default classes")
        if device:
            for m in (self.person, self.ppe, self.pose, self.equip):
                m.to(device)

    def _ppe_name(self, i):
        n = self.ppe.names[int(i)]
        return self._ppe_map.get(n, n) if self._ppe_map else n

    def detect_persons(self, frame, conf=0.3):
        """Return [(track_id, x1, y1, x2, y2, conf)] using BoT-SORT + ReID.
        Must be called on EVERY frame for stable IDs."""
        r = self.person.track(frame, persist=True, tracker=TRACKER,
                              classes=[0], conf=conf, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                tid = int(b.id.item()) if b.id is not None else -1
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((tid, x1, y1, x2, y2, float(b.conf[0])))
        return out

    def detect_ppe(self, frame, conf=0.25):
        """Return {class_name: [(x1,y1,x2,y2,conf)]} from the PPE model."""
        r = self.ppe.predict(frame, conf=conf, verbose=False)[0]
        out = {}
        if r.boxes is not None:
            for b in r.boxes:
                name = self._ppe_name(b.cls[0])
                out.setdefault(name, []).append((*b.xyxy[0].tolist(), float(b.conf[0])))
        return out

    def detect_ppe_per_person(self, frame, persons, conf=None, pad=0.18,
                              imgsz=384, class_conf=None):
        """Run the PPE model on padded crops of each tracked person.

        Tight crops make small items (helmet, gloves) far easier for the
        model than a full construction frame, and remove the fragile
        IoU/containment matching step. imgsz=384 is both faster and more
        accurate than the default 640 letterbox for these small crops.
        Returns {track_id: {class: conf}}, keeping the best detection per
        PPE class per person.
        """
        H, W = frame.shape[:2]
        crops, tids = [], []
        for tid, x1, y1, x2, y2, _ in persons:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # pad generously upward: the helmet sits above the head box
            px1 = int(max(0, cx - bw * (0.5 + pad)))
            py1 = int(max(0, cy - bh * (0.5 + 2.0 * pad)))
            px2 = int(min(W, cx + bw * (0.5 + pad)))
            py2 = int(min(H, cy + bh * (0.5 + pad)))
            if px2 - px1 < 8 or py2 - py1 < 8:
                continue
            crops.append(frame[py1:py2, px1:px2])
            tids.append(tid)
        if not crops:
            return {}

        # run the model at a low floor, then apply the per-class operating
        # points: one flat threshold cannot serve prompts whose score
        # distributions differ by an order of magnitude (see PPE_CONF).
        floor = PPE_MODEL_FLOOR if conf is None else conf
        cls_conf = PPE_CONF if class_conf is None else class_conf
        out = {}
        for tid, r in zip(tids, self.ppe.predict(crops, conf=floor, imgsz=imgsz,
                                                 verbose=False)):
            matched = {}
            if r.boxes is not None:
                for b in r.boxes:
                    name = self._ppe_name(b.cls[0])
                    c = float(b.conf[0])
                    if c < cls_conf.get(name, floor):
                        continue
                    if c > matched.get(name, 0.0):
                        matched[name] = c
            if matched:
                out[tid] = matched
        return out

    def detect_equipment(self, frame, conf=0.15):
        """Return [(label, x1,y1,x2,y2,conf)] for construction machinery."""
        r = self.equip.predict(frame, conf=conf, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                label = self.equip.names[int(b.cls[0])]
                if label != "person":
                    out.append((label, *b.xyxy[0].tolist(), float(b.conf[0])))
        return out

    def detect_poses(self, frame, conf=0.3):
        """Return [(x1,y1,x2,y2,conf, kpts[17,3])] for posed persons."""
        r = self.pose.predict(frame, conf=conf, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b, k in zip(r.boxes, r.keypoints.data):
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((x1, y1, x2, y2, float(b.conf[0]), k.cpu().numpy()))
        return out
