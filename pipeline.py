"""Construction AI demo pipeline.

video/webcam -> pretrained models -> tracking -> rules -> incidents + alert clips
                -> annotated video (H.264) + events.json
"""
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

PIPELINE_VERSION = "v6"   # shown in the dashboard; bump on logic changes

import cv2
import numpy as np

from vision import safety_rules as sr
from vision.activity import (ActivityTracker, CONFIRM_S, IDLE_AFTER_S,
                             WALK_SPEED, WORK_SPEED)
from vision.alerting import severity_map
from vision.detectors import Detectors
from vision.gate import GATE_MIN_FRAMES, GateChecker
from vision.registry import WorkerRegistry
from vision.shifts import normalise_shifts, shift_summary

# Dark palette for boxes: every box is drawn as a dark core with a white
# halo (draw_box) and a dark label chip with white text (draw_label), so it
# stays clearly visible on any background. Values are BGR.
COLORS = {
    "person": (138, 58, 30), "helmet": (0, 100, 0), "vest": (0, 90, 180),
    "gloves": (0, 110, 110), "boots": (60, 60, 80), "human": (138, 58, 30),
    "excavator": (0, 0, 139), "dump truck": (0, 0, 139),
    "wheel loader": (0, 0, 139),
}
ALERT_COLORS = {
    "no_helmet": (0, 0, 139), "no_vest": (9, 83, 180),
    "possible_fall": (64, 20, 173), "danger_zone": (128, 0, 128),
    "proximity": (64, 0, 128),
}
# bright variants kept for the on-video alert banner (readable on its black chip)
ALERT_TEXT = {
    "no_helmet": (0, 0, 255), "no_vest": (0, 100, 255),
    "possible_fall": (0, 0, 255), "danger_zone": (255, 0, 255),
    "proximity": (255, 0, 128),
}
ALERT_LABELS = {
    "no_helmet": "NO HELMET", "no_vest": "NO VEST",
    "possible_fall": "POSSIBLE FALL", "danger_zone": "DANGER ZONE",
    "proximity": "UNSAFE PROXIMITY",
}
ZONE_C = {"Danger Zone": (0, 165, 255),       # in-video polygon (BGR orange)
          "Zone 1": (255, 0, 255), "Zone 2": (0, 165, 255)}   # camera zones


def _zone_token(name):
    """Short filename token for a zone name ('Danger Zone' -> '_dz')."""
    if not name:
        return ""
    if name == "Danger Zone":
        return "_dz"
    tail = "".join(c for c in name.split()[-1] if c.isalnum())
    return f"_z{tail.lower()}" if tail else ""

GRACE = 60              # frames before an unseen incident is closed (auto-scaled to fps)
CLIP_LEAD = 40          # frames of context saved before the incident starts
CLIP_TAIL = 20          # frames saved after the incident ends
CLIP_MAX_FRAMES = 180   # hard cap per clip
MAX_CLIPS = 30          # total clips per run
MIN_INCIDENT_FRAMES = 3

# PPE temporal smoothing. The previous version latched an item ON after a
# single detection and only dropped it after 20 consecutive misses; because
# PPE runs every 2nd processed frame, at stride 5 that held an item "present"
# for ~8 s of video after one lucky hit, which (together with the old hi-vis
# fallback) reported 99.9% helmet/vest compliance and never raised a PPE
# alert. Smoothing is now symmetric and expressed in seconds, so behaviour
# does not change with --stride.
PPE_OFF_S = 1.5         # sustained absence (video seconds) before "missing"
PPE_ON_HITS = 1         # detections needed to (re)confirm an item as present
PPE_EVERY = 2           # PPE runs on every Nth processed frame
EQUIP_EVERY = 5         # run the heavy equipment model every Nth frame
EQUIP_MIN_SIDE = 48     # px; machinery boxes smaller than this are noise
# Machinery confirmation. The conf floor had to drop to 0.10 because a large,
# unmistakable truck in the sample footage peaks at only 0.19 -- at the old 0.2
# default no machinery was ever detected and proximity alerts could not fire.
# A lower floor lets sporadic open-vocab hallucinations through (a phantom
# "excavator" on the concrete-pour clip), so confirmation is now 3-of-4
# samples: persistent real machinery survives, one-off hallucinations do not.
EQUIP_CONFIRM_N = 3     # hits needed within the recent window to confirm
EQUIP_WINDOW = 4        # inference samples in that window (x EQUIP_EVERY)
# Hi-vis colour assist. Deliberately conservative: on 29 hand-labelled crops
# the old 0.10 threshold (with a generic blue band) fired on 10/10 workers
# wearing no vest. At 0.45 on the fluorescent-only score it only fires on
# unmistakable full-torso hi-vis, so it can rescue a vest the model misses
# without silently clearing real violations.
HIVIS_MIN_FRAC = 0.45   # torso hi-vis pixel fraction counting as a vest

REID_PATH = Path(__file__).resolve().parent / "yolo26n-reid.onnx"
REID_SAMPLE_EVERY = 4   # extract track embeddings on every Nth fresh frame
REID_MAX_PER_TRACK = 16 # appearance samples averaged per track
REID_MIN_SIM = 0.35     # advisory-only: strong appearance match preference
REID_CONFIRM_SIM = 0.55 # cosine similarity that confirms a merge candidate
MERGE_GAP_S = 10.0      # max time gap for chaining fragmented tracks

# PPE grace auto-cap for file sources: never wait longer than 25% of the
# footage (2 s floor), so short demo clips still produce alerts.
PPE_GRACE_FRAC = 0.25
PPE_GRACE_MIN_S = 2.0
REGISTRY_FILENAME = "worker_registry.json"


def cap_ppe_grace(requested_s, duration_s, frac=PPE_GRACE_FRAC,
                  min_s=PPE_GRACE_MIN_S):
    """Auto-scale the PPE grace period to the clip length (file sources).

    A 2-min default grace on a 45 s clip can never fire; cap it at 25% of
    the footage duration (2 s floor). 'Immediate' (0 s) stays 0."""
    if requested_s <= 0 or duration_s is None or duration_s <= 0:
        return requested_s
    return min(float(requested_s), max(min_s, frac * duration_s))


def _build_reid():
    """Optional ReID encoder for appearance-verified track merging."""
    try:
        from ultralytics.trackers.utils.reid import ReID
        if REID_PATH.exists():
            return ReID(str(REID_PATH), device="cpu")
    except Exception:
        pass
    return None


def run(source, out_path, events_path, zone=None, zones=None, conf_person=0.3,
        conf_ppe=None, conf_equip=0.10, danger_radius_m=3.0, show_pose=True,
        max_side=960, stride=1, max_frames=None, on_frame=None, detectors=None,
        ppe_grace_s=120.0, danger_grace_s=1.0, ppe_zone_only=False,
        low_light=False, gate_mode=False, severity=None, shifts=None,
        use_registry=True, registry_path=None):
    """source: video path or 0 for webcam.

    ppe_grace_s: a missing helmet/vest only raises an alert after the worker
    has been continuously without it for this many seconds of video time
    (auto-capped to 25% of the clip length on file sources).
    danger_grace_s: a worker only raises a danger-zone alert after being
    continuously inside the zone for this many seconds of video time.
    zones: [{"name": "Danger Zone", "poly": [...]}] (points in original
    video coordinates). A "Danger Zone" poly of None falls back to the
    default bottom-centre polygon.
    ppe_zone_only: helmet/vest alerts only apply to workers inside a zone.
    low_light: CLAHE-enhance the frame for model inference (annotations
    still render on the original frame).
    gate_mode: check PPE once per worker on first stable sighting instead
    of alerting continuously (results in top-level gate_checks).
    severity: per-alert-type severity overrides (see vision.alerting).
    shifts: shift-window config for per-shift summaries (vision.shifts).
    use_registry/registry_path: persistent W-xx worker identities matched
    by ReID embeddings across runs and cameras.
    """
    d = detectors or Detectors()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {source!r} (webcam permission?)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = max_frames or 10 ** 9

    scale = min(1.0, max_side / max(W, H))
    W_p, H_p = int(W * scale) & ~1, int(H * scale) & ~1

    # resolve the danger-zone polygon (scaled to the processed frame size)
    if not zones:
        zones = [{"name": "Danger Zone", "poly": zone}]
    _norm = []
    for i, z in enumerate(zones[:2]):
        name, poly = (z.get("name") or f"Zone {i + 1}"), z.get("poly")
        if poly is None and name == "Danger Zone":
            poly = sr.default_zone(W, H)      # classic default zone
        if not poly:
            continue
        _norm.append({"name": name,
                      "poly": [(int(x * scale), int(y * scale)) for x, y in poly]})
    zones = _norm or [{"name": "Danger Zone",
                       "poly": [(int(x * scale), int(y * scale))
                                for x, y in sr.default_zone(W, H)]}]

    clips_dir = Path(out_path).parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_fps = fps / stride          # clips contain only processed frames
    grace = max(GRACE, int(fps)) // stride   # keep the same wall-clock close time
    # auto-cap the PPE grace on file sources so short clips still alert
    is_cam = isinstance(source, int) or \
        (isinstance(source, str) and source.strip().isdigit())
    ppe_grace_eff = cap_ppe_grace(ppe_grace_s,
                                  None if is_cam else total / max(fps, 1e-6))
    ppe_grace_f = max(0, int(ppe_grace_eff * fps))
    # symmetric PPE hysteresis, expressed in seconds so --stride does not
    # change how long an item stays latched (see PPE_OFF_S)
    _ppe_interval_s = PPE_EVERY * max(stride, 1) / max(fps, 1e-6)
    ppe_off_n = max(2, round(PPE_OFF_S / max(_ppe_interval_s, 1e-6)))
    ppe_since = {}          # tid -> {item: first frame seen missing}
    danger_grace_f = max(0, int(danger_grace_s * fps))
    zone_since = {}         # (tid, zone name) -> (first frame in, last frame in)
    clahe = cv2.createCLAHE(2.0, (8, 8)) if low_light else None
    gate = GateChecker(min_frames=GATE_MIN_FRAMES) if gate_mode else None
    gate_checks = []        # one entry per worker checked at the "gate"

    def clip_out_name(inc):
        ztok = _zone_token(inc.get("zone"))
        return f"{inc['type']}_w{inc['person']}_f{inc['start_f']}{ztok}.mp4"

    def clip_json_path(abs_or_rel):
        """Clip path as stored in events.json: relative to the run dir."""
        if not abs_or_rel:
            return abs_or_rel
        try:
            return str(Path(abs_or_rel).resolve()
                       .relative_to(Path(out_path).resolve().parent))
        except Exception:
            return str(abs_or_rel)

    vw = None
    events, incidents = [], []
    active = {}
    clips_written = 0
    buffer = deque(maxlen=CLIP_LEAD)
    counts = defaultdict(int)
    person_rows = []      # per-person per-frame records
    pidx = -1             # processed-frame index (maps to annotated video frames)
    workers_seen = set()
    equip_seen = defaultdict(int)
    t0 = time.time()
    fi = -1

    def open_clip(inc):
        nonlocal clips_written
        if clips_written >= MAX_CLIPS:
            return None, 0
        path = clips_dir / clip_out_name(inc)
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                            clip_fps, (W_p, H_p))
        for b, its, bfi in buffer:
            w.write(render_clip_frame(b, its, inc, zones, bfi, fps))
        clips_written += 1
        return w, len(buffer)

    def finalize(key, inc):
        if inc.get("writer"):
            inc["writer"].release()
            inc["writer"] = None
        if inc["frames"] < MIN_INCIDENT_FRAMES:
            if inc.get("clip") and os.path.exists(inc["clip"]):
                os.remove(inc["clip"])
            return
        if inc.get("clip") and os.path.exists(inc["clip"]):
            if inc["nclip"] < 8:
                os.remove(inc["clip"])
                inc["clip"] = None
            else:
                try:
                    reencode_h264(inc["clip"])
                except Exception:
                    pass
        incidents.append({
            "type": inc["type"], "person": inc["person"],
            "zone": inc.get("zone"),
            "equipment": inc.get("equipment"), "dist_m": inc.get("dist_m"),
            "start_frame": inc["start_f"], "end_frame": inc["last_seen"],
            "start_s": round(inc["start_f"] / fps, 1),
            "end_s": round(inc["last_seen"] / fps, 1),
            "duration_s": round((inc["last_seen"] - inc["start_f"]) / fps, 1),
            "clip": clip_json_path(inc.get("clip")),
        })

    # per-track caches: PPE match, pose state, keypoints (persist between heavy-model frames)
    ppe_cache, pose_cache, pose_kpts, pose_f = {}, {}, {}, {}
    ppe_streak = {}         # tid -> {item: consecutive fresh-frame misses}
    h_samples = deque(maxlen=400)   # recent tracked person heights (px) for scale
    mpp = None              # metres per pixel, calibrated from median height
    last_equipment = []
    equip_hist = {}       # label -> deque of recent hit/miss inference flags
    equip_boxes = {}      # label -> last raw boxes (shown while confirmed)
    fall_det = sr.FallDetector()
    acts = ActivityTracker(fps, stride)
    reid = _build_reid()
    emb_sums, emb_cnt = {}, {}
    fresh_i = -1             # index of fresh (heavy-model) frames
    live_map = {}            # raw id -> merged id for the live display

    def refresh_live_map():
        """Re-run track merging on the rows accumulated so far, so the live
        view shows stable merged worker IDs instead of raw tracker churn."""
        nonlocal live_map
        embs = {}
        for t, s in emb_sums.items():
            e = s / max(emb_cnt[t], 1)
            n = np.linalg.norm(e)
            if n > 1e-6:
                embs[t] = e / n
        live_map = merge_fragmented_tracks(
            person_rows, embeddings=embs,
            gap_frames=max(30, int(MERGE_GAP_S * fps) // stride))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (W_p, H_p), interpolation=cv2.INTER_AREA)
        fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if max_frames is not None and fi >= max_frames:
            break
        # stride skips whole frames (tracking included): 3x faster in Fast mode,
        # BoT-SORT + ReID keeps IDs stable at the reduced effective rate
        if fi % stride != 0:
            continue

        if vw is None:
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / stride, (frame.shape[1], frame.shape[0]))

        # low-light mode: models see a CLAHE-enhanced copy, annotations and
        # clips still render on the original frame
        inf_frame = frame
        if clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            inf_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        persons = d.detect_persons(inf_frame, conf_person)
        persons = [p for p in persons if (p[4] - p[2]) >= 10]  # drop pixel-specks only
        pidx += 1
        workers_seen.update(p[0] for p in persons if p[0] >= 0)
        # live scale calibration: median tracked person height ~ 1.7 m tall
        for p in persons:
            _h = p[4] - p[2]
            if p[0] >= 0 and 40 <= _h:
                h_samples.append(_h)
        if len(h_samples) >= 30:
            mpp = 1.7 / float(np.median(h_samples))
        fresh_i += 1

        # equipment is heavy and machinery moves slowly: every Nth frame.
        # Open-vocab detections flicker around the conf threshold, so a
        # label is confirmed by EQUIP_CONFIRM_N hits within the last
        # EQUIP_WINDOW samples (not strictly consecutive) and then shown
        # continuously from its last known box until unseen for the whole
        # window. Size floor + window keep one-off hallucinations out.
        if fresh_i % EQUIP_EVERY == 0:
            raw_eq = d.detect_equipment(inf_frame, conf_equip)
            raw_eq = [e for e in raw_eq
                      if (e[3] - e[1]) >= EQUIP_MIN_SIDE
                      and (e[4] - e[2]) >= EQUIP_MIN_SIDE]
            by_label = {}
            for e in raw_eq:
                by_label.setdefault(e[0], []).append(e)
            for lb, boxes in by_label.items():
                equip_hist.setdefault(
                    lb, deque(maxlen=EQUIP_WINDOW)).append(1)
                equip_boxes[lb] = boxes
            for lb in list(equip_hist):
                if lb not in by_label:
                    equip_hist[lb].append(0)
                    if sum(equip_hist[lb]) == 0:   # absent the whole window
                        equip_hist.pop(lb, None)
                        equip_boxes.pop(lb, None)
            confirmed = {lb for lb, h in equip_hist.items()
                         if sum(h) >= EQUIP_CONFIRM_N}
            last_equipment = [e for lb in sorted(confirmed)
                              for e in equip_boxes.get(lb, [])]
            for e in last_equipment:
                equip_seen[e[0]] += 1
        # PPE per person on padded crops, temporally smoothed: a worker
        # only counts as missing an item after PPE_MISS_LIMIT misses.
        # A hi-vis torso color check backs up the model for vests (small or
        # distant workers the model struggles with).
        if fresh_i % PPE_EVERY == 0:
            raw_ppe = d.detect_ppe_per_person(inf_frame, persons, conf_ppe)
            for p in persons:
                tid = p[0]
                if tid < 0:
                    continue
                streak = ppe_streak.setdefault(tid, {})
                vest_backup = (sr.hivis_score(inf_frame, p[1:5])
                               >= HIVIS_MIN_FRAC)
                for item in ("helmet", "vest", "gloves", "boots"):
                    detected = item in raw_ppe.get(tid, {}) or \
                        (item == "vest" and vest_backup)
                    st = streak.setdefault(item, {"hits": 0, "miss": 0,
                                                  "on": False})
                    if detected:
                        st["miss"] = 0
                        st["hits"] += 1
                        if st["hits"] >= PPE_ON_HITS:
                            st["on"] = True
                    else:
                        st["hits"] = 0
                        st["miss"] += 1
                        if st["miss"] >= ppe_off_n:
                            st["on"] = False
                ppe_cache[tid] = {it for it, st in streak.items() if st["on"]}
        # appearance samples for ReID-verified track merging
        if reid is not None and fresh_i % REID_SAMPLE_EVERY == 0:
            dets = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                             for tid, x1, y1, x2, y2, _ in persons if tid >= 0],
                            dtype=np.float32).reshape(-1, 4)
            if len(dets):
                try:
                    feats = reid(inf_frame, dets)
                except Exception:
                    feats = []
                tids = [p[0] for p in persons if p[0] >= 0]
                for tid, f in zip(tids, feats):
                    if f is None or emb_cnt.get(tid, 0) >= REID_MAX_PER_TRACK:
                        continue
                    emb_sums[tid] = emb_sums.get(tid, 0) + f.astype(np.float32)
                    emb_cnt[tid] = emb_cnt.get(tid, 0) + 1
        poses = d.detect_poses(inf_frame, conf_person) if show_pose else []
        for p in persons:
            best, bi = 0.0, None
            for q in poses:
                ov = sr.iou(p[1:5], q[:4])
                if ov > best:
                    best, bi = ov, q
            if bi is not None and best > 0.3:
                raw = sr.pose_state(bi[5], p[4] - p[2])
                pose_cache[p[0]] = fall_det.update(
                    p[0], raw, bi[5], p[4] - p[2], pidx)
                pose_kpts[p[0]] = bi[5]
                pose_f[p[0]] = fi
        equipment = last_equipment

        frame_alerts = []
        tag_boxes = []       # placed label chips this frame (collision-free)
        items = []           # per-frame draw records for clip re-rendering
        annot = frame.copy()

        for z in zones:
            zi = zone_i(z["poly"])
            zc = ZONE_C.get(z["name"], (255, 0, 255))
            overlay = annot.copy()
            cv2.fillPoly(overlay, [zi], zc)
            cv2.addWeighted(overlay, 0.18, annot, 0.82, 0, annot)
            cv2.polylines(annot, [zi], True, zc, 2)
            cv2.putText(annot, z["name"].upper(), (zi[0][0], zi[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, zc, 2)

        eq_centers = [(lb, sr.ground_center((ex1, ey1, ex2, ey2)))
                      for lb, ex1, ey1, ex2, ey2, _ in equipment]

        person_recs = []
        for tid, x1, y1, x2, y2, conf in persons:
            color = COLORS["person"]
            matched = ppe_cache.get(tid, {})
            in_zones = [z["name"] for z in zones
                        if sr.box_in_zone((x1, y1, x2, y2), z["poly"])]
            in_zone = bool(in_zones)

            # gate mode: PPE checked ONCE per worker, on first stable
            # sighting; no continuous helmet/vest alerts afterwards
            if gate is not None and tid >= 0:
                gchk = gate.check(tid, round(fi / fps, 2), matched)
                if gchk is not None:
                    gchk["frame"] = fi
                    gate_checks.append(gchk)
                    events.append({"frame": fi, "time_s": round(fi / fps, 2),
                                   "type": "gate_check", "person": tid,
                                   "compliant": gchk["compliant"],
                                   "missing": gchk["missing"]})

            # sustained-violation gate: alert only after `ppe_grace_s` without
            # PPE — and, in zone-only mode, only while inside a zone (time
            # spent outside a zone does not count toward the grace period)
            if tid >= 0:
                since = ppe_since.setdefault(tid, {})
                if gate is not None or (ppe_zone_only and not in_zone):
                    since.clear()
                    missing = []
                else:
                    for item in ("helmet", "vest"):
                        if item in matched:
                            since.pop(item, None)
                        else:
                            since.setdefault(item, fi)
                    missing = [item for item in ("helmet", "vest")
                               if item in since and fi - since[item] >= ppe_grace_f]
            elif gate is None:
                missing = [i for i in ("helmet", "vest") if i not in matched]
            else:
                missing = []
            state = pose_cache.get(tid, "unknown")
            kpts = pose_kpts.get(tid)
            # activity: only pass keypoints matched on THIS frame (else stale)
            activity = acts.update(tid, (x1, y1, x2, y2),
                                   kpts if pose_f.get(tid) == fi else None,
                                   state, fi)
            if show_pose and kpts is not None:
                for kx, ky, kc in kpts:
                    if kc > 0.3:
                        cv2.circle(annot, (int(kx), int(ky)), 3, (255, 255, 255), -1)

            # nearest equipment distance for this worker
            g = sr.ground_center((x1, y1, x2, y2))
            near = min(((lb, round(sr.pixel_to_m(math.hypot(g[0] - eg[0], g[1] - eg[1]),
                                            m_per_px=mpp), 1))
                        for lb, eg in eq_centers), key=lambda t: t[1], default=None)

            person_recs.append({
                "pidx": pidx, "frame": fi, "time_s": round(fi / fps, 2),
                "person": tid, "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "pose": state, "activity": activity,
                "helmet": "helmet" in matched, "vest": "vest" in matched,
                "gloves": "gloves" in matched, "boots": "boots" in matched,
                "in_zone": in_zone, "zones": in_zones,
                "nearest_eq": near[0] if near else None,
                "nearest_m": near[1] if near else None,
            })

            # danger-zone confirmation gate, per zone: alert only after the
            # worker has been continuously inside THAT zone for danger_grace_s
            zone_hits = []
            if tid >= 0:
                for zn in in_zones:
                    ent = zone_since.get((tid, zn))
                    if ent is None or fi - ent[1] > fps:   # gap > 1s: restart
                        ent = (fi, fi)
                    zone_since[(tid, zn)] = (ent[0], fi)
                    if fi - ent[0] >= danger_grace_f:
                        zone_hits.append(zn)
                for z in zones:
                    if z["name"] not in in_zones:
                        zone_since.pop((tid, z["name"]), None)
            else:
                zone_hits = in_zones
            for zn in zone_hits:
                frame_alerts.append({"type": "danger_zone", "person": tid,
                                     "zone": zn})
                color = ZONE_C.get(zn, ALERT_COLORS["danger_zone"])
            if "helmet" in missing:
                frame_alerts.append({"type": "no_helmet", "person": tid})
                color = ALERT_COLORS["no_helmet"]
            if "vest" in missing:
                frame_alerts.append({"type": "no_vest", "person": tid})
                color = ALERT_COLORS["no_vest"]
            if state == "possible_fall":
                frame_alerts.append({"type": "possible_fall", "person": tid})
                color = ALERT_COLORS["possible_fall"]

            draw_box(annot, (int(x1), int(y1)), (int(x2), int(y2)), color)
            # clean label: worker id + position (activity) + short PPE marks
            tag = f"P{tid} · {activity}"
            if matched:
                tag += " · " + " ".join(i[0].upper() for i in sorted(matched))
            else:
                tag += " · no PPE"
            draw_label(annot, tag, int(x1),
                       int(y1) - 27 if y1 > 29 else int(y1) + 4,
                       accent=color, taken=tag_boxes)
            items.append({"person": tid, "x1": x1, "y1": y1, "x2": x2,
                          "y2": y2, "color": color, "tag": tag,
                          "kpts": kpts if show_pose else None})

        for label, ex1, ey1, ex2, ey2, conf in equipment:
            c = COLORS.get(label, (0, 0, 139))
            draw_box(annot, (int(ex1), int(ey1)), (int(ex2), int(ey2)), c)
            draw_label(annot, f"{label} {conf:.2f}", int(ex1),
                       int(ey1) - 28 if ey1 > 30 else int(ey1) + 4,
                       accent=c, scale=0.55, taken=tag_boxes)
            items.append({"eq": label, "x1": ex1, "y1": ey1, "x2": ex2,
                          "y2": ey2, "color": c, "tag": f"{label} {conf:.2f}"})

        for p in sr.unsafe_proximity(persons, equipment, danger_radius_m,
                                     m_per_px=mpp):
            frame_alerts.append({"type": "proximity", "person": p["person"],
                                 "equipment": p["equipment"], "dist_m": p["dist_m"]})

        seen, uniq = set(), []
        for a in frame_alerts:
            k = (a["type"], a["person"], a.get("zone"), a.get("equipment"))
            if k not in seen:
                seen.add(k)
                uniq.append(a)

        if uniq:
            y = annot.shape[0] - 10
            for a in reversed(uniq[-3:]):
                if a["type"] == "proximity":
                    txt = (f"{ALERT_LABELS[a['type']]} - {a.get('equipment', '')}"
                           f" {a.get('dist_m', '')}m")
                else:
                    txt = f"{ALERT_LABELS[a['type']]} - worker {a['person']}"
                    if a.get("zone"):
                        txt += f" · {a['zone']}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annot, (6, y - th - 6), (tw + 12, y + 4), (0, 0, 0), -1)
                cv2.putText(annot, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            ALERT_TEXT[a["type"]], 2)
                y -= th + 16

        for a in uniq:
            events.append({"frame": fi, "time_s": round(fi / fps, 2), **a})
            counts[a["type"]] += 1

        for rec in person_recs:
            rec["alerts"] = sorted({a["type"] for a in uniq if a["person"] == rec["person"]})
            person_rows.append(rec)

        annot = draw_hud(annot, fi, fps, counts, len(persons), len(equipment))
        vw.write(annot)

        # ---- incident aggregation + clips ----
        # buffer raw frames + draw records so clip lead-in frames can be
        # re-rendered with only the incident's worker visible
        buffer.append((frame, items, fi))
        live_keys = set()
        for a in uniq:
            key = (a["type"], a["person"], a.get("equipment"), a.get("zone"))
            live_keys.add(key)
            if key not in active:
                inc = {"type": a["type"], "person": a["person"],
                       "zone": a.get("zone"),
                       "equipment": a.get("equipment"), "dist_m": a.get("dist_m"),
                       "start_f": fi, "last_seen": fi, "frames": 1}
                inc["writer"], inc["nclip"] = open_clip(inc)
                ztok = _zone_token(a.get("zone"))
                inc["clip"] = (str(clips_dir / f"{a['type']}_w{a['person']}_f{fi}{ztok}.mp4")
                               if inc["writer"] else None)
                active[key] = inc
            else:
                inc = active[key]
                inc["last_seen"] = fi
                inc["frames"] += 1
                if a.get("dist_m") is not None:
                    inc["dist_m"] = a["dist_m"]
        for key in list(active):
            inc = active[key]
            age = fi - inc["last_seen"]
            w = inc.get("writer")
            # clip frames: live frames while the alert is on (skip the frame
            # the incident was born on — already written via the lead-in
            # buffer), plus a short tail after it ends. Rendered per-incident
            # so only the involved worker appears.
            live_write = key in live_keys and inc["start_f"] != fi
            tail_write = key not in live_keys and age <= CLIP_TAIL
            if w and inc["nclip"] < CLIP_MAX_FRAMES and (live_write or tail_write):
                w.write(render_clip_frame(frame, items, inc, zones, fi, fps))
                inc["nclip"] += 1
            if w and inc["nclip"] >= CLIP_MAX_FRAMES:
                w.release()
                inc["writer"] = None
            if age > grace:
                finalize(key, inc)
                del active[key]

        if on_frame is not None:
            if pidx % 25 == 0:
                refresh_live_map()
            recs = [{**r, "person": live_map.get(r["person"], r["person"])}
                    for r in person_recs]
            # live per-worker incident counts (finalized + active), on the
            # same merged IDs the live view shows
            pinc = defaultdict(int)
            for i in incidents:
                pinc[live_map.get(i["person"], i["person"])] += 1
            for key in active:
                pinc[live_map.get(key[1], key[1])] += 1
            on_frame(annot, fi, len(persons), len(equipment), uniq,
                     dict(counts), len(incidents) + len(active), recs,
                     dict(pinc))

    for key in list(active):
        finalize(key, active[key])
        del active[key]

    cap.release()
    if vw:
        vw.release()
    try:
        reencode_h264(out_path)
    except Exception as e:
        print(f"warning: H.264 re-encode skipped ({e})")

    # drop ghost tracks (seen < 5 frames): spurious detections that never became people
    MIN_TRACK_FRAMES = 5
    from collections import Counter as _Counter
    track_len = _Counter(r["person"] for r in person_rows)
    keep = {t for t, n in track_len.items() if n >= MIN_TRACK_FRAMES}
    person_rows = [r for r in person_rows if r["person"] in keep]
    incidents = [i for i in incidents if i["person"] in keep]
    events = [e for e in events if "person" not in e or e["person"] in keep]
    gate_checks = [g for g in gate_checks if g["person"] in keep]

    # merge fragmented track IDs so one physical person = one ID end-to-end
    embeddings = {}
    for t, s in emb_sums.items():
        e = s / max(emb_cnt[t], 1)
        n = np.linalg.norm(e)
        if n > 1e-6:
            embeddings[t] = e / n
    merge_gap = max(30, int(MERGE_GAP_S * fps) // stride)
    mapping = merge_fragmented_tracks(person_rows, embeddings=embeddings,
                                      gap_frames=merge_gap)
    for r in person_rows:
        r["person"] = mapping[r["person"]]
    for g in gate_checks:
        g["person"] = mapping[g["person"]]
    # merged concurrent duplicates can leave two rows for one (person, frame):
    # keep the first — events/alerts are already deduped by (type, person)
    seen_pk, dedup = set(), []
    for r in person_rows:
        k = (r["person"], r["pidx"])
        if k not in seen_pk:
            seen_pk.add(k)
            dedup.append(r)
    person_rows = dedup
    for i in incidents:
        i["person"] = mapping[i["person"]]
    for e in events:
        if "person" in e:
            e["person"] = mapping[e["person"]]
    incidents = consolidate_incidents(incidents, fps)
    activity_out = activity_summary(person_rows, fps, stride)
    n_workers = len(set(mapping.values()))
    from collections import Counter as _C2
    _conc = _C2(r["pidx"] for r in person_rows)
    max_concurrent = max(_conc.values()) if _conc else 0

    # persistent worker registry: match merged workers against known W-ids
    # by ReID appearance so identities survive across runs and cameras
    worker_ids, reg_known, reg_new = {}, 0, 0
    if use_registry and embeddings:
        merged_emb = {}
        for t, e in embeddings.items():
            merged_emb.setdefault(mapping.get(t, t), []).append(e)
        final_emb = {}
        for m, vecs in merged_emb.items():
            v = np.mean(vecs, axis=0)
            n = np.linalg.norm(v)
            if n > 1e-6:
                final_emb[int(m)] = v / n
        if final_emb:
            try:
                rpath = registry_path or (Path(__file__).resolve().parent
                                          / "output" / REGISTRY_FILENAME)
                reg = WorkerRegistry(rpath)
                _known = reg.size
                worker_ids = reg.assign(final_emb)
                reg_known = sum(1 for w in worker_ids.values()
                                if int(w.split("-")[1]) <= _known)
                reg_new = len(worker_ids) - reg_known
            except Exception as e:
                print(f"warning: worker registry skipped ({e})")
    if worker_ids:
        for r in person_rows:
            r["worker_id"] = worker_ids.get(r["person"])
        for i in incidents:
            i["worker_id"] = worker_ids.get(i["person"])
        for g in gate_checks:
            g["worker_id"] = worker_ids.get(g["person"])
        for e in events:
            if "person" in e:
                e["worker_id"] = worker_ids.get(e["person"])

    data = {"video": str(source), "total_frames": fi + 1,
            "zones": [{"name": z["name"],
                       "poly": [[int(x), int(y)] for x, y in z["poly"]]}
                      for z in zones],
            "zone": [[int(x), int(y)] for x, y in zones[0]["poly"]],
            "processing_fps": round((fi + 1) / max(time.time() - t0, 1e-6), 1),
            "alert_counts": dict(counts),
            "stats": {"person_frames": len(person_rows),
                      "workers_seen": n_workers,
                      "track_ids_raw": len(keep),
                      "max_concurrent": max_concurrent,
                      "equipment_seen": dict(equip_seen),
                      "video_fps": round(fps, 1), "stride": stride,
                      "ppe_grace_s": round(ppe_grace_eff, 1),
                      "ppe_grace_requested_s": round(ppe_grace_s, 1),
                      "danger_grace_s": danger_grace_s,
                      "ppe_zone_only": bool(ppe_zone_only),
                      "low_light": bool(low_light),
                      "gate_mode": bool(gate_mode),
                      "severity": severity_map(severity),
                      "proximity_base_m": danger_radius_m,
                      "registry_known": reg_known, "registry_new": reg_new,
                      "m_per_px": round(mpp, 6) if mpp else None,
                      "started_at": started_at,
                      "date": time.strftime("%Y-%m-%d %H:%M:%S")},
            "activity": activity_out,
            "gate_checks": gate_checks,
            "incidents": incidents, "person_frames": person_rows,
            "events": events}
    data["shifts"] = shift_summary(data, shifts)
    with open(events_path, "w") as f:
        json.dump(data, f, indent=2)
    inc_counts = defaultdict(int)
    for i in incidents:
        inc_counts[i["type"]] += 1
    return dict(inc_counts)


def run_multi(zone_sources, out_dir, conf_person=0.3, conf_ppe=None,
              conf_equip=0.10, danger_radius_m=3.0, show_pose=True,
              max_side=960, stride=1, max_frames=None, on_zone_frame=None,
              ppe_grace_s=120.0, danger_grace_s=1.0, ppe_zone_only=False,
              low_light=False, gate_mode=False, severity=None, shifts=None,
              use_registry=True, registry_path=None):
    """Two camera views on one screen, processed AT THE SAME TIME.

    zone_sources: [("Zone 1", video_path_1), ("Zone 2", video_path_2)].
    Both videos run in parallel threads (fresh Detectors + tracker per
    camera, so IDs never bleed across feeds); live frames from both are
    marshalled back to the main thread, so on_zone_frame fires for EITHER
    zone as frames arrive — both feeds move together. After both finish,
    results are merged (worker IDs made globally unique, every row tagged
    with its zone) into ONE out_dir/events.json plus
    out_dir/annotated_zone<i>.mp4 per camera.

    on_zone_frame(zone_name, zone_idx, zone_total, annot, fi, n_p, n_e,
                  counts, n_inc, recs, pinc) mirrors run()'s on_frame per
    camera (always called from the calling thread).
    """
    import threading
    from queue import Queue

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = Queue(maxsize=4)          # small buffer paces producers (backpressure)

    def worker(zname, zi, src):
        try:
            det = Detectors()     # own models + tracker state per camera
            av = str(out_dir / f"annotated_zone{zi}.mp4")
            evj = out_dir / f"zone{zi}_events.json"

            def cb(annot, fi, n_p, n_e, uniq, counts, n_inc, recs,
                   pinc=None):
                for r in recs:
                    r["zone"] = zname
                q.put((zname, zi, annot, fi, n_p, n_e, dict(counts),
                       n_inc, recs, pinc))

            run(src, av, str(evj), conf_person=conf_person,
                conf_ppe=conf_ppe, conf_equip=conf_equip,
                danger_radius_m=danger_radius_m, show_pose=show_pose,
                max_side=max_side, stride=stride, max_frames=max_frames,
                on_frame=cb, detectors=det, ppe_grace_s=ppe_grace_s,
                danger_grace_s=danger_grace_s, ppe_zone_only=ppe_zone_only,
                low_light=low_light, gate_mode=gate_mode, severity=severity,
                shifts=shifts, use_registry=use_registry,
                registry_path=registry_path)
            q.put(("__done__", zname))
        except BaseException as e:                     # noqa: BLE001
            q.put(("__error__", zname, repr(e)))

    threads = [threading.Thread(target=worker, args=(zn, i, s))
               for i, (zn, s) in enumerate(zone_sources, 1)]
    for t in threads:
        t.start()
    finished, failure = set(), None
    while len(finished) < len(threads):
        item = q.get()
        if item[0] == "__done__":
            finished.add(item[1])
        elif item[0] == "__error__":
            failure = failure or item[2]
            finished.add(item[1])
            # keep draining so the other worker is not stuck on a full queue
        elif failure is None and on_zone_frame is not None:
            try:
                _, zi, annot, fi, n_p, n_e, counts, n_inc, recs, pinc = item
                on_zone_frame(item[0], zi, len(zone_sources), annot, fi,
                              n_p, n_e, counts, n_inc, recs, pinc)
            except Exception:                          # noqa: BLE001
                failure = "live view failed"
    for t in threads:
        t.join()
    if failure:
        raise RuntimeError(failure)

    # ---------------- merge both zones into one events.json ----------------
    multi = {"multi_zone": True, "zone_sources": [], "worker_zones": {},
             "incidents": [], "person_frames": [], "events": [],
             "gate_checks": [],
             "activity": {"totals_s": defaultdict(float), "per_worker_s": {},
                          "segments": [], "config": None}}
    alert_counts = defaultdict(int)
    total_frames = 0
    t0 = time.time()
    zone_fps = []
    zone_grace = []
    started_ats = []

    pid_off = 0
    for zi, (zname, src) in enumerate(zone_sources, 1):
        evj = out_dir / f"zone{zi}_events.json"
        d = json.loads(evj.read_text())
        evj.unlink()
        av = str(out_dir / f"annotated_zone{zi}.mp4")

        rows = d.get("person_frames") or []
        ids = {r["person"] for r in rows} | \
              {i["person"] for i in d.get("incidents", [])}
        off = pid_off
        for r in rows:
            r["person"] += off
            r["zone"] = zname
            multi["worker_zones"][str(r["person"])] = zname
        for i in d.get("incidents", []):
            i["person"] += off
            dz = i.pop("zone", None)          # in-video polygon name
            if dz and i["type"] == "danger_zone":
                i["dz_poly"] = dz
            i["zone"] = zname
        for e in d.get("events", []):
            if "person" in e:
                e["person"] += off
            e["zone"] = zname
        for g in d.get("gate_checks") or []:
            g = dict(g)
            g["person"] += off
            g["zone"] = zname
            multi["gate_checks"].append(g)
        pid_off = off + (max(ids) + 1 if ids else 0)

        act = d.get("activity") or {}
        multi["activity"]["per_worker_s"].update(
            {str(int(k) + off): v for k, v in act.get("per_worker_s", {}).items()})
        for k, v in (act.get("totals_s") or {}).items():
            multi["activity"]["totals_s"][k] += v
        for s in act.get("segments", []):
            s["person"] += off
            s["zone"] = zname
            multi["activity"]["segments"].append(s)
        multi["activity"]["config"] = multi["activity"]["config"] or \
            act.get("config")

        zs = d.get("stats", {})
        if zs.get("video_fps"):
            zone_fps.append((d.get("total_frames", 0) or 1,
                             zs["video_fps"]))
        if zs.get("ppe_grace_s") is not None:
            zone_grace.append(zs["ppe_grace_s"])
        if zs.get("started_at"):
            started_ats.append(zs["started_at"])
        n_rows = len(rows)
        multi["zone_sources"].append({
            "name": zname, "video": d.get("video"),
            "annotated": Path(av).name,
            "workers": zs.get("workers_seen", 0),
            "person_frames": n_rows,
            "incidents": len(d.get("incidents", [])),
            "helmet_pct": round(100 * sum(bool(r.get("helmet")) for r in rows)
                                / max(n_rows, 1), 1),
            "vest_pct": round(100 * sum(bool(r.get("vest")) for r in rows)
                              / max(n_rows, 1), 1),
            "equipment_seen": zs.get("equipment_seen", {}),
            "processing_fps": d.get("processing_fps", 0),
            "raw_tracks": zs.get("track_ids_raw", 0),
        })
        multi["incidents"] += d.get("incidents", [])
        multi["person_frames"] += rows
        multi["events"] += d.get("events", [])
        for k, v in (d.get("alert_counts") or {}).items():
            alert_counts[k] += v
        total_frames += d.get("total_frames", 0)

    multi["video"] = multi["zone_sources"][0]["video"] if multi["zone_sources"] else ""
    multi["total_frames"] = total_frames
    multi["processing_fps"] = round(total_frames / max(time.time() - t0, 1e-6), 1)
    multi["alert_counts"] = dict(alert_counts)
    eq = defaultdict(int)
    for z in multi["zone_sources"]:
        for k, v in z["equipment_seen"].items():
            eq[k] += v
    multi["stats"] = {
        "person_frames": len(multi["person_frames"]),
        "workers_seen": sum(z["workers"] for z in multi["zone_sources"]),
        "track_ids_raw": sum(z.get("raw_tracks", 0)
                             for z in multi["zone_sources"]),
        "max_concurrent": max([z["person_frames"] for z in multi["zone_sources"]]
                              + [0]),
        "equipment_seen": dict(eq),
        "video_fps": round(total_frames / sum(n / f for n, f in zone_fps), 1)
        if zone_fps and total_frames else 25.0,
        "stride": stride,
        "ppe_grace_s": round(sum(zone_grace) / len(zone_grace), 1)
        if zone_grace else ppe_grace_s,
        "ppe_grace_requested_s": ppe_grace_s,
        "danger_grace_s": danger_grace_s,
        "ppe_zone_only": bool(ppe_zone_only),
        "low_light": bool(low_light),
        "gate_mode": bool(gate_mode),
        "severity": severity_map(severity),
        "proximity_base_m": danger_radius_m,
        "started_at": min(started_ats) if started_ats
        else time.strftime("%Y-%m-%d %H:%M:%S"),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_zone": {z["name"]: {"workers": z["workers"],
                                 "incidents": z["incidents"],
                                 "helmet_pct": z["helmet_pct"],
                                 "vest_pct": z["vest_pct"]}
                     for z in multi["zone_sources"]},
    }
    multi["activity"]["totals_s"] = dict(multi["activity"]["totals_s"])
    multi["shifts"] = shift_summary(multi, shifts)

    with open(out_dir / "events.json", "w") as f:
        json.dump(multi, f, indent=2)
    inc_counts = defaultdict(int)
    for i in multi["incidents"]:
        inc_counts[i["type"]] += 1
    return dict(inc_counts)


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def activity_summary(rows, fps, stride, gap_s=1.0):
    """Collapse per-frame activity labels into per-worker time segments,
    per-worker seconds per activity, and site-wide totals."""
    from collections import defaultdict

    per = defaultdict(lambda: defaultdict(float))
    segments, cur = [], None
    gap_p = max(2, int(gap_s * fps / max(stride, 1)))
    dt = max(stride, 1) / max(fps, 1e-6)
    for r in sorted(rows, key=lambda r: (r["person"], r["pidx"])):
        act = r.get("activity") or "unknown"
        if cur is not None and cur["person"] == r["person"] \
                and cur["activity"] == act and r["pidx"] - cur["end_p"] <= gap_p:
            cur["end_s"] = r["time_s"]
            cur["end_p"] = r["pidx"]
        else:
            if cur is not None:
                segments.append(cur)
            cur = {"person": r["person"], "activity": act,
                   "start_s": r["time_s"], "end_s": r["time_s"],
                   "end_p": r["pidx"]}
    if cur is not None:
        segments.append(cur)
    totals = defaultdict(float)
    for s in segments:
        s["duration_s"] = round(s["end_s"] - s["start_s"] + dt, 1)
        s["end_s"] = round(s["end_s"], 1)
        s.pop("end_p")
        per[str(s["person"])][s["activity"]] += s["duration_s"]
        totals[s["activity"]] += s["duration_s"]
    return {
        "totals_s": {k: round(v, 1) for k, v in sorted(totals.items())},
        "per_worker_s": {k: {a: round(s, 1) for a, s in sorted(v.items())}
                         for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))},
        "segments": segments,
        "config": {"walk_speed_h_per_s": WALK_SPEED, "work_speed_h_per_s": WORK_SPEED,
                   "idle_after_s": IDLE_AFTER_S, "confirm_s": CONFIRM_S},
    }


def merge_fragmented_tracks(rows, gap_frames=90, dist_factor=3.0,
                            embeddings=None, overlap_ok=3):
    """Chain fragmented track IDs of the same physical person.

    ByteTrack issues a new ID when it loses a worker (occlusion, blur, dropout).
    A later track B is merged into an earlier track A when A ends shortly before
    B starts, their positions are within ~dist_factor * person-height, and they
    barely co-exist in time (small overlaps are tolerated: trackers keep lost
    tracks alive for a while, so a re-detected worker's new ID can overlap its
    zombie predecessor). Geometry decides; ReID embeddings only break ties when
    decisively similar (>= REID_CONFIRM_SIM) — they never veto, since noisy
    appearance samples on short fragments caused missed merges. Returns
    {old_id: new_id} with ids renumbered by first appearance.
    """
    from collections import defaultdict

    tracks = defaultdict(list)
    for r in rows:
        tracks[r["person"]].append(r)
    summ = {}
    for tid, rs in tracks.items():
        rs.sort(key=lambda r: r["pidx"])
        f, l = rs[0], rs[-1]
        mc = np.mean([_center(r["bbox"]) for r in rs], axis=0)
        summ[tid] = {"start": f["pidx"], "end": l["pidx"],
                     "fp": _center(f["bbox"]), "lp": _center(l["bbox"]),
                     "mc": tuple(mc),
                     "h": ((f["bbox"][3] - f["bbox"][1]) + (l["bbox"][3] - l["bbox"][1])) / 2}
    parent = {t: t for t in summ}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def cos_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    # pass 1 — concurrent duplicate tracks: the tracker occasionally emits a
    # second ID for the same person at the same time. Detected by per-frame
    # co-location (median centre distance on shared frames, robust for moving
    # workers — mean-track centres are not). Merge the shorter into the longer;
    # ReID appearance can veto.
    idxs = {t: set(r["pidx"] for r in tracks[t]) for t in summ}
    cts = {t: {r["pidx"]: _center(r["bbox"]) for r in tracks[t]} for t in summ}
    hts = {t: float(np.mean([r["bbox"][3] - r["bbox"][1]
                             for r in tracks[t]])) for t in summ}
    tids = sorted(summ, key=lambda t: summ[t]["start"])
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            if find(a) == find(b):
                continue
            sa, sb = summ[a], summ[b]
            if sb["end"] < sa["start"]:     # b entirely before a (sorted, rare)
                continue
            common = idxs[a] & idxs[b]
            if len(common) < 10:
                continue
            overlap = len(common) / max(min(len(idxs[a]), len(idxs[b])), 1)
            if overlap <= 0.6:
                continue
            dmed = float(np.median([
                math.hypot(cts[a][f][0] - cts[b][f][0],
                           cts[a][f][1] - cts[b][f][1]) for f in common]))
            if dmed >= 0.25 * max(hts[a], hts[b], 1):
                continue
            short, long_ = (a, b) if len(idxs[a]) < len(idxs[b]) else (b, a)
            parent[find(short)] = find(long_)

    # pass 2 — chain fragments: geometry proposes, appearance (ReID) confirms
    order = sorted(summ, key=lambda t: summ[t]["start"])
    for b in order:
        cands = []
        for a in order:
            if a == b or find(a) == find(b):
                continue
            sa, sb = summ[a], summ[b]
            if sb["start"] - sa["end"] > gap_frames or \
                    sa["end"] - sb["start"] > overlap_ok:
                continue
            dpx = math.hypot(sa["lp"][0] - sb["fp"][0], sa["lp"][1] - sb["fp"][1])
            if dpx < dist_factor * max(sa["h"], sb["h"], 1):
                cands.append((dpx, a))
        if not cands:
            continue
        eb = embeddings.get(b) if embeddings else None
        if eb is not None:
            sims = [(cos_sim(embeddings[c[1]], eb), c[1]) for c in cands
                    if c[1] in embeddings]
            if sims and max(sims)[0] >= REID_CONFIRM_SIM:
                parent[find(b)] = find(max(sims)[1])   # decisive appearance match
                continue
        parent[find(b)] = find(min(cands)[1])           # geometry decides
    groups = defaultdict(list)
    for t in summ:
        groups[find(t)].append(t)
    mapping = {}
    for i, (_, members) in enumerate(
            sorted(groups.items(), key=lambda kv: min(summ[m]["start"] for m in kv[1])), 1):
        for m in members:
            mapping[m] = i
    return mapping


def consolidate_incidents(incidents, fps, gap=None):
    """After ID remap, merge same-type incidents of the same person that are
    adjacent in time (were separate only because of the ID split). Gap is in
    raw video frames; default keeps a fixed ~1.6 s wall-clock window at any
    stride."""
    from collections import defaultdict

    if gap is None:
        gap = int(1.6 * fps)

    groups = defaultdict(list)
    for i in incidents:
        groups[(i["type"], i["person"], i.get("equipment"),
                i.get("zone"))].append(i)
    out = []
    for items in groups.values():
        items.sort(key=lambda i: i["start_frame"])
        cur = None
        for it in items:
            if cur and it["start_frame"] - cur["end_frame"] <= gap:
                if it.get("clip") and (not cur.get("clip")
                                       or it["end_frame"] - it["start_frame"] >
                                       cur["end_frame"] - cur["start_frame"]):
                    cur["clip"] = it["clip"]
                cur["end_frame"] = max(cur["end_frame"], it["end_frame"])
                cur["end_s"] = it["end_s"]
                cur["duration_s"] = round((cur["end_frame"] - cur["start_frame"]) / fps, 1)
            else:
                if cur:
                    out.append(cur)
                cur = dict(it)
        if cur:
            out.append(cur)
    out.sort(key=lambda i: i["start_frame"])
    return out


def zone_i(pts):
    return np.array(pts, dtype=np.int32)


def draw_box(img, p1, p2, color, t=3):
    """High-contrast box: white halo around a dark core, visible on any
    background (bright or dark footage)."""
    cv2.rectangle(img, p1, p2, (255, 255, 255), t + 3)
    cv2.rectangle(img, p1, p2, color, t)


def draw_label(img, text, x, y, accent=None, scale=0.5, thick=2, taken=None):
    """Dark label chip with bold white text (and a colored accent bar),
    clamped so it always stays fully inside the frame. When `taken` is a
    list of already-placed chip rects, the chip shifts down until it does
    not overlap any of them, so worker/equipment labels never jumble."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 5
    ch = th + 2 * pad
    H, W = img.shape[:2]
    x = int(max(0, min(x, W - tw - 2 * pad - 5)))
    y = int(max(2, min(y, H - ch - 2)))
    if taken is not None:
        w_box = tw + 2 * pad + 5
        while y + ch < H - 2 and any(
                x < tx2 + 4 and tx1 - 4 < x + w_box and
                y < ty2 + 3 and ty1 - 3 < y + ch
                for tx1, ty1, tx2, ty2 in taken):
            y += ch + 4
        taken.append((x, y, x + w_box, y + ch))
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad + 5, y + ch), (15, 15, 15), -1)
    if accent is not None:
        cv2.rectangle(img, (x, y), (x + 4, y + ch), accent, -1)
    cv2.putText(img, text, (x + pad + 5, y + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def render_clip_frame(raw, items, inc, zones, fi, fps):
    """Render one frame of an alert clip showing ONLY the worker involved in
    the incident (plus the machinery box for proximity alerts). Other
    detections are intentionally left out, so each clip is about one
    violation and one person."""
    img = raw.copy()
    for z in zones:
        zi = zone_i(z["poly"])
        zc = ZONE_C.get(z["name"], (255, 0, 255))
        overlay = img.copy()
        cv2.fillPoly(overlay, [zi], zc)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [zi], True, zc, 2)
        cv2.putText(img, z["name"].upper(), (zi[0][0], zi[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, zc, 2)
    for it in items:
        if it.get("person") == inc["person"]:
            draw_box(img, (int(it["x1"]), int(it["y1"])),
                     (int(it["x2"]), int(it["y2"])), it["color"])
            if it.get("kpts") is not None:
                for kx, ky, kc in it["kpts"]:
                    if kc > 0.3:
                        cv2.circle(img, (int(kx), int(ky)), 3,
                                   (255, 255, 255), -1)
            draw_label(img, it["tag"], int(it["x1"]),
                       int(it["y1"]) - 27 if it["y1"] > 29 else int(it["y1"]) + 4,
                       accent=it["color"])
        elif inc["type"] == "proximity" and it.get("eq") == inc.get("equipment"):
            draw_box(img, (int(it["x1"]), int(it["y1"])),
                     (int(it["x2"]), int(it["y2"])), it["color"])
            draw_label(img, it["tag"], int(it["x1"]),
                       int(it["y1"]) - 28 if it["y1"] > 30 else int(it["y1"]) + 4,
                       accent=it["color"], scale=0.55)
    # clip header: incident + worker + zone + time (no site-wide counts)
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 0), (W, 40), (25, 25, 25), -1)
    head = f"{ALERT_LABELS[inc['type']]} - WORKER {inc['person']}"
    if inc.get("zone"):
        head += f" - {inc['zone'].upper()}"
    elif inc["type"] == "proximity" and inc.get("equipment"):
        head += f" - {inc['equipment'].upper()}"
    cv2.putText(img, head, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                ALERT_TEXT[inc["type"]], 2)
    cv2.putText(img, f"t={fi / max(fps, 1):.1f}s", (W - 135, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def reencode_h264(path):
    """Re-encode mp4v -> H.264 (yuv420p, faststart) so browsers/Streamlit play it."""
    import subprocess

    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = path + ".tmp.mp4"
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
                   check=True)
    os.replace(tmp, path)


def draw_hud(img, fi, fps, counts, n_persons, n_equip):
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 0), (W, 46), (25, 25, 25), -1)
    txt = f"Frame {fi} | {fps:.0f} fps | Workers {n_persons} | Equipment {n_equip} | "
    txt += " ".join(f"{ALERT_LABELS[k].split()[0].lower()}:{v}" for k, v in counts.items())
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input/videos/construction.mp4")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--out", default="output/demo_annotated.mp4")
    ap.add_argument("--events", default="output/events.json")
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--conf-person", type=float, default=0.3)
    ap.add_argument("--conf-ppe", type=float, default=None,
                    help="model-level PPE conf floor; per-class operating "
                         "points in vision.detectors.PPE_CONF apply above it")
    ap.add_argument("--conf-equip", type=float, default=0.10,
                    help="machinery conf floor; a clearly visible truck in "
                         "the sample footage peaks at 0.19, so 0.2 hid it. "
                         "Size floor + temporal confirmation reject noise.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ppe-grace", type=float, default=120.0,
                    help="seconds of continuous missing PPE before alerting "
                         "(auto-capped to 25%% of clip length on files)")
    ap.add_argument("--danger-grace", type=float, default=1.0,
                    help="seconds inside the danger zone before alerting")
    ap.add_argument("--zone", default=None, help="JSON list of polygon points")
    ap.add_argument("--ppe-zone-only", action="store_true",
                    help="helmet/vest alerts only inside the defined zones")
    ap.add_argument("--low-light", action="store_true",
                    help="CLAHE-enhance frames for model inference")
    ap.add_argument("--gate-mode", action="store_true",
                    help="check PPE once per worker at first sighting "
                         "instead of continuous alerts")
    ap.add_argument("--severity", default=None,
                    help="JSON severity overrides, e.g. "
                         "'{\"no_helmet\": \"Medium\"}'")
    ap.add_argument("--no-registry", action="store_true",
                    help="disable the persistent W-xx worker registry")
    a = ap.parse_args()
    zone = json.loads(a.zone) if a.zone else None
    src = 0 if a.webcam else a.video
    zones_arg = [{"name": "Danger Zone", "poly": zone}] if zone else None
    counts = run(src, a.out, a.events, zones=zones_arg, zone=None,
                 danger_radius_m=a.radius,
                 conf_person=a.conf_person, conf_ppe=a.conf_ppe,
                 conf_equip=a.conf_equip, stride=a.stride,
                 ppe_grace_s=a.ppe_grace, danger_grace_s=a.danger_grace,
                 ppe_zone_only=a.ppe_zone_only, low_light=a.low_light,
                 gate_mode=a.gate_mode,
                 severity=json.loads(a.severity) if a.severity else None,
                 use_registry=not a.no_registry)
    print("incidents:", counts)
    print("saved:", a.out, a.events)
