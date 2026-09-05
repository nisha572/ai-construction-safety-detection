"""Pure-geometry safety rules: PPE compliance, fall, danger zone, proximity."""
import math

import cv2
import numpy as np

# COCO keypoints: 5 Lsh 6 Rsh 11 Lhip 12 Rhip 13 Lknee 14 Rknee


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match_ppe(person_box, ppe_dets, iou_thresh=0.15):
    """Associate PPE detections with a person box by centre-containment + IoU."""
    matched = {}
    px1, py1, px2, py2 = person_box[:4]
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    for cls, boxes in ppe_dets.items():
        best, best_score = None, 0.0
        for bx1, by1, bx2, by2, conf in boxes:
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            inside = px1 <= bcx <= px2 and py1 <= bcy <= py2
            score = (0.6 + 0.4 * iou(person_box, (bx1, by1, bx2, by2))) if inside else iou(person_box, (bx1, by1, bx2, by2))
            if score > best_score:
                best, best_score = cls, score
        if best is not None and best_score >= iou_thresh:
            matched[cls] = True
    return matched


def ppe_violations(person_box, ppe_dets, required=("helmet", "vest")):
    matched = match_ppe(person_box, ppe_dets)
    missing = [item for item in required if item not in matched]
    return missing, matched


def pose_state(kpts, box_h):
    """Classify standing / sitting / possible_fall from pretrained keypoints.

    Image coords: y grows downward, so a lying torso has |dy| small vs |dx|.
    """
    if box_h <= 0:
        return "unknown"

    def visible(idxs):
        return [i for i in idxs if kpts[i][2] > 0.3]

    sh_i, hp_i, kn_i = visible((5, 6)), visible((11, 12)), visible((13, 14))
    if not sh_i or not hp_i:
        return "unknown"
    sh = kpts[sh_i][:, :2].mean(0)
    hp = kpts[hp_i][:, :2].mean(0)
    sv = hp - sh                                   # shoulder -> hip vector
    torso = float(np.linalg.norm(sv))

    # fallen: torso roughly horizontal AND long relative to (now short) bbox height
    if torso / max(box_h, 1e-6) > 0.45 and abs(sv[1]) < abs(sv[0]) * 0.6:
        return "possible_fall"

    # sitting: thigh roughly horizontal (knees at hip height)
    if kn_i:
        kn = kpts[kn_i][:, :2].mean(0)
        thigh = kn - hp
        if abs(thigh[1]) < abs(thigh[0]) * 0.6 and abs(thigh[1]) < box_h * 0.15:
            return "sitting"
    return "standing"


class FallDetector:
    """Temporal confirmation layer on top of the single-frame pose_state.

    A raw "possible_fall" only becomes a confirmed alert when it persists
    across several observations or follows a sudden downward hip movement;
    a confirmed fall clears only after the worker is upright again for a
    few consecutive frames. Cuts one-frame classifier flicker false alarms.
    """

    def __init__(self, confirm_votes=3, recover_votes=4, window=45,
                 drop_ratio=0.30):
        self.confirm_votes = confirm_votes   # raw-fall observations needed
        self.recover_votes = recover_votes   # upright observations to clear
        self.window = window                 # processed-frame lookback
        self.drop_ratio = drop_ratio         # hip drop vs box height
        self.hist = {}                       # tid -> state dict

    def update(self, tid, raw_state, kpts, box_h, pidx):
        """Feed one pose observation; returns the state to display/alert on."""
        h = self.hist.setdefault(
            tid, {"votes": [], "hips": [], "fall_votes": 0,
                  "recover": 0, "confirmed": False, "last": pidx})

        hip_y = None
        if kpts is not None:
            vis = [kpts[i][1] for i in (11, 12) if kpts[i][2] > 0.3]
            if vis:
                hip_y = float(np.mean(vis))

        h["votes"].append((pidx, raw_state == "possible_fall"))
        if hip_y is not None and box_h > 0:
            h["hips"].append((pidx, hip_y, box_h))
        # drop observations older than the window
        for k in ("votes", "hips"):
            while h[k] and pidx - h[k][0][0] > self.window:
                h[k].pop(0)

        if raw_state == "possible_fall":
            h["recover"] = 0
            if not h["confirmed"]:
                n_fall = sum(1 for _, f in h["votes"] if f)
                # (a) sustained horizontal torso, or (b) sudden hip drop then fall
                # (use the tallest box in the window: a lying box is short)
                dropped = False
                if hip_y is not None and len(h["hips"]) >= 2:
                    _, y0, bh0 = min(h["hips"], key=lambda t: t[1])
                    bh_max = max(b for _, _, b in h["hips"])
                    if (hip_y - y0) / max(bh_max, bh0, 1) >= self.drop_ratio:
                        dropped = True
                if n_fall >= self.confirm_votes or dropped:
                    h["confirmed"] = True
        else:
            if h["confirmed"]:
                h["recover"] += 1
                if h["recover"] >= self.recover_votes:
                    h["confirmed"] = False
                    h["recover"] = 0
                    h["votes"].clear()

        return "possible_fall" if h["confirmed"] else raw_state


def hivis_score(frame, box, s_min=110, v_min=110):
    """Fraction of the torso wearing *fluorescent* hi-vis colour.

    Conservative on purpose. An earlier version also counted a generic blue
    band (H 90-115) at low saturation, which fired on denim, shadow, sky and
    glass facades: measured against 29 hand-labelled worker crops it marked a
    vest on 10 out of 10 workers who were not wearing one (a 100% false-
    positive rate), which silently disabled every no-vest alert.

    Only orange->lime hues at high saturation AND high value survive here,
    sampled from a narrow torso window so background does not leak in.
    Returns 0.0 for tiny/invalid boxes.
    """
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return 0.0
    torso = frame[max(0, int(y1 + bh * 0.25)):int(y1 + bh * 0.60),
                  max(0, int(x1 + bw * 0.22)):int(x2 - bw * 0.22)]
    if torso.size == 0:
        return 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], bool)
    for lo, hi in (((8, s_min, v_min), (32, 255, 255)),    # orange -> yellow
                   ((32, s_min, v_min), (52, 255, 255))):  # yellow -> lime
        mask |= cv2.inRange(hsv, lo, hi).astype(bool)
    return float(mask.mean())


def point_in_zone(pt, zone):
    return cv2.pointPolygonTest(np.array(zone, np.int32), pt, False) >= 0


def box_in_zone(box, zone, thresh=0.25):
    """True if >= thresh of the person's lower-centre area is inside the zone."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    sample_pts = [((x1 + x2) / 2, y2),                # feet
                  (x1 + w * 0.25, y2 - h * 0.05),
                  (x1 + w * 0.75, y2 - h * 0.05),
                  ((x1 + x2) / 2, y1 + h * 0.75)]     # lower torso
    hits = sum(point_in_zone(p, zone) for p in sample_pts)
    return hits >= max(1, int(len(sample_pts) * thresh))


def ground_center(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, y2)


def pixel_to_m(px, m_per_px=None, h_est_px=520.0, h_est_m=1.7):
    """Monocular pixel->metre scale. Prefer a live calibration (m_per_px,
    from the median tracked person height assuming ~1.7 m); fall back to a
    fixed reference height for legacy callers / warm-up frames."""
    if m_per_px is None:
        m_per_px = h_est_m / max(h_est_px, 1e-6)
    return px * m_per_px


# Extra keep-out margin per machine type (metres), added on top of the base
# danger radius: big/slow machinery needs a wider berth than a pickup.
EQUIP_EXTRA_M = {"crane": 2.0, "excavator": 1.0, "bulldozer": 0.5,
                 "dump truck": 0.5, "wheel loader": 0.5, "bus": 0.5,
                 "forklift": 0.25, "tractor": 0.25, "truck": 0.25,
                 "van": 0.0, "pickup truck": 0.0, "car": 0.0}


def proximity_limit(label, base_m=3.0, extra=None):
    """Keep-out distance for one machine label: base radius + type margin."""
    extra = EQUIP_EXTRA_M if extra is None else extra
    return float(base_m) + float(extra.get(label, 0.0))


def unsafe_proximity(persons, equipment, danger_radius_m=3.0, min_person_h=60.0,
                     m_per_px=None, extra_radii=None):
    """Pair every worker with nearby machinery using ground-centre distance.

    The alert threshold is per equipment type: danger_radius_m plus the
    EQUIP_EXTRA_M margin (a crane alerts much further out than a van)."""
    alerts = []
    eq_scaled = []
    for label, ex1, ey1, ex2, ey2, _c in equipment:
        eq_scaled.append((label, ground_center((ex1, ey1, ex2, ey2)),
                          proximity_limit(label, danger_radius_m, extra_radii)))
    for tid, x1, y1, x2, y2, conf in persons:
        h = y2 - y1
        if h < min_person_h:
            continue
        g = ground_center((x1, y1, x2, y2))
        for label, eg, limit in eq_scaled:
            d_px = math.hypot(g[0] - eg[0], g[1] - eg[1])
            d_m = pixel_to_m(d_px, m_per_px=m_per_px)
            if d_m < limit:
                alerts.append({"person": tid, "equipment": label, "dist_m": round(d_m, 1)})
    return alerts


def default_zone(w, h):
    """Default danger zone polygon (bottom-centre area of the frame)."""
    return [(int(w * 0.35), int(h * 0.72)),
            (int(w * 0.65), int(h * 0.72)),
            (int(w * 0.75), int(h * 0.98)),
            (int(w * 0.25), int(h * 0.98))]
