"""Per-worker activity classification: working / idle / sitting / standing /
walking / fallen.

Three signals are fused per tracked worker, all normalised by person height
so they work at any distance:
  - bbox-centre translation speed      -> walking
  - arm motion from YOLO pose wrists/elbows, relative to the body -> working
  - safety_rules pose state (temporal) -> sitting / confirmed fall

Temporal smoothing keeps false detections down: speeds go through an EMA,
activity switches are debounced (a new label must persist CONFIRM_S worth of
frames, except 'fallen' which is already temporally confirmed upstream), and
'standing' only decays into 'idle' after IDLE_AFTER_S of no activity.
"""
import math

WALK_SPEED = 0.25   # bbox speed in person-heights/s above which = walking
WORK_SPEED = 0.16   # arm-keypoint speed in person-heights/s above which = working
IDLE_AFTER_S = 5.0  # stationary + no arm motion this long -> idle
CONFIRM_S = 1.0     # a new activity must persist this long before switching
TAU = 0.6           # EMA time constant for the speeds (seconds)
MAX_GAP_S = 1.0     # sightings further apart add no motion and no elapsed time

KEYPTS = (7, 8, 9, 10)   # COCO elbows + wrists


class ActivityTracker:
    """Feed one observation per worker per processed frame via update();
    get back the temporally smoothed activity label."""

    def __init__(self, fps=25.0, stride=1):
        self.fps = max(fps, 1e-6)
        self.dt_frame = max(stride, 1) / self.fps
        self.confirm_n = max(2, round(CONFIRM_S / self.dt_frame))
        self.hist = {}

    def update(self, tid, bbox, kpts, pose, fi):
        """tid: track id; bbox: (x1, y1, x2, y2); kpts: 17x3 keypoints or None
        (only pass keypoints matched on THIS frame); pose: safety_rules pose
        state ('standing'/'sitting'/'possible_fall'/'unknown'); fi: raw frame
        index. Returns one of the six activity labels."""
        if tid is None or tid < 0:
            return "unknown"
        h = max(bbox[3] - bbox[1], 20.0)
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        st = self.hist.setdefault(tid, {
            "cx": cx, "cy": cy, "fi": fi, "kp": None,
            "v_c": 0.0, "v_k": 0.0, "cur": "standing", "cand": None,
            "votes": 0, "active_f": fi,
        })
        dt = (fi - st["fi"]) / self.fps
        if dt > MAX_GAP_S:          # worker reappearing after a gap: no motion
            st.update(cx=cx, cy=cy, fi=fi, kp=None, v_c=0.0, v_k=0.0)
        elif dt > 0:
            alpha = 1.0 - math.exp(-dt / TAU)
            v_c = math.hypot(cx - st["cx"], cy - st["cy"]) / h / dt
            st["v_c"] += alpha * (v_c - st["v_c"])
            if kpts is not None:
                cur = {i: (float(kpts[i][0]), float(kpts[i][1]))
                       for i in KEYPTS if kpts[i][2] > 0.3}
                prev = st["kp"] or {}
                common = [i for i in cur if i in prev]
                if common:
                    v_k = sum(math.hypot(cur[i][0] - prev[i][0],
                                         cur[i][1] - prev[i][1])
                              for i in common) / len(common) / h / dt
                    v_k = max(0.0, v_k - v_c)     # arm motion relative to body
                    st["v_k"] += alpha * (v_k - st["v_k"])
                else:
                    st["v_k"] *= 1.0 - alpha      # no data -> decay
                st["kp"] = cur
            else:
                st["kp"] = None
                st["v_k"] *= 1.0 - alpha
            st["cx"], st["cy"], st["fi"] = cx, cy, fi
        # dt == 0 (first sighting / same frame): zero speeds, classify below

        if pose == "possible_fall":
            raw = "fallen"
        elif pose == "sitting":
            raw = "sitting"
        elif st["v_c"] >= WALK_SPEED:
            raw = "walking"
        elif st["v_k"] >= WORK_SPEED:
            raw = "working"
        else:
            raw = "standing"

        if raw == "fallen":         # already smoothed by the FallDetector
            st["cur"], st["cand"], st["votes"] = "fallen", None, 0
            st["active_f"] = fi
        elif raw != st["cur"]:
            if st["cand"] == raw:
                st["votes"] += 1
            else:
                st["cand"], st["votes"] = raw, 1
            if st["votes"] >= self.confirm_n:
                st["cur"], st["cand"], st["votes"] = raw, None, 0
                st["active_f"] = fi       # confirmed switch = real activity
        else:
            st["cand"], st["votes"] = None, 0
        return self._out(st, fi)

    def _out(self, st, fi):
        if st["cur"] == "standing" and \
                (fi - st["active_f"]) / self.fps >= IDLE_AFTER_S:
            return "idle"
        return st["cur"]
