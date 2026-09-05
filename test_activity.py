"""Offline tests for worker activity classification (no video needed)."""
import numpy as np

from vision.activity import ActivityTracker

FPS = 25.0
STRIDE = 1
H = 200.0  # person height in px


def kpts(wrist_dx):
    """17x3 COCO keypoints: elbows/wrists offset by wrist_dx, all visible."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    k[7] = [100, 120, 0.9]              # L elbow
    k[8] = [160, 120, 0.9]              # R elbow
    k[9] = [90 + wrist_dx, 150, 0.9]    # L wrist
    k[10] = [170 + wrist_dx, 150, 0.9]  # R wrist
    return k


def run_scenario(tr, n, bbox_fn, kpt_fn, pose="standing"):
    out = []
    for i in range(n):
        x1, y1 = bbox_fn(i)
        bbox = (x1, y1, x1 + 80, y1 + H)
        out.append(tr.update(1, bbox, kpt_fn(i), pose, fi=i))
    return out


def test_static_then_idle():
    tr = ActivityTracker(FPS, STRIDE)
    out = run_scenario(tr, 200, lambda i: (100.0, 100.0), lambda i: kpts(0.0))
    assert out[-1] == "idle"
    assert "standing" in out


def test_walking():
    tr = ActivityTracker(FPS, STRIDE)
    out = run_scenario(tr, 60,
                       lambda i: (100.0 + 40 * i, 100.0), lambda i: kpts(0.0))
    assert out[-1] == "walking"


def test_working():
    tr = ActivityTracker(FPS, STRIDE)
    out = run_scenario(tr, 80, lambda i: (100.0, 100.0),
                       lambda i: kpts(25.0 if i % 2 else -25.0))
    assert out[-1] == "working"


def test_sitting_pose_wins():
    tr = ActivityTracker(FPS, STRIDE)
    out = run_scenario(tr, 60, lambda i: (100.0, 100.0), lambda i: kpts(0.0),
                       pose="sitting")
    assert out[-1] == "sitting"


def test_confirmed_fall_immediate():
    tr = ActivityTracker(FPS, STRIDE)
    acts = [tr.update(1, (100, 100, 180, 100 + H), kpts(0), "possible_fall",
                      fi=i) for i in range(3)]
    assert acts[0] == "fallen"          # no debounce delay for falls


def test_tracking_gap():
    tr = ActivityTracker(FPS, STRIDE)
    tr.update(1, (100, 100, 180, 100 + H), kpts(0), "standing", fi=0)
    after = tr.update(1, (400, 100, 480, 100 + H), kpts(0), "standing", fi=250)
    assert after in ("standing", "idle")   # no phantom speed from the jump


def test_single_frame_flicker_ignored():
    tr = ActivityTracker(FPS, STRIDE)
    out = []
    for i in range(40):
        k = kpts(60.0) if i == 20 else kpts(0.0)
        out.append(tr.update(1, (100, 100, 180, 100 + H), k, "standing", fi=i))
    assert out.count("working") == 0


def test_idle_resets_when_work_starts():
    tr = ActivityTracker(FPS, STRIDE)
    out = []
    for i in range(300):
        k = kpts(30.0 if i % 2 else -30.0) if 150 <= i else kpts(0.0)
        out.append(tr.update(1, (100, 100, 180, 100 + H), k, "standing", fi=i))
    assert "idle" in out
    assert out[-1] == "working"


def test_untracked_id():
    tr = ActivityTracker(FPS, STRIDE)
    assert tr.update(-1, (0, 0, 10, 10), None, "standing", fi=0) == "unknown"


def test_activity_summary():
    from pipeline import activity_summary

    rows = [
        {"person": 1, "pidx": 0, "time_s": 0.0, "activity": "standing"},
        {"person": 1, "pidx": 1, "time_s": 0.12, "activity": "standing"},
        {"person": 1, "pidx": 2, "time_s": 0.24, "activity": "working"},
        {"person": 1, "pidx": 3, "time_s": 0.36, "activity": "working"},
        {"person": 1, "pidx": 40, "time_s": 4.0, "activity": "working"},  # gap
        {"person": 2, "pidx": 0, "time_s": 0.0, "activity": "walking"},
    ]
    summ = activity_summary(rows, fps=25, stride=1)
    assert len(summ["segments"]) == 4            # gap split the working run
    assert abs(summ["totals_s"]["standing"] - 0.2) < 0.05
    assert abs(summ["totals_s"]["working"] - 0.2) < 0.05
    assert set(summ["per_worker_s"]) == {"1", "2"}
