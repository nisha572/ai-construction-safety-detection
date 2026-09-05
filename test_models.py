"""Sanity test of all pretrained model families on sample frames of the
construction video. Skips automatically when models/videos are missing."""
from pathlib import Path

import pytest

VIDEO = Path("input/videos/construction.mp4")

pytestmark = pytest.mark.skipif(
    not VIDEO.exists(), reason="sample video not present")


def _frames():
    import cv2
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in [0, n // 3, 2 * n // 3, n - 2]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    print(f"\nvideo: {fps:.0f} fps, {n} frames, {len(frames)} test frames")
    return frames


def test_person_detection():
    from ultralytics import YOLO
    m = YOLO("models/yolov8n.pt")
    total = 0
    for r in m.predict(_frames(), classes=[0], conf=0.3, verbose=False):
        total += len(r.boxes)
    assert total > 0, "person detector found no people in the sample video"


def test_ppe_detection():
    from vision.detectors import _load_ppe
    ppe, ppe_map = _load_ppe()
    found = set()
    for r in ppe.predict(_frames(), conf=0.2, verbose=False):
        if r.boxes is not None:
            for c in r.boxes.cls:
                n = ppe.names[int(c)]
                found.add(ppe_map.get(n, n) if ppe_map else n)
    assert found, "PPE model returned no classes on sample frames"


def test_equipment_detection():
    from ultralytics import YOLOWorld
    m = YOLOWorld("models/yolov8x-worldv2.pt")
    try:
        m.set_classes(["excavator", "dump truck", "wheel loader", "person"])
    except Exception as e:
        pytest.skip(f"open-vocab prompts need the CLIP text encoder "
                    f"(network): {e}")
    seen = set()
    for r in m.predict(_frames(), conf=0.15, verbose=False):
        seen |= {m.names[int(c)] for c in r.boxes.cls} \
            if r.boxes is not None else set()
    assert "person" in seen, "YOLO-World found no persons (sanity anchor)"


def test_pose_estimation():
    from ultralytics import YOLO
    m = YOLO("models/yolov8n-pose.pt")
    total = 0
    for r in m.predict(_frames(), conf=0.3, verbose=False):
        total += len(r.boxes)
    assert total > 0, "pose model found no people in the sample video"
