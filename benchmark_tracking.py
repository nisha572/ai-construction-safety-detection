"""Benchmark the person detection/tracking layer only.

Runs one (YOLO26 model x tracker config) combination over the same video with
the exact production settings of Detectors.detect_persons (track every frame,
persist=True, classes=[0], conf=0.3) and measures:

- detection density / confidence (missed-detection proxies)
- raw track IDs, ghost-filtered IDs, final merged workers
- identity fragmentation (raw IDs per physical worker = ID-switch proxy)
- track gaps after merging (occlusion stability)
- tracking FPS

Results are appended to output/tracking_benchmark.json; `--compare` prints the
table and cross-config missed-detection estimates.
"""
import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from pipeline import merge_fragmented_tracks

VIDEO = "input/videos/20109885-sd_960_540_25fps.mp4"
RESULTS = Path("output/tracking_benchmark.json")
GHOST_MIN_FRAMES = 5   # same as pipeline.py
GAP_MIN = 3            # frames absent to count as a real track gap


def run_config(model_path, tracker_cfg, label, max_frames=None, conf=0.3):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(VIDEO)
    rows, frame_counts, confs = [], [], []
    t0 = time.time()
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if max_frames is not None and fi >= max_frames:
            break
        r = model.track(frame, persist=True, tracker=tracker_cfg,
                        classes=[0], conf=conf, verbose=False)[0]
        n = 0
        if r.boxes is not None:
            for b in r.boxes:
                tid = int(b.id.item()) if b.id is not None else -1
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                rows.append({"pidx": fi, "person": tid,
                             "bbox": [round(x1), round(y1), round(x2), round(y2)]})
                confs.append(float(b.conf[0]))
                n += 1
        frame_counts.append(n)
    cap.release()
    wall = time.time() - t0
    total = fi + 1

    # --- same post-processing as pipeline.py ---
    track_len = Counter(r["person"] for r in rows)
    keep = {t for t, n in track_len.items() if n >= GHOST_MIN_FRAMES}
    rows_kept = [r for r in rows if r["person"] in keep]
    mapping = merge_fragmented_tracks(rows_kept)

    # segments per final worker + gap analysis on merged timelines
    seg_per_worker = Counter(mapping.values())
    tl = defaultdict(list)
    for r in rows_kept:
        tl[mapping[r["person"]]].append(r["pidx"])
    gaps_all = []
    for wid, idxs in tl.items():
        idxs = sorted(set(idxs))
        for a, b in zip(idxs, idxs[1:]):
            if b - a > GAP_MIN:
                gaps_all.append(b - a - 1)

    res = {
        "label": label, "model": model_path, "tracker": tracker_cfg,
        "frames": total, "wall_s": round(wall, 1),
        "track_fps": round(total / max(wall, 1e-6), 1),
        "person_frames": len(rows),
        "person_frames_kept": len(rows_kept),
        "mean_count": round(len(rows) / max(total, 1), 2),
        "max_concurrent": max(frame_counts) if frame_counts else 0,
        "count_std": round(statistics.pstdev(frame_counts), 2) if frame_counts else 0,
        "frames_covered_pct": round(100 * sum(1 for c in frame_counts if c > 0)
                                    / max(total, 1), 1),
        "mean_conf": round(statistics.mean(confs), 3) if confs else 0,
        "median_conf": round(statistics.median(confs), 3) if confs else 0,
        "raw_ids_total": len(track_len),
        "raw_ids": len(keep),
        "final_workers": len(set(mapping.values())),
        "id_breaks_repaired": len(keep) - len(set(mapping.values())),
        "frag_ratio": round(len(keep) / max(len(set(mapping.values())), 1), 2),
        "seg_per_worker_median": round(statistics.median(seg_per_worker.values()), 1)
        if seg_per_worker else 0,
        "seg_per_worker_max": max(seg_per_worker.values()) if seg_per_worker else 0,
        "track_gaps": len(gaps_all),
        "gap_median_len": round(statistics.median(gaps_all), 1) if gaps_all else 0,
        "frame_counts": frame_counts,
    }
    return res


def save(res):
    RESULTS.parent.mkdir(exist_ok=True)
    db = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    db[res["label"]] = res
    RESULTS.write_text(json.dumps(db, indent=2))


def compare():
    db = json.loads(RESULTS.read_text())
    env = [max(cnts) for cnts in zip(*(r["frame_counts"] for r in db.values()))]
    env_total = sum(env)
    cols = ["label", "track_fps", "person_frames", "mean_count", "max_concurrent",
            "mean_conf", "raw_ids", "final_workers", "id_breaks_repaired",
            "frag_ratio", "seg_per_worker_max", "track_gaps", "gap_median_len"]
    head = f"{cols[0]:<28}" + "".join(f"{c:>18}" for c in cols[1:])
    print(head)
    print("-" * len(head))
    for r in sorted(db.values(), key=lambda r: r["label"]):
        missed = env_total - r["person_frames"]
        line = f"{r['label']:<28}" + "".join(f"{r[c]:>18}" for c in cols[1:])
        print(line)
        print(f"{'  rel missed vs envelope':<28}{100 * missed / max(env_total, 1):>17.1f}%"
              f"{'(abs ' + str(missed) + ')':>18}")
    print(f"\nenvelope (max persons/frame across configs): {env_total} person-frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tracker")
    ap.add_argument("--label")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    else:
        res = run_config(a.model, a.tracker, a.label, a.frames)
        save(res)
        del res["frame_counts"]
        print(json.dumps(res, indent=2))
