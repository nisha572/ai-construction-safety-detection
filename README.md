# Construction AI · Site Safety

End-to-end computer-vision safety monitoring for construction sites, built
**entirely from pretrained models — no custom training**. Point it at a video
file, an RTSP camera or a webcam and it detects and tracks workers, checks PPE
compliance, spots heavy machinery, classifies what each worker is doing, and
applies geometric safety rules (danger-zone intrusion, unsafe proximity to
machinery, falls).

Every run produces an annotated H.264 video, per-incident clips, a structured
`events.json`, a one-page PDF safety report, and a Streamlit dashboard to
review it all.

```
video / RTSP / webcam
        │
        ▼
  person detection ──► BoT-SORT + ReID tracking ──► fragment merging ──► W-xx registry
        │                                                                     │
        ├─► PPE model on per-worker crops ──► per-class thresholds ──► hysteresis
        ├─► YOLO-World machinery detection ──► size floor + 3-of-4 confirmation
        └─► pose model ──► activity classifier + temporal fall confirmation
        │
        ▼
  geometric safety rules ──► sustained-violation gating ──► incidents + clips
        │
        ▼
  annotated H.264 video · events.json · PDF report · dashboard
```

---

## Table of contents

- [What it detects](#what-it-detects)
- [Measured accuracy](#measured-accuracy)
- [Install](#install)
- [Models](#models)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Output format](#output-format)
- [How it works](#how-it-works)
- [Detection tuning — the numbers behind the defaults](#detection-tuning--the-numbers-behind-the-defaults)
- [Known limitations](#known-limitations)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [Licence](#licence)

---

## What it detects

| Capability | How | Alert type |
|---|---|---|
| Workers | YOLO26s person detection + BoT-SORT/ReID tracking | — |
| Hard hat | Open-vocabulary PPE model on per-worker crops | `no_helmet` |
| Hi-vis vest | Same model + conservative fluorescent-colour assist | `no_vest` |
| Gloves / boots | Same model (**advisory only** — see limitations) | — |
| Heavy machinery | YOLO-World, 12 open-vocabulary machinery classes | — |
| Danger-zone intrusion | Point-in-polygon on the worker's ground contact | `danger_zone` |
| Unsafe proximity | Ground-plane distance, per-machine keep-out radius | `proximity` |
| Falls | Pose keypoints + temporal vote confirmation | `possible_fall` |
| Activity | Walking / working / idle / sitting / standing / fallen | — |

Alerts are **sustained-violation gated**: a missing helmet raises an alert only
after the worker has been continuously without it for `--ppe-grace` seconds, so
a single bad frame never produces an incident.

---

## Measured accuracy

Measured on the bundled sample footage, against **29 hand-labelled worker
crops** drawn from three different clips. This is a small validation set and
the numbers should be read as directional, not as a benchmark.

### PPE detection rate vs. ground truth

On the 44 s concrete-pour clip (`construction.mp4`, 9 tracked workers, 1794
person-frames), where every worker genuinely wears a hard hat and roughly two
thirds wear a hi-vis vest:

| Item | Ground truth | Reported | Verdict |
|---|---|---|---|
| Helmet | 100% | **99.2%** | matches |
| Vest | 65.5% (19/29 labelled crops) | **66.9%** | matches |

Before the tuning described below, the same clip reported **99.9% helmet and
99.9% vest** and raised **zero** PPE alerts — the system claimed perfect
compliance on footage containing real violations.

### End-to-end, before vs. after

| | Before | After |
|---|---|---|
| `construction.mp4` — vest reported present | 99.9% | 66.9% |
| `construction.mp4` — `no_vest` alert-frames (incidents) | 0 (0) | 9 (1) |
| `construction.mp4` — phantom machinery (confirmed frames) | — | 1 |
| Truck clip — machinery detected | **none** | truck, 30 frames |
| Truck clip — `proximity` incidents | **impossible** | 5 |
| Offline test suite | 38 passing | 43 passing |

Throughput on an Apple M-series CPU at `--stride 5`, 960 px input: **6.6–8.4
processed fps** (roughly 4 minutes for a 44 s clip). All models run on CPU here;
`--device mps`/CUDA is picked up automatically by Ultralytics when available.

---

## Install

Requires **Python 3.11–3.13**.

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls PyTorch, Ultralytics, OpenCV, Streamlit and ONNX Runtime — around
1.6 GB.

---

## Models

All weights are **pretrained and distributed separately** — they total ~1.1 GB
and three of them exceed GitHub's 100 MB file limit, so they are git-ignored.
Place them as follows before running:

| Path | Purpose | Size |
|---|---|---|
| `models/yolo26s.pt` | person detection + tracking | 19 MB |
| `models/yolo26n-pose.pt` | pose keypoints (COCO 17) | 7.5 MB |
| `models/yolov8x-worldv2.pt` | YOLO-World machinery detection | 140 MB |
| `yoloe-11s-seg.pt` | open-vocabulary PPE detection | 27 MB |
| `yolo26n-reid.onnx` | ReID appearance encoder for BoT-SORT | 9.4 MB |
| `models/best-4.pt` | legacy PPE fallback (optional) | 5.1 MB |

Fallbacks are automatic: `yolo26s` → `yolo26n`, `yolo26n-pose` →
`yolov8n-pose`, and the open-vocabulary PPE model → `models/best-4.pt` if YOLOE
or its text encoder is unavailable (e.g. offline). Model paths resolve against
the repository root, so the pipeline runs from any working directory.

---

## Quick start

```bash
# CLI
python pipeline.py --video input/videos/construction.mp4 --stride 3

# Dashboard
streamlit run dashboard/app.py
```

Expected CLI output:

```
incidents: {'danger_zone': 11, 'no_vest': 1}
saved: output/demo_annotated.mp4 output/events.json
```

More examples:

```bash
# Live camera
python pipeline.py --webcam

# Custom danger zone, in original video coordinates
python pipeline.py --video site.mp4 --zone '[[100,400],[800,400],[900,540],[50,540]]'

# Only alert on PPE inside the drawn zones (ignore passers-by)
python pipeline.py --video site.mp4 --ppe-zone-only

# Entry-gate mode: check each worker's PPE once, on first sighting
python pipeline.py --video gate.mp4 --gate-mode

# Night footage
python pipeline.py --video night.mp4 --low-light

# Alert immediately instead of waiting for a sustained violation
python pipeline.py --video site.mp4 --ppe-grace 0
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--video PATH` | `input/videos/construction.mp4` | Source video |
| `--webcam` | off | Use camera 0 instead of a file |
| `--out PATH` | `output/demo_annotated.mp4` | Annotated video output |
| `--events PATH` | `output/events.json` | Structured run record |
| `--stride N` | `1` | Process every Nth frame. `3` ≈ "Fast" |
| `--conf-person F` | `0.3` | Person detection confidence |
| `--conf-ppe F` | auto | Model-level PPE floor; per-class operating points in `vision.detectors.PPE_CONF` apply above it |
| `--conf-equip F` | `0.10` | Machinery confidence floor (see tuning notes) |
| `--radius M` | `3.0` | Base machinery keep-out radius in metres |
| `--ppe-grace S` | `120` | Seconds of continuous missing PPE before alerting; auto-capped to 25% of clip length |
| `--danger-grace S` | `1.0` | Seconds inside a zone before alerting |
| `--zone JSON` | default polygon | Danger-zone polygon in original video coordinates |
| `--ppe-zone-only` | off | PPE alerts only inside zones |
| `--gate-mode` | off | One PPE check per worker at first sighting |
| `--low-light` | off | CLAHE enhancement for inference only |
| `--severity JSON` | defaults | Per-alert severity overrides |
| `--no-registry` | off | Disable persistent `W-xx` worker identities |

---

## Output format

Each run writes `output/<name>/`:

- **Annotated video** — H.264 MP4 with boxes, PPE marks, zones, alert banner and HUD.
- **`clips/`** — one short MP4 per incident, re-rendered to show only the worker involved.
- **`events.json`** — the full run record.

```jsonc
{
  "video": "input/videos/construction.mp4",
  "total_frames": 1111,
  "zones": [{"name": "Danger Zone", "poly": [[336,388], ...]}],
  "processing_fps": 6.6,
  "alert_counts": {"danger_zone": 764, "no_vest": 9},
  "stats": {
    "workers_seen": 9,           // after fragment merging
    "max_concurrent": 9,
    "equipment_seen": {},
    "m_per_px": 0.0153,          // live scale calibration
    "ppe_grace_s": 11.1,         // auto-capped from the requested 120
    "registry_known": 9, "registry_new": 0
  },
  "incidents": [                 // consolidated, one per sustained violation
    {"type": "no_vest", "person": 3, "start_s": 12.4, "end_s": 18.2,
     "duration_s": 5.8, "clip": "clips/no_vest_w3_f310.mp4"}
  ],
  "person_frames": [...],        // per-worker per-frame records
  "activity": {"totals_s": {"working": 13.6, "walking": 77.0, ...}},
  "shifts": [...],
  "events": [...]
}
```

`person_frames` rows carry the per-frame PPE booleans, pose, activity, zone
membership and nearest-machine distance — everything the dashboard and PDF are
built from.

Generate the PDF separately:

```python
import report
report.make_pdf("output/events.json", "output/safety_report.pdf")
```

---

## How it works

**Tracking.** YOLO26s detections feed BoT-SORT with a dedicated ReID encoder
(`yolo26n-reid.onnx`). Trackers fragment when workers occlude each other, so a
union-find pass chains fragments back together: geometry proposes a merge
(gap, distance, minimal time overlap) and ReID appearance confirms it when
decisively similar. Tracks seen for fewer than 5 frames are dropped as ghosts.
A persistent registry then matches each merged worker against known `W-xx`
identities by appearance, so IDs survive across runs *and* across cameras.

**PPE.** Rather than detecting PPE across the whole frame and matching boxes to
people, the model runs on padded crops of each tracked worker. Tight crops make
small items far easier to detect and remove the fragile IoU-matching step
entirely. Each class then gets its own confidence threshold, and the result
passes through temporal hysteresis before it can raise an alert.

**Machinery.** YOLO-World with 12 open-vocabulary machinery prompts, every 5th
frame. Open-vocabulary scores flicker, so a label must be seen in 3 of the last
4 samples before it is drawn or used, and boxes below 48 px on a side are
discarded.

**Rules.** All geometry, no learning: point-in-polygon for zones, ground-plane
distance for proximity (with a per-machine keep-out margin — a crane gets +2 m,
a pickup +0), and pose-keypoint geometry for falls with vote-based temporal
confirmation. Pixel distances are converted to metres using a scale calibrated
live from the median tracked person height, assuming ~1.7 m.

---

## Detection tuning — the numbers behind the defaults

The defaults here are not guesses; each was measured. This section records what
was found, because the failure modes are subtle and easy to reintroduce.

### 1. The hi-vis colour fallback cleared every real violation

A colour check backed up the vest model for distant workers. Its hue ranges
included a **generic blue band at low saturation** — which matches denim, deep
shadow, sky, and the blue glass facade that fills the background of one sample
clip.

Measured on the 29 labelled crops, the original threshold fired on **10 out of
10 workers who were wearing no vest**: a 100% false-clear rate. Combined with
the latch below, this alone guaranteed that `no_vest` could never fire.

The scorer is now fluorescent-only (orange→lime, high saturation *and* value,
sampled from a narrow torso window) and the threshold is 0.45:

| Threshold | Fires on true vests | Fires on NON-vests (false clears) |
|---|---|---|
| Original scorer @ 0.10 | 19/19 | **10/10** |
| New scorer @ 0.10 | 11/19 | 3/10 |
| New scorer @ 0.30 | 4/19 | 1/10 |
| **New scorer @ 0.45** (default) | 3/19 | **0/10** |

It now only rescues unmistakable full-torso hi-vis, and never clears a real
violation on the validation set.

### 2. One detection latched PPE "present" for ~8 seconds

The old smoothing was asymmetric: a single detection marked an item present,
and it took **20 consecutive misses** to clear. PPE runs every 2nd processed
frame, so at `--stride 5` that is ~8 seconds of video held on one lucky hit.

Smoothing is now symmetric and expressed in **seconds** (`PPE_OFF_S = 1.5`), so
`--stride` no longer changes how long an item stays latched. A regression test
asserts this holds across strides 1–10.

### 3. One flat PPE threshold cannot serve all four classes

The open-vocabulary model is calibrated very differently per prompt. Measured
over 84 worker crops:

| Class | Median confidence | Detected at conf ≥ 0.1 |
|---|---|---|
| `helmet` | **0.754** | 95% of crops |
| `vest` | **0.065** | 27% of crops |
| `gloves` | 0.112 | 48% of crops |
| `boots` | 0.076 | 37% of crops |

A single threshold either floods helmets with noise or accepts every vest. Each
class now has its own operating point (`vision.detectors.PPE_CONF`).

### 4. Machinery was undetectable at the old threshold

A large, unmistakable truck occupying a quarter of the frame scores a maximum
of **0.19** with YOLO-World — below the old `--conf-equip 0.2` default. No
machinery was ever detected, so `proximity` alerts could not fire at all.

The floor is now 0.10. That admits sporadic hallucinations (a phantom
"excavator" on the concrete-pour clip), so confirmation was tightened from
2-of-4 samples to **3-of-4**: the phantom drops from 7 confirmed frames to 1,
while the real truck holds at 30.

---

## Known limitations

Stated plainly, because they matter for anyone considering this for real use.

**Helmet recall falls off with distance.** On the wider truck clip, helmets are
detected on ~70% of person-frames although essentially every worker wears one.
The sustained-violation grace absorbs most of that, but a worker missed
continuously for longer than the grace period will produce a false `no_helmet`
alert. One such false incident appears in the truck clip.

**Vest detection is weak at small scales.** At ~100 px worker height the model's
vest score barely separates wearers from non-wearers (median 0.122 vs 0.073).
Fusing model score with colour reaches only ~0.69 accuracy against a 0.66
"assume everyone wears one" baseline. The current settings are tuned to *avoid
false clears* rather than to maximise raw accuracy — it is better to raise a
reviewable alert than to silently certify a violation as compliant.

**`gloves` and `boots` are advisory only.** They are reported in
`events.json` and on the video, but no alert is derived from them and their
detection rates (3–38% depending on clip) are not trustworthy. Do not build a
compliance check on them.

**No person-to-PPE association.** The system reports "this worker has a helmet"
by running the model on that worker's crop, which is reliable for helmets but
means a helmet held in the hand, or one belonging to an overlapping worker in
the same crop, can be credited to the wrong person.

**Monocular distance is approximate.** Scale is inferred from median person
height assuming 1.7 m, on a flat-ground assumption. Proximity distances are
indicative, not survey-grade, and degrade with camera tilt and perspective.

**Single-site, offline.** No multi-stream orchestration beyond the two-camera
mode, no alerting integrations, no persistence beyond `events.json` and the
worker registry.

**Validation set is small.** 29 hand-labelled crops from 3 clips. Numbers are
directional. A real deployment needs a labelled set from its own cameras.

---

## Tests

```bash
pytest test_rules.py test_activity.py test_pipeline.py   # 43 tests, offline, ~2 s
pytest test_models.py                                    # 4 tests, needs weights + sample video
```

The offline suite needs no models and no video. It covers the geometry rules,
the activity classifier, track merging, incident consolidation, and includes
regression tests pinning the three detection bugs described above so they
cannot silently return.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ultralytics'`** — the virtualenv is not
active, or dependencies are not installed.

**Ultralytics pip-installs `onnx` mid-run** — should not happen; `onnx` is
pinned in `requirements.txt` precisely because the ReID encoder needs it.
Reinstall requirements if you see it.

**`cannot open source ...`** — bad path, or macOS camera permission. Grant
camera access under System Settings → Privacy & Security → Camera.

**No PPE alerts at all** — check `stats.ppe_grace_s` in `events.json`. The
requested grace is auto-capped to 25% of clip length, but a 120 s default on a
long video still means a worker must be bare-headed for two solid minutes. Use
`--ppe-grace 0` to alert immediately.

**Everything reports as compliant** — that was the bug fixed in this revision.
Verify `HIVIS_MIN_FRAC` is 0.45 and `PPE_OFF_S` is set, not the old
`PPE_MISS_LIMIT`.

**`GMC failed, falling back to identity`** — BoT-SORT global motion
compensation with OpenCV 5. Harmless in normal runs; it appears when frames of
differing sizes are fed to one tracker.

**Video will not play** — output is re-encoded to H.264. If the re-encode is
skipped, `imageio-ffmpeg` is missing.

---

## Project layout

```
pipeline.py               core engine: detect -> track -> rules -> incidents/clips
report.py                 events.json -> one-page PDF safety report
dashboard/app.py          Streamlit console (live view, run history, result tabs)
benchmark_tracking.py     model x tracker benchmark harness
vision/
  detectors.py            model wrappers, per-class PPE thresholds, path anchoring
  safety_rules.py         geometry: PPE match, fall, zones, proximity, hi-vis
  activity.py             working / idle / walking / sitting / fallen classifier
  alerting.py             alert labels + tunable severity
  gate.py                 entry-gate PPE check mode
  registry.py             persistent W-xx worker identities
  shifts.py               clock-time shift windows and per-shift summaries
config/*.yaml             tracker configs (BoT-SORT + ReID is the active one)
test_*.py                 pytest suite
```

---

## Roadmap

1. **Retrain the PPE stage on site footage.** The open-vocabulary model is the
   accuracy ceiling here; a small fine-tune on real camera angles would move
   vest detection further than any threshold change can.
2. **Tiled inference** for wide, elevated shots, to recover small-object recall.
3. **Person-to-PPE spatial association**, so an overlapping worker's helmet
   cannot be credited to the wrong person.
4. **Homography-based ground plane** to replace the median-height scale estimate.
5. **Alerting integrations** — webhook / Slack / email on sustained violations.
6. **Model export** (ONNX / CoreML / TensorRT) for on-site edge deployment.

---

## Licence

Code in this repository is MIT-licensed. Note that the Ultralytics models it
depends on (YOLO26, YOLOE, YOLO-World via the `ultralytics` package) carry
**AGPL-3.0**; commercial use without releasing your source requires an
[Ultralytics Enterprise licence](https://www.ultralytics.com/license).

---

## Safety notice

This is a demonstration and research system. Its recall limitations are
documented above and are significant. **Do not rely on it as the sole control
for worker safety.** Automated PPE monitoring should supplement human
supervision and site safety procedures, never replace them.
