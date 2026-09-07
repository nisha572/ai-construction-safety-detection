"""Construction AI live safety demo — live detection, incidents, clips + PDF report."""
import base64
import io
import json
import math
import shutil
import struct
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_ZONE_PICKER = True
except Exception:
    streamlit_image_coordinates = None
    HAS_ZONE_PICKER = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "output" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _import_legacy_run():
    """One-time: move the old fixed output/ files into the run-history store."""
    legacy_json = ROOT / "output" / "events.json"
    if not legacy_json.exists():
        return
    try:
        d = json.loads(legacy_json.read_text())
        date = (d.get("stats") or {}).get("date") or "legacy"
        rid = date.replace("-", "").replace(":", "").replace(" ", "_") + "_imported"
        rd = RUNS_DIR / rid
        rd.mkdir(parents=True, exist_ok=True)
        clips_src = ROOT / "output" / "clips"
        clips_dst = rd / "clips"
        moved_clips = clips_src.exists() and not clips_dst.exists()
        if moved_clips:
            shutil.move(str(clips_src), str(clips_dst))
        # clip paths inside events.json are absolute — repoint them
        for i in d.get("incidents", []):
            c = i.get("clip")
            if c:
                i["clip"] = str(rd / "clips" / Path(c).name)
        (rd / "events.json").write_text(json.dumps(d, indent=2))
        legacy_json.unlink()
        v = ROOT / "output" / "demo_annotated.mp4"
        if v.exists():
            shutil.move(str(v), str(rd / "annotated.mp4"))
    except Exception:
        pass


_import_legacy_run()


@st.cache_data(show_spinner=False)
def list_runs():
    """All saved runs, newest first, with display metadata."""
    runs = []
    if RUNS_DIR.exists():
        for rd in sorted(RUNS_DIR.iterdir(), reverse=True):
            j = rd / "events.json"
            if not j.exists():
                continue
            try:
                d = json.loads(j.read_text())
            except Exception:
                continue
            s = d.get("stats", {})
            runs.append({"id": rd.name, "dir": str(rd),
                         "date": s.get("date", rd.name),
                         "video": Path(d.get("video", "?")).stem,
                         "incidents": len(d.get("incidents", [])),
                         "workers": s.get("workers_seen", 0)})
    return runs

st.set_page_config(page_title="Construction AI · Site Safety",
                   page_icon=":material/construction:", layout="wide")

ALERT_NAMES = {"no_helmet": "No helmet", "no_vest": "No vest",
               "possible_fall": "Possible fall", "danger_zone": "Danger zone",
               "proximity": "Unsafe proximity"}
ALERT_ICONS = {"no_helmet": ":material/sports_motorsports:",
               "no_vest": ":material/health_and_safety:",
               "possible_fall": ":material/emergency:",
               "danger_zone": ":material/warning:",
               "proximity": ":material/construction:"}
SEV_BADGE = {"no_helmet": "orange", "no_vest": "yellow", "possible_fall": "red",
             "danger_zone": "red", "proximity": "red"}
POSSIBLE_FALL, DANGER_ZONE, PROXIMITY = "possible_fall", "danger_zone", "proximity"
PPE_GRACE_OPTS = {"Immediate": 0.0, "30 s": 30.0, "1 min": 60.0, "2 min": 120.0}
DEFAULT_SHIFTS_UI = [{"name": "Morning", "start": "06:00", "end": "14:00"},
                     {"name": "Afternoon", "start": "14:00", "end": "22:00"},
                     {"name": "Night", "start": "22:00", "end": "06:00"}]

ACTIVITY_ORDER = ["working", "walking", "sitting", "standing", "idle", "fallen"]
ACTIVITY_NAMES = {"working": "Working", "walking": "Walking", "sitting": "Sitting",
                  "standing": "Standing", "idle": "Idle", "fallen": "Fallen",
                  "unknown": "Unknown"}
ACTIVITY_C = {"working": "#2ed573", "walking": "#22d3ee", "sitting": "#1e90ff",
              "standing": "#feca57", "idle": "#8fa3bd", "fallen": "#ff4757",
              "unknown": "#57606f"}


@st.cache_data
def alert_chime():
    """Short two-tone notification ding (WAV bytes), played on new incidents."""
    sr, dur = 44100, 0.35
    frames = []
    for i in range(int(sr * dur)):
        t = i / sr
        env = min(1.0, t / 0.01) * math.exp(-7 * t)
        v = 0.55 * math.sin(2 * math.pi * 880 * t) + \
            0.35 * math.sin(2 * math.pi * 1318.5 * t)
        frames.append(env * v)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(32767 * max(-1.0, min(1.0, f))))
            for f in frames))
    return buf.getvalue()


def tick(ok_):
    return ":green[✔]" if ok_ else ":red[✘]"


POSE_C = {"standing": "#2ed573", "sitting": "#1e90ff",
          "possible_fall": "#ff4757", "unknown": "#8fa3bd"}


def _chip(label, color):
    return (f'<span class="chip2" style="background:{color}22;'
            f'color:{color};border:1px solid {color}44">{label}</span>')


def person_line(r):
    ppe = f"{tick(r['helmet'])} helmet · {tick(r['vest'])} vest"
    if r["gloves"]:
        ppe += " · gloves"
    if r["boots"]:
        ppe += " · boots"
    act = ACTIVITY_NAMES.get(r.get("activity"), "")
    head = f"**P{r['person']}**"
    if r.get("worker_id"):
        head += f" ({r['worker_id']})"
    head += f" — {r['pose']}"
    if r.get("zone"):
        head += f" · {r['zone']}"
    if act and act != "Unknown":
        head += f" · {act}"
    lines = [head, ppe]
    extra = []
    if r["alerts"]:
        extra.append(" · ".join(ALERT_NAMES.get(a, a) for a in r["alerts"]))
    zns = r.get("zones")
    if isinstance(zns, list) and zns:
        extra.append("inside " + ", ".join(zns))
    elif r.get("in_zone"):
        extra.append("inside danger zone")
    nm = r.get("nearest_m")
    if nm is not None and nm < 8:
        extra.append(f"{r.get('nearest_eq')} at {nm} m")
    if extra:
        lines.append(":material/notification_important: " + " · ".join(extra))
    return "  \n".join(lines)


def worker_cards(live_hist, limit=12, zone=None):
    """Per-worker status cards for the live feed. With zone=, only that
    camera zone's workers are shown (two-zone runs). Alert chips use the
    worker's accumulated alerts so past violations stay visible."""
    cards = []
    for pid in sorted(live_hist):
        h = live_hist[pid]
        r = h["last"]
        if zone is not None and r.get("zone") != zone:
            continue
        alerts = sorted(h.get("alerts") or set())
        cls = "alert" if alerts else "ok"
        row1 = [_chip(r["pose"], POSE_C.get(r["pose"], "#8fa3bd"))]
        act = r.get("activity")
        if act and act in ACTIVITY_NAMES:
            row1.insert(0, _chip(ACTIVITY_NAMES[act], ACTIVITY_C[act]))
        if r.get("zone"):
            row1.append(_chip(r["zone"].upper(), "#67e8f9"))
        row1 += [_chip(ALERT_NAMES.get(a, a).upper(), "#ff4757") for a in alerts]
        row2 = [_chip(("Helmet " if r["helmet"] else "No helmet ") + ("✓" if r["helmet"] else "✗"),
                      "#2ed573" if r["helmet"] else "#ff4757"),
                _chip(("Vest " if r["vest"] else "No vest ") + ("✓" if r["vest"] else "✗"),
                      "#2ed573" if r["vest"] else "#ff4757")]
        if r["gloves"]:
            row2.append(_chip("Gloves", "#67e8f9"))
        if r["boots"]:
            row2.append(_chip("Boots", "#67e8f9"))
        zns = r.get("zones")
        if isinstance(zns, list) and zns:
            row2 += [_chip(zn, "#e05fd8" if zn == "Zone 1" else "#ff9f43")
                     for zn in zns]
        elif r["in_zone"]:
            row2.append(_chip("Danger zone", "#e05fd8"))
        nm = r.get("nearest_m")
        if nm is not None and nm < 8:
            row2.append(_chip(f"{r.get('nearest_eq') or 'equip'} {nm} m", "#ff9f43"))
        cards.append(
            f'<div class="wcard {cls}"><div class="wnum">P{pid}</div>'
            f'<div style="flex:1;min-width:0"><div class="wrow1">'
            + "".join(row1) + "</div>"
            f'<div class="wrow2">{"".join(row2)}'
            f'<span class="seen">seen {h["frames"]}f</span></div></div></div>')
        if len(cards) >= limit:
            break
    return ("".join(cards) or '<div style="color:#8fa3bd;padding:10px 4px;'
            'font-size:.85rem">No workers detected yet…</div>')


st.markdown("""
<style>
    .hero{background:linear-gradient(115deg,#0b1220 0%,#13233c 55%,#0e3a4d 100%);
        border-radius:18px;padding:24px 30px;margin-bottom:10px;
        border:1px solid #22304a;border-left:4px solid #22d3ee;
        display:flex;align-items:center;gap:26px}
    .hero h1{color:#f1f5f9;font-size:1.8rem;margin:0 0 6px;letter-spacing:.3px}
    .hero h1 span{color:#22d3ee}
    .hero p{color:#94a8c3;margin:0;font-size:.95rem}
    .hero-art{flex:0 0 auto;position:relative;width:92px;height:92px}
    .hero-art svg{position:absolute;inset:0}
    .ring{animation:ring 2.6s ease-out infinite}
    .ring2{animation:ring 2.6s ease-out .9s infinite}
    @keyframes ring{0%{opacity:.75;transform:scale(.55)}75%{opacity:0;
        transform:scale(1.15)}100%{opacity:0;transform:scale(1.15)}}
    .helmet{animation:bob 3.2s ease-in-out infinite}
    @keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
    .tl{overflow-x:auto;padding:6px 0 2px;white-space:nowrap}
    .tl .cell{display:inline-block;width:9px;height:22px;margin:0 .5px;
        border-radius:2px;vertical-align:middle}
    .live-dot{width:10px;height:10px;border-radius:50%;background:#ff4757;
        display:inline-block;margin-right:9px;vertical-align:middle;
        animation:pulse 1.3s infinite}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,71,87,.65)}
        70%{box-shadow:0 0 0 9px rgba(255,71,87,0)}
        100%{box-shadow:0 0 0 0 rgba(255,71,87,0)}}
    .wcard{display:flex;gap:12px;align-items:center;background:#111a2c;
        border:1px solid #253248;border-radius:10px;padding:8px 12px;
        margin:0 0 6px 0}
    .wcard.ok{border-left:3px solid #2ed573}
    .wcard.alert{border-left:3px solid #ff4757}
    .wnum{font-weight:800;color:#67e8f9;font-size:1.02rem;min-width:36px;
        text-align:center}
    .wrow1{display:flex;flex-wrap:wrap;align-items:center;gap:5px}
    .wrow2{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-top:4px}
    .chip2{font-size:.7rem;padding:1.5px 9px;border-radius:10px;
        font-weight:600;letter-spacing:.2px;display:inline-block}
    .seen{color:#8fa3bd;font-size:.7rem;margin-left:auto}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-art">
    <svg class="ring" viewBox="0 0 100 100"><circle cx="50" cy="50" r="44"
      fill="none" stroke="#22d3ee" stroke-width="2" opacity=".5"/></svg>
    <svg class="ring2" viewBox="0 0 100 100"><circle cx="50" cy="50" r="44"
      fill="none" stroke="#22d3ee" stroke-width="2" opacity=".35"/></svg>
    <svg class="helmet" viewBox="0 0 100 100">
      <defs><linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#67e8f9"/><stop offset="1"
          stop-color="#0e7490"/></linearGradient></defs>
      <circle cx="50" cy="50" r="46" fill="#0b1220" stroke="#22d3ee"
        stroke-width="1.5"/>
      <path d="M25 62 a25 22 0 0 1 50 0 Z" fill="url(#hg)"/>
      <rect x="20" y="62" width="60" height="7" rx="3.5" fill="#67e8f9"/>
      <rect x="46" y="38" width="8" height="12" rx="3" fill="#f1f5f9"/>
    </svg>
  </div>
  <div>
    <h1>Construction AI <span>·</span> Site Safety Console</h1>
    <p>Worker tracking &middot; PPE verification &middot; equipment monitoring &middot; fall
       detection — grouped into incidents with video clips and a PDF report.</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.container(horizontal=True):
    st.badge("Pretrained models only", icon=":material/psychology:", color="violet")
    st.badge("YOLO26s + BoT-SORT ReID", icon=":material/person_search:", color="blue")
    st.badge("Crop-based PPE", icon=":material/content_cut:", color="primary")
    st.badge("Temporal fall check", icon=":material/schedule:", color="green")

VIDEO_DIR = ROOT / "input" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Upload path. input/videos/ can legitimately be empty -- the sample clips are
# not committed (licence-restricted stock, or real site recordings), so without
# this the picker had nothing to list and the page stopped dead.
with st.sidebar:
    up = st.file_uploader("Add a video", type=["mp4", "mov", "avi"],
                          accept_multiple_files=True,
                          help="Uploaded clips are saved to input/videos/ "
                               "and appear in the pickers below.")
    for f in up or []:
        dest = VIDEO_DIR / Path(f.name).name
        if not dest.exists():
            dest.write_bytes(f.getbuffer())
            st.success(f"Added {dest.name}")

videos = sorted([p for ext in ("*.mp4", "*.mov", "*.avi")
                 for p in VIDEO_DIR.glob(ext)],
                key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
if not videos:
    st.info("Upload a video in the sidebar to get started, or drop one into "
            "`input/videos/`.")
    st.stop()


@st.cache_data(show_spinner=False)
def video_info(path):
    """Probe duration / resolution / fps for a nicer picker label."""
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    res = f"{int(h)}p" if h in (360, 480, 540, 720, 1080) else f"{int(w)}×{int(h)}"
    return f"{n / fps:.0f}s · {res} · {fps:.0f} fps"


def video_label(path):
    return f"{Path(path).stem}  ·  {video_info(str(path))}"


@st.cache_data(show_spinner=False)
def presence_heatmap(video_path, source_path, rows, person=None, zones=None):
    """Worker presence heatmap over a clean frame of the site.

    Accumulates every tracked person-box (lower body weighted) from
    person_frames into a coarse density grid, blurs it and blends a TURBO
    colormap over a background frame from the source video (resized to the
    processed resolution the boxes were recorded in)."""
    cap = cv2.VideoCapture(video_path)
    Wp = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    Hp = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    cap.release()
    bg = None
    if source_path and Path(source_path).exists():
        try:
            cap = cv2.VideoCapture(source_path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 3))
            ok, f = cap.read()
            cap.release()
            if ok:
                bg = cv2.resize(f, (Wp, Hp), interpolation=cv2.INTER_AREA)
        except Exception:
            bg = None
    if bg is None:
        cap = cv2.VideoCapture(video_path)
        ok, f = cap.read()
        cap.release()
        bg = f if ok else np.zeros((Hp, Wp, 3), np.uint8)

    cell = 4
    heat = np.zeros((Hp // cell + 1, Wp // cell + 1), np.float32)
    for r in rows:
        if person is not None and r.get("person") != person:
            continue
        x1, y1, x2, y2 = r.get("bbox", (0, 0, 0, 0))
        gx1, gx2 = int(x1 / cell), max(int(x1 / cell) + 1, int(x2 / cell))
        gy1 = int((y1 + 0.4 * (y2 - y1)) / cell)
        gy2 = max(gy1 + 1, int(y2 / cell))
        heat[gy1:gy2, gx1:gx2] += 1.0

    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), 2.0)
        heat = heat / heat.max()
        heat = cv2.resize(heat, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        hm = cv2.applyColorMap((255 * heat ** 0.6).astype(np.uint8),
                               cv2.COLORMAP_TURBO)
        out = cv2.addWeighted(hm, 0.55, bg, 0.45, 0)
    else:
        out = bg
    for z in (zones or []):
        poly = z.get("poly") if isinstance(z, dict) else z
        name = z.get("name") if isinstance(z, dict) else None
        if poly and len(poly) >= 3:
            col = ZONE_BGR.get(name, (255, 0, 255))
            cv2.polylines(out, [np.array(poly, np.int32)], True, col, 2)
            if name:
                p0 = np.array(poly, np.int32)[0]
                cv2.putText(out, name.upper(), (int(p0[0]), int(p0[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

STRIDE = {"Fast": 3, "Balanced": 2, "Best": 1}
WEBCAM_FRAMES = 400


@st.cache_resource(show_spinner="Loading pretrained models (YOLO26)…")
def load_models():
    from vision.detectors import Detectors
    return Detectors()


with st.sidebar:
    st.header("Controls")
    webcam = st.checkbox("Use webcam instead", help="Live camera, ~60 s capture")
    single_mode = not webcam and st.checkbox(
        "Single video mode",
        help="Optional: analyse one video. Leave OFF for the default "
             "Zone 1 + Zone 2 two-camera view.")
    two_zone = not webcam and not single_mode
    if not webcam:
        if two_zone:
            st.caption("**Zone 1 + Zone 2** — two cameras on one screen, "
                       "analysed simultaneously")
            _z1d = next((i for i, v in enumerate(videos)
                         if v.name == "construction.mp4"), 0)
            zone1_sel = st.selectbox("Zone 1 video", videos, index=_z1d,
                                     format_func=video_label, key="z1_sel")
            _others = [v for v in videos if v.name != zone1_sel.name] or videos
            _z2d = next((i for i, v in enumerate(_others)
                         if v.name == "MicrosoftTeams-video.mp4"), 0)
            zone2_sel = st.selectbox("Zone 2 video", _others, index=_z2d,
                                     format_func=video_label, key="z2_sel")
        else:
            _def = next((i for i, v in enumerate(videos)
                         if v.name == "construction.mp4"), 0)
            source_sel = st.selectbox("Video", videos, index=_def,
                                      format_func=video_label)
            speed = st.segmented_control("Analysis speed", ["Fast", "Balanced", "Best"],
                                         default="Fast")
            st.caption("Fast ≈ 2 min on 45s footage · Best = every frame")
    grace_lbl = st.select_slider("PPE alert grace period",
                                 options=list(PPE_GRACE_OPTS), value="2 min",
                                 help="Helmet/vest alerts fire only after a worker "
                                      "has been continuously without PPE this long")
    st.caption("On video files the grace is auto-capped at 25% of the clip "
               "length, so short demos still alert. Danger-zone alerts "
               "confirm after ~1 s continuously inside the zone.")

    with st.expander("Detection options", icon=":material/tune:"):
        low_light = st.toggle("Low-light enhancement", value=False,
                              help="CLAHE contrast boost on frames fed to the "
                                   "models (annotations stay on the original "
                                   "frame) — for dim footage/webcam")
        ppe_zone_only = st.toggle("PPE alerts inside zones only", value=False,
                                  help="Helmet/vest rules apply only to "
                                       "workers inside a defined zone — "
                                       "passersby outside zones are ignored")
        gate_mode = st.toggle("Gate check mode", value=False,
                              help="Check PPE once per worker on first "
                                   "stable sighting (entry/exit gate) "
                                   "instead of alerting continuously")
    with st.expander("Alert severity", icon=":material/priority_high:"):
        from vision.alerting import DEFAULT_SEVERITY, SEV_LEVELS  # noqa: E402

        severity_cfg = {}
        for _t, _lbl in ALERT_NAMES.items():
            severity_cfg[_t] = st.selectbox(
                _lbl, SEV_LEVELS, index=SEV_LEVELS.index(
                    DEFAULT_SEVERITY[_t]), key=f"sev_{_t}")
    with st.expander("Shifts", icon=":material/schedule:",
                     expanded=False):
        st.caption("Split the session into clock-time shifts (anchored on "
                   "the run's start time) for per-shift summaries.")
        shift_cfg = []
        for _i, _s in enumerate(DEFAULT_SHIFTS_UI):
            _c1, _c2, _c3 = st.columns([2, 1, 1])
            _name = _c1.text_input("Shift", _s["name"],
                                   key=f"shift_name_{_i}", label_visibility="collapsed")
            _start = _c2.text_input("Start", _s["start"], key=f"shift_a_{_i}",
                                    label_visibility="collapsed")
            _end = _c3.text_input("End", _s["end"], key=f"shift_b_{_i}",
                                  label_visibility="collapsed")
            shift_cfg.append({"name": _name, "start": _start, "end": _end})
    _reg_path = ROOT / "output" / "worker_registry.json"
    with st.expander("Worker registry", icon=":material/badge:"):
        _n_reg = 0
        if _reg_path.exists():
            try:
                _n_reg = len(json.loads(_reg_path.read_text())
                             .get("workers", {}))
            except Exception:
                pass
        st.caption(f"{_n_reg} known worker identity "
                   f"{'identities' if _n_reg != 1 else 'identity'} — workers "
                   "keep the same W-ID across runs and cameras (matched by "
                   "appearance).")
        if st.button("Reset worker registry", icon=":material/delete:",
                     disabled=not _reg_path.exists()):
            _reg_path.unlink(missing_ok=True)
            st.toast("Worker registry cleared — the next run starts "
                     "registering W-IDs from scratch.", icon=":material/delete:")

    start = st.button("Start live detection", icon=":material/play_arrow:",
                      type="primary", width="stretch")
    with st.expander("Pipeline detail", icon=":material/account_tree:"):
        st.markdown("""
        - **Tracking** — YOLO26s + BoT-SORT ReID, per-frame stable worker IDs
        - **PPE** — helmet / vest / gloves / boots on padded person crops
        - **Equipment** — YOLO-World: excavator / dump truck / wheel loader
          (confirmed over consecutive samples to cut false hits)
        - **Pose** — standing / sitting / fallen, temporally confirmed
        - **Activity** — working / idle / walking from pose + multi-frame motion
        - **Geometry** — two danger zones (Zone 1 + Zone 2) + per-machine
          proximity radius (crane > excavator > pickup)
        - **Worker registry** — persistent W-IDs across runs and cameras
          (appearance-matched, resettable in the sidebar)
        - **Extras** — gate-check mode (PPE at entry), zone-only PPE rules,
          low-light enhancement, shift summaries, tunable alert severity
        - **Output** — incidents grouped, clipped, reported
        """)
    runs = list_runs()
    if runs:
        st.header("Run history")
        _labels = {r["id"]: f"{r['date'][:16]} · {r['video']}"
                   for r in runs}
        _ids = [r["id"] for r in runs]
        # a pending jump (new run finished / selected run deleted) is applied
        # here, BEFORE the selectbox is instantiated — the only legal place
        _jump = st.session_state.pop("run_jump", None)
        if _jump and _jump in _ids:
            st.session_state.run_sel = _jump
        if st.session_state.get("run_sel") not in _ids:
            st.session_state.run_sel = _ids[0]
        st.selectbox("Viewing run", _ids, key="run_sel",
                     format_func=lambda i: _labels.get(i, i))
        with st.expander(f"Manage runs ({len(runs)})"):
            for r in runs:
                cc1, cc2 = st.columns([4, 1])
                cc1.caption(f"{r['date'][:16]} · {r['incidents']} incidents"
                            f" · {r['workers']} workers")
                if cc2.button(":material/delete:", key=f"del_{r['id']}",
                              help="Delete this run and its clips"):
                    shutil.rmtree(r["dir"], ignore_errors=True)
                    list_runs.clear()
                    if st.session_state.get("run_sel") == r["id"]:
                        _left = [x["id"] for x in list_runs()]
                        st.session_state.run_jump = _left[0] if _left else None
                    st.rerun()
    try:
        from pipeline import PIPELINE_VERSION as _pv
    except Exception:   # stale pipeline module cached by an old server
        _pv = "STALE — fully stop this server (Ctrl+C) and start it again"
    st.caption(f"Construction AI demo · v2.2 · pipeline {_pv}"
               + ("" if _pv.startswith("STALE") else
                  " (restart the server if this doesn't match after code changes)"))

# ---------------- danger-zone editor (single video: one polygon) ----------
ZONE_BGR = {"Zone 1": (255, 0, 255), "Zone 2": (0, 165, 255),
            "Danger Zone": (0, 165, 255)}
if not webcam and not two_zone:
    with st.expander("Danger zone (optional) — click the frame to draw it",
                     icon=":material/format_shapes:"):
        if not HAS_ZONE_PICKER:
            st.info("Install `streamlit-image-coordinates` to draw a custom "
                    "danger zone. The default zone is used for now.")
        else:
            if st.session_state.get("zone_video") != str(source_sel):
                st.session_state.zone_pts = []
                st.session_state.zone_video = str(source_sel)
            pts = st.session_state.get("zone_pts", [])

            @st.cache_data(show_spinner=False)
            def zone_preview(path):
                cap = cv2.VideoCapture(path)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 4))
                ok, fr = cap.read()
                cap.release()
                return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None

            prev = zone_preview(str(source_sel))
            if prev is None:
                st.warning("Could not read a preview frame for this video.")
            else:
                disp = prev.copy()
                col = ZONE_BGR["Danger Zone"]
                if len(pts) >= 3:
                    poly = np.array(pts, dtype=np.int32)
                    overlay = disp.copy()
                    cv2.fillPoly(overlay, [poly], col)
                    cv2.addWeighted(overlay, 0.18, disp, 0.82, 0, disp)
                    cv2.polylines(disp, [poly], True, col, 2)
                    cv2.putText(disp, "DANGER ZONE", (int(poly[0][0]),
                                int(poly[0][1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                for px, py in pts:
                    cv2.circle(disp, (int(px), int(py)), 5, col, -1)
                cL, cR = st.columns([3, 1], vertical_alignment="top",
                                    gap="medium")
                with cL:
                    click = streamlit_image_coordinates(disp, key="zone_click")
                with cR:
                    st.metric("Points", len(pts), border=True)
                    b1, b2 = st.columns(2)
                    if b1.button("Undo", icon=":material/undo:",
                                 width="stretch"):
                        if pts:
                            pts.pop()
                        st.session_state.zone_last = None
                        st.rerun()
                    if b2.button("Clear", icon=":material/delete:",
                                 width="stretch"):
                        pts.clear()
                        st.session_state.zone_last = None
                        st.rerun()
                    if len(pts) >= 3:
                        st.success(f"Custom zone active · {len(pts)} points")
                    else:
                        st.caption("Default zone will be used — click 3+ "
                                   "points on the frame.")
                if click:
                    xy = (int(round(click["x"])), int(round(click["y"])))
                    if xy != st.session_state.get("zone_last"):
                        pts.append(xy)
                        st.session_state.zone_last = xy
                        st.rerun()

# ---------------- live run ----------------
if start:
    from pipeline import run

    ppe_grace = PPE_GRACE_OPTS[grace_lbl]

    # every run gets its own folder: events.json + annotated video(s) + clips/
    if two_zone:
        _stem = f"2zone_{zone1_sel.stem}_{zone2_sel.stem}"
        stride = 3                      # Fast, per camera
    elif webcam:
        _stem = "webcam"
        stride = 2
    else:
        _stem = source_sel.stem
        stride = STRIDE[speed]
    rid = time.strftime("%Y%m%d_%H%M%S") + "_" + _stem
    rid = "".join(c if (c.isalnum() or c in "_-") else "_" for c in rid)
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    if two_zone:
        # ---------- two cameras, one screen: Zone 1 | Zone 2 ----------
        from pipeline import run_multi

        def _nf(p):
            c = cv2.VideoCapture(p)
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            c.release()
            return max(n, 1)

        sources = [("Zone 1", str(zone1_sel)), ("Zone 2", str(zone2_sel))]
        nf_z = [_nf(s[1]) for s in sources]

        with st.container(border=True):
            st.markdown('<span class="live-dot"></span>'
                        '<b style="font-size:1.02rem">Two-zone live '
                        'detection</b>', unsafe_allow_html=True)
            status_ph = st.empty()
            bar = st.progress(0.0, text="Starting…")
            g1, g2 = st.columns(2, gap="small")
            with g1:
                st.markdown(f"**ZONE 1** · {zone1_sel.name}")
                zf1 = st.empty()
                z1_info = st.empty()
                st.markdown("**Zone 1 workers**")
                z1_feed = st.empty()
            with g2:
                st.markdown(f"**ZONE 2** · {zone2_sel.name}")
                zf2 = st.empty()
                z2_info = st.empty()
                st.markdown("**Zone 2 workers**")
                z2_feed = st.empty()
            st.markdown("**Combined — all workers, both zones**")
            m1, m2 = st.columns(2)
            w_ph, e_ph = m1.empty(), m2.empty()
            m3, m4 = st.columns(2)
            i_ph, f_ph = m3.empty(), m4.empty()
            feed_ph = st.empty()

        live_hist = {}
        _last_img = {}          # per-zone image-update throttle
        _last_hud = [0.0]
        _prog = {}
        notified_mult = {}      # pid -> highest 'every 2 incidents' multiple
        last_chime = [0.0]
        chime_n = [0]
        sound_ph = st.empty()
        chime_wav = alert_chime()

        def play_chime():
            chime_n[0] += 1
            b64 = base64.b64encode(chime_wav + b"\x00\x00" * chime_n[0]).decode()
            sound_ph.markdown(
                f'<audio autoplay src="data:audio/wav;base64,{b64}"></audio>',
                unsafe_allow_html=True)

        def on_zone_frame(zname, zi, ztot, annot, fi, n_p, n_e, counts,
                          n_inc, recs, pinc=None):
            now = time.time()
            if now - _last_img.get(zname, 0) > 0.12:
                _last_img[zname] = now
                ph = zf1 if zname == "Zone 1" else zf2
                ph.image(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB),
                         width="stretch")
            for r in recs:
                hh = live_hist.setdefault(r["person"],
                                          {"frames": 0, "alerts": set(),
                                           "last": r})
                hh["frames"] += 1
                hh["alerts"].update(r["alerts"] or [])
                hh["last"] = r
            _prog[zname] = fi / nf_z[zi - 1]
            # per-zone info line under each video
            alerts_now = sorted({a for r in recs for a in (r["alerts"] or [])})
            zinfo = (f"{n_p} workers · {n_e} equipment · "
                     f"{n_inc} incidents"
                     + (" · :material/notification_important: **ALERTS:** "
                        + ", ".join(ALERT_NAMES.get(a, a) for a in alerts_now)
                        if alerts_now else ""))
            (z1_info if zname == "Zone 1" else z2_info).markdown(zinfo)
            # alert notifications: toast + sound every 2 incidents per worker
            for pid, n2 in sorted((pinc or {}).items()):
                if n2 < 2:
                    continue
                mult = n2 // 2
                if notified_mult.get(pid, 0) >= mult:
                    continue
                notified_mult[pid] = mult
                hh = live_hist.get(pid)
                kinds = sorted(hh["alerts"]) if hh else []
                zw = (hh["last"].get("zone") if hh else None) or ""
                msg = f"Worker {pid}{(' · ' + zw) if zw else ''} · " \
                      f"{2 * mult} incidents"
                if kinds:
                    msg += " — " + " · ".join(ALERT_NAMES.get(a, a)
                                              for a in kinds)
                st.toast(msg, icon=":material/notification_important:",
                         duration=6)
                if time.time() - last_chime[0] >= 2.5:
                    last_chime[0] = time.time()
                    play_chime()
            if now - _last_hud[0] > 0.4:
                _last_hud[0] = now
                _p = [_prog.get(zn, 0.0) for zn, _ in sources]
                bar.progress(min(sum(_p) / len(_p), 1.0),
                             text=" · ".join(f"{zn} {p:.0%}"
                                             for (zn, _), p in
                                             zip(sources, _p)))
                status_ph.markdown(
                    f"<span style='color:#8fa3bd;font-size:.85rem'>"
                    + " · ".join(
                        f"<b style='color:#22d3ee'>{zn}</b>"
                        for (zn, _), p in zip(sources, _p) if p < 1.0)
                    + " running simultaneously</span>", unsafe_allow_html=True)
                w_ph.metric("Workers (last zone)", n_p,
                            icon=":material/groups:")
                e_ph.metric("Equipment (last zone)", n_e,
                            icon=":material/front_loader:")
                i_ph.metric("Incidents", n_inc,
                            icon=":material/notification_important:")
                f_ph.metric("Falls", counts.get("possible_fall", 0),
                            icon=":material/emergency:")
                z1_feed.markdown(worker_cards(live_hist, zone="Zone 1"),
                                 unsafe_allow_html=True)
                z2_feed.markdown(worker_cards(live_hist, zone="Zone 2"),
                                 unsafe_allow_html=True)
                feed_ph.markdown(worker_cards(live_hist, limit=18),
                                 unsafe_allow_html=True)

        with st.spinner("Analysing both zones…"):
            run_multi(sources, run_dir, stride=stride,
                      on_zone_frame=on_zone_frame, ppe_grace_s=ppe_grace,
                      ppe_zone_only=ppe_zone_only, low_light=low_light,
                      gate_mode=gate_mode, severity=severity_cfg,
                      shifts=shift_cfg)
        bar.progress(1.0, text="Done")
        st.session_state.run_jump = rid       # applied pre-widget on rerun
        list_runs.clear()
        st.rerun()

    if webcam:
        source, nframes = 0, WEBCAM_FRAMES
        st.warning("Camera starting — allow access if macOS asks.")
    else:
        source, nframes = str(source_sel), 0
        cap = cv2.VideoCapture(source)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    models = load_models()

    with st.container(border=True):
        hc1, hc2 = st.columns([1, 1], vertical_alignment="center",
                              gap="small")
        with hc1:
            st.markdown('<span class="live-dot"></span>'
                        '<b style="font-size:1.02rem">Live detection</b>',
                        unsafe_allow_html=True)
        with hc2.container(horizontal_alignment="right"):
            status_ph = st.empty()
        left, right = st.columns([3, 2], vertical_alignment="top")
        with left:
            frame_ph = st.empty()
            bar = st.progress(0.0, text="Starting…")
        with right:
            m1, m2 = st.columns(2)
            w_ph, e_ph = m1.empty(), m2.empty()
            m3, m4 = st.columns(2)
            i_ph, f_ph = m3.empty(), m4.empty()
            st.markdown("**Workers in this frame**")
            feed_ph = st.empty()

    live_hist = {}
    tick_clock = [0.0]
    last_chime = [0.0]
    chime_n = [0]
    notified_mult = {}      # pid -> highest 'every 2 incidents' multiple notified
    sound_ph = st.empty()
    chime_wav = alert_chime()

    def play_chime():
        """Raw <audio autoplay> with unique bytes per call.

        st.empty().audio(..., autoplay=True) is not reliably re-triggered by
        browsers when the element is swapped in place mid-run (autoplay only
        applies at element load); a freshly-mounted HTML audio element with a
        unique data URI plays every time."""
        chime_n[0] += 1
        b64 = base64.b64encode(chime_wav + b"\x00\x00" * chime_n[0]).decode()
        sound_ph.markdown(
            f'<audio autoplay src="data:audio/wav;base64,{b64}"></audio>',
            unsafe_allow_html=True)

    def on_frame(annot, fi, n_p, n_e, alerts, counts, n_inc, person_recs,
                 person_inc=None):
        frame_ph.image(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB), width="stretch")
        for r in person_recs:
            hh = live_hist.setdefault(r["person"], {"frames": 0, "alerts": set(),
                                                    "last": r})
            hh["frames"] += 1
            hh["alerts"].update(r["alerts"] or [])
            hh["last"] = r
        if time.time() - tick_clock[0] > 0.5:
            tick_clock[0] = time.time()
            n_working = sum(1 for r in person_recs
                            if r.get("activity") == "working")
            bar.progress(min(fi / max(nframes - 1, 1), 1.0),
                         text=f"Frame {fi} of {nframes}")
            status_ph.markdown(
                f"<span style='color:#8fa3bd;font-size:.85rem'>"
                f"<b style='color:#c3d0e2'>{n_p}</b> workers · "
                f"<b style='color:#2ed573'>{n_working}</b> working · "
                f"<b style='color:#c3d0e2'>{n_e}</b> equipment · "
                f"<b style='color:{'#ff6b81' if n_inc else '#2ed573'}'>{n_inc}</b>"
                f" incidents</span>", unsafe_allow_html=True)
            w_ph.metric("Workers", n_p, icon=":material/groups:")
            e_ph.metric("Equipment", n_e, icon=":material/front_loader:")
            i_ph.metric("Incidents", n_inc,
                        icon=":material/notification_important:")
            f_ph.metric("Falls", counts.get("possible_fall", 0),
                        icon=":material/emergency:")
            feed_ph.markdown(worker_cards(live_hist), unsafe_allow_html=True)
        # alert notification per worker: toast + sound every 2 incidents
        # (2, 4, 6 ...). person_inc = live per-worker incident counts.
        for pid, n in sorted((person_inc or {}).items()):
            if n < 2:
                continue
            mult = n // 2
            if notified_mult.get(pid, 0) >= mult:
                continue
            notified_mult[pid] = mult
            hh = live_hist.get(pid)
            kinds = sorted(hh["alerts"]) if hh else []
            msg = f"Worker {pid} · {2 * mult} incidents"
            if kinds:
                msg += " — " + " · ".join(ALERT_NAMES.get(a, a) for a in kinds)
            st.toast(msg, icon=":material/notification_important:", duration=6)
            if time.time() - last_chime[0] >= 2.5:
                last_chime[0] = time.time()
                play_chime()

    with st.spinner("Analysing…"):
        zones_arg = None
        if not webcam:
            _pts = st.session_state.get("zone_pts") or []
            zones_arg = [
                {"name": "Danger Zone",
                 "poly": [(int(x), int(y)) for x, y in _pts] if len(_pts) >= 3 else None},
            ]
        try:
            counts = run(source, str(run_dir / "annotated.mp4"),
                         str(run_dir / "events.json"), zones=zones_arg,
                         stride=stride,
                         max_frames=WEBCAM_FRAMES if webcam else None,
                         on_frame=on_frame, detectors=models,
                         ppe_grace_s=ppe_grace,
                         ppe_zone_only=ppe_zone_only, low_light=low_light,
                         gate_mode=gate_mode, severity=severity_cfg,
                         shifts=shift_cfg)
        except RuntimeError as e:
            st.error(f"{e}")
            st.stop()
    bar.progress(1.0, text="Done")
    st.session_state.run_jump = rid       # applied pre-widget on next rerun
    list_runs.clear()
    st.rerun()

# ---------------- results (for the selected run) ----------------
runs = list_runs()
if not runs:
    st.info("No runs yet — pick an input and press **Start live detection**.")
    st.stop()
_sel_id = st.session_state.get("run_sel") or runs[0]["id"]
RUN = next((r for r in runs if r["id"] == _sel_id), runs[0])
RUN_DIR = Path(RUN["dir"])
OUT_JSON = str(RUN_DIR / "events.json")
OUT_MP4 = str(RUN_DIR / "annotated.mp4")

if not Path(OUT_JSON).exists():
    st.info("This run has no events file — run a new analysis.")
    st.stop()

d = json.load(open(OUT_JSON))
MULTI = bool(d.get("multi_zone"))
ZSRC = d.get("zone_sources") or []
WZ = d.get("worker_zones") or {}

# person id -> persistent worker id ("W-03"), wherever it was recorded
WID = {}
for _r in d.get("person_frames") or []:
    if _r.get("worker_id"):
        WID[_r["person"]] = _r["worker_id"]
for _i in d.get("incidents") or []:
    if _i.get("worker_id"):
        WID[_i["person"]] = _i["worker_id"]
for _g in d.get("gate_checks") or []:
    if _g.get("worker_id"):
        WID[_g["person"]] = _g["worker_id"]


def wid_label(pid):
    w = WID.get(pid)
    return f"P{pid} · {w}" if w else f"P{pid}"


def clip_path(p):
    """Resolve a stored clip path: runs since v6 keep them relative to the
    run dir; legacy runs stored absolute paths."""
    if not p:
        return p
    pp = Path(p)
    return str(pp if pp.is_absolute() else RUN_DIR / pp)


def zvid(zname):
    """Path to a zone's annotated video (two-zone runs)."""
    for z in ZSRC:
        if z["name"] == zname:
            return str(RUN_DIR / z["annotated"])
    return OUT_MP4


def worker_zone_label(pid):
    zw = WZ.get(str(pid))
    parts = [wid_label(pid)]
    if zw:
        parts.append(zw)
    return " · ".join(parts)


inc = pd.DataFrame(d.get("incidents", []))
if inc.empty:
    inc = pd.DataFrame(columns=["type", "person", "zone", "equipment",
                                "dist_m", "start_frame", "end_frame",
                                "start_s", "end_s", "duration_s", "clip",
                                "worker_id"])
    if not d.get("gate_checks"):
        st.success("No incidents in the last run.")

inc["name"] = inc.apply(
    lambda r: f"{ALERT_ICONS.get(r['type'], ':material/warning:')} "
              f"{ALERT_NAMES.get(r['type'], r['type'])}"
              + (f" ({r['zone']})" if pd.notna(r.get("zone"))
                 and r.get("zone") else ""), axis=1)
inc["clip"] = inc["clip"].fillna("")

from report import compliance  # noqa: E402
from vision.alerting import severity_map  # noqa: E402

helm, vest, overall = compliance(d)
st_stats = d.get("stats", {})
SEV = severity_map(st_stats.get("severity"))
n_crit = int(inc["type"].map(lambda t: SEV.get(t) == "Critical").sum()) \
    if len(inc) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Incidents", len(inc), icon=":material/notification_important:",
          border=True)
k2.metric("Critical", n_crit, icon=":material/priority_high:", border=True)

_pf = pd.DataFrame(d.get("person_frames", []))
_spark = []
if not _pf.empty:
    _ser = _pf.groupby("pidx").size().sort_index()
    _step = max(1, len(_ser) // 300)
    _spark = _ser.iloc[::_step].tolist()
k3.metric("Workers in frame",
          st_stats.get("max_concurrent", st_stats.get("workers_seen", 0)),
          icon=":material/groups:", border=True,
          chart_data=_spark or None, chart_type="line")
k4.metric("PPE compliance", f"{overall}%",
          icon=":material/verified_user:", border=True)
k5.metric("Frames analysed", d["total_frames"],
          icon=":material/movie:", border=True)

with st.container(horizontal=True):
    for t, n in inc["type"].value_counts().items():
        st.badge(f"{ALERT_NAMES.get(t, t)} · {n}",
                 icon=ALERT_ICONS.get(t, ":material/warning:"),
                 color=SEV_BADGE.get(t, "gray"))

cA, cB = st.columns(2, gap="large")
with cA.container(border=True):
    st.markdown("**Alerts over time** · per second of footage")
    _ev = pd.DataFrame(d.get("events", []))
    if not _ev.empty:
        _ev = _ev[_ev["type"] != "gate_check"]     # internal bookkeeping rows
    if not _ev.empty:
        _ev["sec"] = _ev["time_s"].astype(int)
        _piv = _ev.pivot_table(index="sec", columns="type", values="frame",
                               aggfunc="count").fillna(0)
        st.area_chart(_piv)
    else:
        st.caption("No alert events recorded.")
with cB.container(border=True):
    st.markdown("**Incidents by type**")
    _tb = inc["type"].value_counts().reset_index(name="count")
    _tb["type"] = _tb["type"].map(lambda k: ALERT_NAMES.get(k, k))
    st.bar_chart(_tb, x="type", y="count")

tab1, tab2, tab5, tab3, tab4, tab6, tab7 = st.tabs([
    ":material/description: Session report",
    ":material/movie: Alert clips",
    ":material/person: Person-wise",
    ":material/list_alt: All incidents",
    ":material/groups: By worker",
    ":material/directions_run: Activity",
    ":material/local_fire_department: Heatmap",
])

with tab1:
    st.subheader("Session report")
    dur = d["total_frames"] * max(st_stats.get("stride", 1), 1) / max(
        st_stats.get("video_fps", 25), 1)
    eq = ", ".join(f"{k} ({v} frames)"
                    for k, v in st_stats.get("equipment_seen", {}).items()) or "none"
    _act_tot = (d.get("activity") or {}).get("totals_s", {})
    _act_trk = sum(_act_tot.values())

    _zrows = []
    if MULTI:
        for z in ZSRC:
            _zrows.extend([
                (f"{z['name']} (video)", Path(z.get("video", "?")).stem),
                (f"{z['name']} workers", str(z.get("workers", 0))),
                (f"{z['name']} incidents", str(z.get("incidents", 0))),
                (f"{z['name']} PPE compliance",
                 f"helmet {z.get('helmet_pct', 0)}% · "
                 f"vest {z.get('vest_pct', 0)}%"),
            ])
    else:
        _pfz = pd.DataFrame(d.get("person_frames", []))
        _zones_d = d.get("zones") or ([{"name": "Danger Zone",
                                        "poly": d.get("zone")}]
                                      if d.get("zone") else [])
        _dzi = inc[inc["type"] == "danger_zone"] if "type" in inc \
            else inc.iloc[0:0]
        for _z in _zones_d:
            _zn = _z["name"]
            if "zones" in _pfz.columns:
                _inside = _pfz["zones"].apply(lambda L: _zn in L
                                              if isinstance(L, list) else False)
            else:                                   # legacy runs
                _inside = _pfz.get("in_zone",
                                   pd.Series(False, index=_pfz.index))
            if "zone" in _dzi.columns:
                _zinc = _dzi[_dzi["zone"] == _zn]
            elif len(_zones_d) == 1:                # legacy single-zone run
                _zinc = _dzi
            else:
                _zinc = _dzi.iloc[0:0]
            _zrows.extend([
                (f"{_zn} incidents", str(len(_zinc))),
                (f"{_zn} entries",
                 f"{int(_inside.sum())} person-frames · "
                 f"{_pfz[_inside]['person'].nunique() if not _pfz.empty else 0}"
                 " workers"),
            ])

    def _act_sh(k):
        v = _act_tot.get(k, 0)
        return f"{v:.0f}s ({100 * v / _act_trk:.0f}%)" if _act_trk else "0s"

    rep_rows = [
        ("Source", " · ".join(f"{z['name']}: {Path(z.get('video', '?')).stem}"
                              for z in ZSRC) if MULTI
         else Path(d.get("video", "?")).name),
        ("Analysed", f"{d['total_frames']} frames ({dur:.0f}s footage) "
                     f"at {d['processing_fps']} fps"),
        ("Workers in frame (peak)",
          str(st_stats.get("max_concurrent", st_stats.get("workers_seen", 0)))),
        ("Unique workers tracked",
          f"{st_stats.get('workers_seen', 0)}"
          + (f" (merged from {st_stats['track_ids_raw']} raw IDs)"
             if st_stats.get("track_ids_raw") else "")),
        ("Equipment seen", eq),
        ("Helmet compliance", f"{helm}%"),
        ("Vest compliance", f"{vest}%"),
        ("Overall PPE compliance", f"{overall}%"),
        ("Incidents", f"{len(inc)} ({n_crit} critical)"),
        ("Clips saved",
          str(len([p for p in inc["clip"] if p and Path(clip_path(p)).exists()]))),
        ("PPE alert grace",
          f"{st_stats.get('ppe_grace_s', 0):.0f}s sustained non-compliance"
          + (f" (requested {st_stats.get('ppe_grace_requested_s', 0):.0f}s,"
             f" capped for clip length)"
             if st_stats.get("ppe_grace_requested_s") is not None
             and st_stats.get("ppe_grace_requested_s")
             != st_stats.get("ppe_grace_s") else "")),
        ("Known worker identities",
          (f"{st_stats.get('registry_known', 0)} known · "
           f"{st_stats.get('registry_new', 0)} new (W-IDs)")
          if (st_stats.get("registry_known") or st_stats.get("registry_new"))
          else "registry not used for this run"),
        ("Time working", _act_sh("working")),
        ("Time idle", _act_sh("idle")),
        ("Time walking / sitting", f"{_act_sh('walking')} / {_act_sh('sitting')}"),
    ] + _zrows
    st.dataframe(pd.DataFrame(rep_rows, columns=["Metric", "Value"]),
                 hide_index=True, width="stretch")

    st.markdown("**Compliance**")
    c1, c2 = st.columns(2)
    c1.progress(min(helm, 100) / 100, text=f"Helmet {helm}%")
    c2.progress(min(vest, 100) / 100, text=f"Vest {vest}%")

    longest = inc.sort_values("duration_s", ascending=False).head(3)
    if len(longest):
        st.markdown("**Longest violations**")
        for _, r in longest.iterrows():
            st.markdown(f"- {r['name']} — {wid_label(r['person'])} · "
                        f"{r['duration_s']}s "
                        f"({r['start_s']}s → {r['end_s']}s)")

    _shift_rows = (d.get("shifts") or {}).get("rows") or []
    if any(r.get("person_frames") or r.get("incidents") for r in _shift_rows):
        st.markdown("**Shift summary** — the session split into clock-time "
                    "shifts (same crew, time-of-day view)")
        _sdf = pd.DataFrame([{
            "shift": r["shift"], "window": r["window"],
            "person-frames": r["person_frames"], "workers": r["workers"],
            "helmet": f"{r['helmet_pct']}%" if r.get("helmet_pct") is not None else "–",
            "vest": f"{r['vest_pct']}%" if r.get("vest_pct") is not None else "–",
            "incidents": r["incidents"],
            "working (s)": round((r.get("activity_s") or {}).get("working", 0)),
        } for r in _shift_rows])
        cSh1, cSh2 = st.columns([5, 2], gap="medium",
                                vertical_alignment="top")
        with cSh1:
            st.dataframe(_sdf, hide_index=True, width="stretch")
        with cSh2:
            _inc_by_shift = pd.DataFrame(
                [{"shift": r["shift"],
                  **{ALERT_NAMES.get(k, k): v
                     for k, v in (r.get("by_type") or {}).items()}}
                 for r in _shift_rows]).fillna(0)
            if _inc_by_shift.columns.size > 1:
                st.bar_chart(_inc_by_shift.set_index("shift"), stack=True)

    _gate = d.get("gate_checks") or []
    if _gate:
        _ok = sum(1 for g in _gate if g.get("compliant"))
        st.markdown(f"**Gate PPE checks** — {_ok}/{len(_gate)} passed "
                    "(each worker checked once, on first sighting)")
        _gdf = pd.DataFrame([{
            "worker": wid_label(g.get("person")),
            "zone": g.get("zone") or "",
            "time (s)": g.get("time_s"),
            "result": "✔ compliant" if g.get("compliant")
            else "✘ missing " + ", ".join(g.get("missing") or []),
        } for g in sorted(_gate, key=lambda g: g.get("time_s", 0))])
        st.dataframe(_gdf, hide_index=True, width="stretch")

    st.markdown("**Worker incident log** — who did what, and when")
    log_rows = []
    for wid, grp in inc.groupby("person"):
        parts = []
        for t, tg in grp.groupby("type"):
            spans = ", ".join(f"{r.start_s:.0f}–{r.end_s:.0f}s"
                              for r in tg.itertuples())
            parts.append(f"{ALERT_NAMES.get(t, t)} ×{len(tg)} ({spans})")
        log_rows.append({"Worker": wid_label(wid), "Incidents": len(grp),
                         "Details": " · ".join(parts)})
    log_df = pd.DataFrame(log_rows).sort_values("Incidents", ascending=False)
    st.dataframe(log_df, hide_index=True, width="stretch")

    try:
        from report import make_pdf
        pdf_path = str(RUN_DIR / "safety_report.pdf")
        make_pdf(OUT_JSON, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF report", f, "safety_report.pdf",
                               "application/pdf", width="stretch",
                               icon=":material/picture_as_pdf:")
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")

with tab2:
    with_clip = inc[inc["clip"] != ""].sort_values("start_frame",
                                                    ascending=False)
    # never crash on clips deleted after the run (stale events.json);
    # clip paths resolve relative to the run dir (absolute in legacy runs)
    with_clip = with_clip[with_clip["clip"].map(
        lambda p: Path(clip_path(p)).exists())]
    if with_clip.empty:
        st.info("No clips were saved for these incidents.")
    else:
        st.caption(f"{len(with_clip)} clips (newest first, up to 12 shown)")
        for _, r in with_clip.head(12).iterrows():
            who = wid_label(r["person"])
            if r["type"] == PROXIMITY:
                who += f" · {r.get('equipment', '')}"
            with st.expander(f"{r['name']} — {who} · {r['start_s']}s → "
                             f"{r['end_s']}s ({r['duration_s']}s)"):
                st.video(clip_path(r["clip"]), width="stretch")

with tab5:
    pf = pd.DataFrame(d.get("person_frames", []))
    if pf.empty:
        st.info("No per-person data recorded (run a new analysis).")
    else:
        st.subheader("Person history — pick a worker, see their full timeline")

        summaries = {int(p): (f"{worker_zone_label(p)} · {len(g)} frames · "
                              f"helmet {int(g.helmet.sum())}/{len(g)} · "
                              f"vest {int(g.vest.sum())}/{len(g)}")
                     for p, g in pf.groupby("person")}
        wsel = st.selectbox("Worker", sorted(summaries), format_func=summaries.get)
        hist = pf[pf["person"] == wsel].sort_values("pidx").reset_index(drop=True)

        n = len(hist)
        with st.container(horizontal=True):
            st.metric("Frames seen", n, border=True)
            st.metric("First seen", f"{hist.time_s.iloc[0]:.1f}s", border=True)
            st.metric("Last seen", f"{hist.time_s.iloc[-1]:.1f}s", border=True)
            st.metric("Helmet rate", f"{100 * hist.helmet.sum() / n:.0f}%",
                      border=True)
            st.metric("Vest rate", f"{100 * hist.vest.sum() / n:.0f}%",
                      border=True)
            if MULTI and "zone" in hist.columns:
                for zn in [z["name"] for z in ZSRC]:
                    st.metric(f"{zn} frames",
                              int((hist.zone == zn).sum()), border=True)
            elif "zones" in hist.columns:
                _zns = sorted({zn for L in hist.zones
                               if isinstance(L, list) for zn in L}) \
                    or ["Danger Zone"]
                for zn in _zns:
                    st.metric(f"{zn} frames",
                              int(hist.zones.apply(
                                  lambda L: zn in L
                                  if isinstance(L, list) else False).sum()),
                              border=True)
            else:
                st.metric("Frames in zone", int(hist.in_zone.sum()),
                          border=True)
            if "activity" in hist.columns:
                top = hist.activity.mode().iloc[0]
                st.metric("Top activity", ACTIVITY_NAMES.get(top, top),
                          border=True)
                st.metric("Working",
                          f"{100 * (hist.activity == 'working').mean():.0f}%",
                          border=True)

        if "activity" in hist.columns:
            with st.container(horizontal=True):
                st.metric("Idle",
                          f"{100 * (hist.activity == 'idle').mean():.0f}%",
                          border=True)
                st.metric("Sitting",
                          f"{100 * (hist.activity == 'sitting').mean():.0f}%",
                          border=True)
                st.metric("Walking",
                          f"{100 * (hist.activity == 'walking').mean():.0f}%",
                          border=True)
                st.metric("Fallen frames",
                          int((hist.activity == 'fallen').sum()), border=True)
                st.metric("Gloves", f"{100 * hist.gloves.mean():.0f}%",
                          border=True)
                st.metric("Boots", f"{100 * hist.boots.mean():.0f}%",
                          border=True)
                st.metric("Incidents",
                          int((inc["person"] == wsel).sum()), border=True)
            _aw = (d.get("activity") or {}).get("per_worker_s", {}).get(
                str(wsel), {})
            if _aw:
                st.caption("Time per position: " + " · ".join(
                    f"**{ACTIVITY_NAMES.get(k, k)}** {v:.0f}s"
                    for k, v in sorted(_aw.items(), key=lambda kv: -kv[1])))

        def strip(pairs, on_c, off_c):
            cells = "".join(
                f'<span title="frame {i}" class="cell" '
                f'style="background:{on_c if v else off_c}"></span>'
                for i, v in pairs)
            return f'<div class="tl">{cells}</div>'

        st.markdown("**Timeline** — each block = one processed frame "
                    "(left → right = time)")
        cA, cB, cC = st.columns(3)
        cA.markdown("Helmet" + strip(zip(hist.pidx, hist.helmet),
                                     "#2ed573", "#ff4757"),
                    unsafe_allow_html=True)
        cB.markdown("Vest" + strip(zip(hist.pidx, hist.vest),
                                   "#2ed573", "#ff4757"),
                    unsafe_allow_html=True)
        POSE_C = {"standing": "#2ed573", "sitting": "#1e90ff",
                  "possible_fall": "#ff4757", "unknown": "#57606f"}
        cells = "".join(
            f'<span title="frame {i}: {v}" class="cell" '
            f'style="background:{POSE_C.get(v, "#57606f")}"></span>'
            for i, v in zip(hist.pidx, hist.pose))
        cC.markdown("Pose" + f'<div class="tl">{cells}</div>',
                    unsafe_allow_html=True)
        if "activity" in hist.columns:
            cells = "".join(
                f'<span title="frame {i}: {v}" class="cell" '
                f'style="background:{ACTIVITY_C.get(v, "#57606f")}"></span>'
                for i, v in zip(hist.pidx, hist.activity))
            st.markdown("**Activity**" + f'<div class="tl">{cells}</div>',
                        unsafe_allow_html=True)
            st.caption("PPE: green worn / red missing · Pose: green standing · "
                       "blue sitting · red possible fall · grey unknown · "
                       "Activity: green working · cyan walking · blue sitting · "
                       "yellow standing · grey idle · red fallen")
        else:
            st.caption("PPE: green worn / red missing · Pose: green standing · "
                       "blue sitting · red possible fall · grey unknown")

        st.markdown("**Frame inspector** — pick a worker above, then scrub "
                    "through their appearances")
        row_i = st.select_slider("Record", options=list(hist.index), value=0)
        row = hist.loc[row_i]
        _vsrc = zvid(row.get("zone")) if MULTI and row.get("zone") else OUT_MP4
        cap = cv2.VideoCapture(_vsrc)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.pidx))
        ok, shot = cap.read()
        cap.release()

        def _bbox4(r):
            b = r.get("bbox") if hasattr(r, "get") else None
            return [int(v) for v in b] if isinstance(b, (list, tuple)) \
                and len(b) == 4 else None

        pL, pR = st.columns([3, 2], vertical_alignment="center")
        with pL:
            if ok:
                shot_r = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
                bb = _bbox4(row)
                if bb:
                    x1, y1, x2, y2 = bb
                    cv2.rectangle(shot_r, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                                  (0, 255, 255), 3)
                    cv2.putText(shot_r, f"P{wsel}", (max(4, x1 - 4),
                                  max(22, y1 - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 255), 3)
                st.image(shot_r, width="stretch")
                st.caption(f"processed frame {int(row.pidx)} · original frame "
                           f"{int(row.frame)} · t={row.time_s}s")
        with pR:
            if ok and _bbox4(row):
                Hc, Wc = shot.shape[:2]
                x1, y1, x2, y2 = _bbox4(row)
                pad = int(0.22 * max(x2 - x1, y2 - y1, 20))
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(Wc, x2 + pad), min(Hc, y2 + pad)
                if cx2 - cx1 > 12 and cy2 - cy1 > 12:
                    crop = shot[cy1:cy2, cx1:cx2]
                    f = 360 / max(crop.shape[0], 1)
                    if f > 1.05:
                        crop = cv2.resize(crop, None, fx=f, fy=f,
                                          interpolation=cv2.INTER_CUBIC)
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                             width="stretch")
                    _pos = ACTIVITY_NAMES.get(row.get("activity"),
                                              row.get("pose", ""))
                    st.caption(f"**{wid_label(wsel)}** zoomed · {_pos}")
            st.markdown("- " + person_line(row))

        with st.container(border=True):
            _cols = ["pidx", "frame", "time_s", "pose", "activity", "helmet",
                     "vest", "gloves", "boots", "in_zone", "nearest_eq",
                     "nearest_m", "alerts"] if "activity" in hist.columns else \
                    ["pidx", "frame", "time_s", "pose", "helmet", "vest",
                     "gloves", "boots", "in_zone", "nearest_eq", "nearest_m",
                     "alerts"]
            t = hist[_cols].copy()
            t["alerts"] = t["alerts"].apply(
                lambda x: ", ".join(ALERT_NAMES.get(a, a) for a in x))
            t.columns = (["proc frame", "frame", "time (s)", "pose", "activity",
                          "helmet", "vest", "gloves", "boots", "in zone",
                          "nearest equip", "dist (m)", "alerts"]
                         if "activity" in hist.columns else
                         ["proc frame", "frame", "time (s)", "pose", "helmet",
                          "vest", "gloves", "boots", "in zone", "nearest equip",
                          "dist (m)", "alerts"])
            st.caption(f"Full history for P{wsel} · {len(t)} rows · newest first")
            st.dataframe(t.sort_values("proc frame", ascending=False),
                         height=340, width="stretch", hide_index=True)
            st.download_button("Download this worker's history CSV",
                               t.to_csv(index=False).encode(),
                               f"person_{wsel}_history.csv",
                               icon=":material/download:")

        wc = inc[(inc["person"] == wsel) & (inc["clip"] != "")]
        wc = wc[wc["clip"].map(lambda p: Path(clip_path(p)).exists())]
        if not wc.empty:
            st.markdown(f"**{len(wc)} incident clips for {wid_label(wsel)}**")
            for _, r in wc.sort_values("start_frame",
                                       ascending=False).iterrows():
                with st.expander(f"{r['name']} · {r['start_s']}s → "
                                 f"{r['end_s']}s ({r['duration_s']}s)"):
                    st.video(clip_path(r["clip"]), width="stretch")

with tab3:
    if "zone" not in inc.columns:
        inc["zone"] = ""
    t = inc[["name", "person", "zone", "equipment", "dist_m", "start_s",
             "end_s", "duration_s", "start_frame"]].copy()
    t.insert(1, "worker_id", inc["person"].map(
        lambda p: WID.get(p) or "–"))
    t.columns = ["incident", "worker", "W-ID", "zone", "equipment", "dist (m)",
                 "start (s)", "end (s)", "duration (s)", "frame"]
    st.caption(f"{len(t)} incidents · newest first")
    st.dataframe(t.sort_values("frame", ascending=False), height=400,
                 width="stretch", hide_index=True)
    st.download_button("Download CSV", t.to_csv(index=False).encode(),
                       "incidents.csv", icon=":material/download:")

with tab4:
    piv = inc.pivot_table(index="person", columns="name", values="start_frame",
                          aggfunc="count", fill_value=0)
    piv["TOTAL"] = piv.sum(axis=1)
    piv = piv.sort_values("TOTAL", ascending=False).reset_index().rename(
        columns={"person": "worker"})
    st.caption("Incidents per tracked worker")
    st.dataframe(piv, width="stretch", hide_index=True)
    st.download_button("Download CSV", piv.to_csv(index=False).encode(),
                       "worker_summary.csv", icon=":material/download:")
    st.markdown("---")
    st.subheader("Full annotated video")
    if MULTI:
        cc1, cc2 = st.columns(2, gap="small")
        for col, z in zip((cc1, cc2), ZSRC):
            with col:
                st.markdown(f"**{z['name'].upper()}** · "
                            f"{Path(z.get('video', '?')).name}")
                zp = zvid(z["name"])
                if Path(zp).exists():
                    st.video(zp, width="stretch")
                    with open(zp, "rb") as f:
                        st.download_button(
                            f"Download {z['name']} video", f,
                            f"annotated_{z['name'].replace(' ', '_').lower()}.mp4",
                            "video/mp4", icon=":material/download:")
                else:
                    st.info(f"{z['name']} video not found.")
    elif Path(OUT_MP4).exists():
        st.video(OUT_MP4, width="stretch")
        with open(OUT_MP4, "rb") as f:
            st.download_button("Download video", f, "annotated.mp4", "video/mp4",
                               icon=":material/download:")
    else:
        st.info("Annotated video not found — run a new analysis.")

with tab6:
    st.subheader("Worker activity — Working · Idle · Sitting · Standing · "
                 "Walking · Fallen")
    st.caption("Fused from person tracking, YOLO pose keypoints and "
               "multi-frame motion, with temporal smoothing to suppress "
               "one-frame flicker.")
    actd = d.get("activity") or {}
    tot = actd.get("totals_s", {})
    per_w = actd.get("per_worker_s", {})
    if not tot:
        st.info("No activity data recorded (run a new analysis).")
        st.stop()

    tracked = sum(tot.values())
    cs = st.columns(len(ACTIVITY_ORDER))
    for c, k in zip(cs, ACTIVITY_ORDER):
        v = tot.get(k, 0)
        c.metric(ACTIVITY_NAMES[k], f"{v:.0f}s",
                 f"{100 * v / tracked:.0f}% of tracked time", delta_color="off",
                 border=True)

    _pf2 = pd.DataFrame(d.get("person_frames", []))
    if per_w:
        df = pd.DataFrame(per_w).T.fillna(0.0)
        df.index = [f"P{int(i)}" for i in df.index]
        df = df[[c for c in ACTIVITY_ORDER if c in df.columns]]
        st.markdown(f"**Seconds per activity, per worker** · "
                    f"{len(df)} workers · {tracked:.0f}s tracked in total")
        st.bar_chart(df, horizontal=True, stack=True,
                     color=[ACTIVITY_C[c] for c in df.columns])
        st.caption("Hover a bar for exact values.")

    if not _pf2.empty and "activity" in _pf2.columns:
        piv = _pf2.pivot_table(index=_pf2["time_s"].astype(int),
                               columns="activity", values="frame",
                               aggfunc="count").fillna(0)
        st.markdown("**Workers per activity over time** · per second of footage")
        st.area_chart(piv)

    seg = pd.DataFrame(actd.get("segments", []))
    if not seg.empty:
        seg["activity"] = seg["activity"].map(lambda a: ACTIVITY_NAMES.get(a, a))
        seg = seg[["person", "activity", "start_s", "end_s", "duration_s"]]
        seg.columns = ["worker", "activity", "start (s)", "end (s)",
                       "duration (s)"]
        st.markdown(f"**Activity timeline** · {len(seg)} segments · "
                    "who was doing what, and when")
        st.dataframe(seg.sort_values("start (s)", ascending=False), height=320,
                     width="stretch", hide_index=True)
        st.download_button("Download activity CSV", seg.to_csv(index=False).encode(),
                           "activity_summary.csv", icon=":material/download:")

with tab7:
    st.subheader("Worker presence heatmap")
    st.caption("Where workers spent their time on site — built from every "
               "tracked person-box of this run. Red = most time spent, "
               "blue = least. Magenta outline = danger zone.")
    _pfh = pd.DataFrame(d.get("person_frames", []))
    if _pfh.empty:
        st.info("No per-person data recorded for this run — run a new analysis.")
    elif MULTI:
        zname = st.selectbox("Zone", [z["name"] for z in ZSRC])
        zrow = next(z for z in ZSRC if z["name"] == zname)
        _rows = _pfh[_pfh.get("zone") == zname]
        _sel = st.selectbox("Show", ["All workers"]
                            + [worker_zone_label(p)
                               for p in sorted(_rows.person.unique())])
        _pid = None if _sel == "All workers" else int(_sel.split(" · ")[0][1:])
        _zv = zvid(zname)
        if not Path(_zv).exists():
            st.info(f"{zname} video missing — run a new analysis.")
        else:
            img = presence_heatmap(_zv, zrow.get("video"),
                                   _rows.to_dict("records"), person=_pid)
            st.image(img, width="stretch")
            st.caption(f"{zname} · {_sel} · {len(_rows):,} person-frames · "
                       "background: source video")
    else:
        _sel = st.selectbox(
            "Show", ["All workers"] + [f"P{p}" for p in
                                       sorted(_pfh.person.unique())],
            help="Heatmap for the whole crew, or one worker at a time")
        _pid = None if _sel == "All workers" else int(_sel[1:])
        if not Path(OUT_MP4).exists():
            st.info("Annotated video missing — heatmap needs its resolution. "
                    "Run a new analysis.")
        else:
            img = presence_heatmap(OUT_MP4, d.get("video"),
                                   _pfh.to_dict("records"), person=_pid,
                                   zones=d.get("zones")
                                   or ([{"name": "Zone 2",
                                         "poly": d["zone"]}]
                                      if d.get("zone") else None))
            st.image(img, width="stretch")
            _n = len(_pfh) if _pid is None else int((_pfh.person == _pid).sum())
            st.caption(f"{_sel} · {_n:,} person-frames of presence · "
                       "background: source video")
