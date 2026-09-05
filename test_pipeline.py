"""Offline tests for pipeline post-processing and the new v6 features:
track merging, incident consolidation, severity config, shift summaries,
gate checks, the worker registry and PPE grace auto-scaling."""
import numpy as np

from pipeline import (consolidate_incidents, cap_ppe_grace,
                      merge_fragmented_tracks)
from vision.alerting import DEFAULT_SEVERITY, severity_map
from vision.gate import GateChecker
from vision.registry import WorkerRegistry
from vision.shifts import (DEFAULT_SHIFTS, normalise_shifts, parse_hhmm,
                           shift_for_minutes, shift_summary)


# ---------------------------------------------------------------- helpers --
def rows_for(tid, pidxs, cx=480.0, cy=260.0, w=80, h=200):
    return [{"person": tid, "pidx": p, "bbox": [cx - w / 2, cy - h / 2,
                                                cx + w / 2, cy + h / 2]}
            for p in pidxs]


# ------------------------------------------------- track merging (v5 core) --
def test_merge_concurrent_duplicate_tracks():
    # same position at the same times: two tracker IDs, one person
    rows = rows_for(1, range(100)) + rows_for(2, range(100), cx=482.0)
    m = merge_fragmented_tracks(rows, gap_frames=90)
    assert m[1] == m[2]


def test_merge_chained_fragments():
    # track 1 ends at pidx 100, track 2 appears nearby at pidx 105
    rows = rows_for(1, range(0, 100)) + rows_for(2, range(105, 200), cx=485.0)
    m = merge_fragmented_tracks(rows, gap_frames=90)
    assert m[1] == m[2]


def test_no_merge_across_the_frame():
    # two workers at opposite corners, alive at the same time
    rows = rows_for(1, range(0, 100), cx=100.0, cy=100.0) + \
        rows_for(2, range(0, 100), cx=860.0, cy=440.0)
    m = merge_fragmented_tracks(rows, gap_frames=90)
    assert m[1] != m[2]


def test_merge_renumbers_by_first_appearance():
    rows = rows_for(7, range(0, 100)) + rows_for(9, range(105, 200), cx=485.0)
    m = merge_fragmented_tracks(rows, gap_frames=90)
    assert set(m.values()) == {1}       # single merged person -> id 1


# -------------------------------------------- incident consolidation (v5) --
def _inc(t, p, s, e, clip=None):
    return {"type": t, "person": p, "start_frame": s, "end_frame": e,
            "start_s": s / 25, "end_s": e / 25,
            "duration_s": (e - s) / 25, "clip": clip}


def test_consolidate_adjacent_same_person():
    incs = [_inc("no_helmet", 1, 100, 150, "clips/a.mp4"),
            _inc("no_helmet", 1, 152, 160, "clips/b.mp4")]   # 2-frame gap
    out = consolidate_incidents(incs, fps=25)
    assert len(out) == 1
    assert out[0]["clip"] == "clips/a.mp4"       # longer clip kept


def test_consolidate_keeps_distant_apart():
    incs = [_inc("no_helmet", 1, 100, 150),
            _inc("no_helmet", 1, 400, 450)]      # 250-frame gap
    assert len(consolidate_incidents(incs, fps=25)) == 2


def test_consolidate_separates_people():
    incs = [_inc("no_helmet", 1, 100, 150), _inc("no_helmet", 2, 105, 155)]
    assert len(consolidate_incidents(incs, fps=25)) == 2


# --------------------------------------------- severity config (acc fix 5) --
def test_severity_defaults_and_overrides():
    assert severity_map() == DEFAULT_SEVERITY
    sev = severity_map({"no_helmet": "Medium", "bogus": "Critical",
                        "possible_fall": "NotALevel"})
    assert sev["no_helmet"] == "Medium"
    assert sev["possible_fall"] == DEFAULT_SEVERITY["possible_fall"]
    assert "bogus" not in sev


# --------------------------------------------------- shifts (new feature 8) --
def test_parse_hhmm_and_wrap():
    assert parse_hhmm("06:00") == 360
    assert parse_hhmm("23:59") == 24 * 60 - 1
    assert shift_for_minutes(7 * 60) == "Morning"
    assert shift_for_minutes(15 * 60) == "Afternoon"
    assert shift_for_minutes(23 * 60) == "Night"
    assert shift_for_minutes(3 * 60) == "Night"      # after midnight wrap
    assert shift_for_minutes(4 * 60 + 59) == "Night"


def test_normalise_shifts_falls_back():
    assert normalise_shifts(None) == DEFAULT_SHIFTS
    assert normalise_shifts([{"name": "Bad"}]) == DEFAULT_SHIFTS
    ns = normalise_shifts([{"name": "Early", "start": "05:00", "end": "13:00"}])
    assert ns == [{"name": "Early", "start": "05:00", "end": "13:00"}]


def test_shift_summary_buckets():
    # 10 s session starting 07:00 (Morning): 250 frames at 25 fps
    data = {"total_frames": 250, "stats": {"started_at": "2026-09-02 07:00:00",
                                           "video_fps": 25.0, "stride": 1},
            "person_frames": [
                {"person": 1, "time_s": 1.0, "helmet": True, "vest": False},
                {"person": 1, "time_s": 2.0, "helmet": True, "vest": False},
                {"person": 2, "time_s": 3.0, "helmet": False, "vest": True},
            ],
            "incidents": [{"type": "no_helmet", "person": 2, "time_s": 3.0}],
            "activity": {"segments": [
                {"person": 1, "activity": "working", "time_s": 1.0,
                 "duration_s": 4.0}]}}
    s = shift_summary(data)
    rows = {r["shift"]: r for r in s["rows"]}
    assert rows["Morning"]["person_frames"] == 3
    assert rows["Morning"]["workers"] == 2
    assert rows["Morning"]["incidents"] == 1
    assert abs(rows["Morning"]["helmet_pct"] - 66.7) < 0.1
    assert rows["Afternoon"]["person_frames"] == 0
    assert rows["Morning"]["activity_s"].get("working") == 4.0


# ------------------------------------------------ gate checks (new feature 10) --
def test_gate_checker_fires_once():
    g = GateChecker(min_frames=5)
    results = [g.check(1, i * 0.04, {"helmet"} if i < 3 else {"helmet",
                                                              "vest"})
               for i in range(10)]
    fired = [r for r in results if r]
    assert len(fired) == 1
    assert fired[0]["compliant"] is True
    assert fired[0]["missing"] == []
    assert fired[0]["time_s"] == (4) * 0.04          # at the 5th frame


def test_gate_checker_missing_items():
    g = GateChecker(min_frames=3)
    r = None
    for i in range(5):
        out = g.check(2, i * 0.04, set())
        if out is not None:
            r = out                     # fires exactly once, then None
    assert r["compliant"] is False
    assert set(r["missing"]) == {"helmet", "vest"}


def test_gate_checker_ignores_untracked():
    g = GateChecker(min_frames=1)
    assert g.check(-1, 0.0, set()) is None


# -------------------------------------------- worker registry (acc fix 1) --
def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=32).astype(np.float32)
    return v / np.linalg.norm(v)


def test_registry_persists_identity(tmp_path):
    p = tmp_path / "registry.json"
    r1 = WorkerRegistry(p).assign({1: _emb(0), 2: _emb(1)})
    assert set(r1.values()) == {"W-01", "W-02"}
    # a NEW run, same appearances (fresh process would reload from disk)
    r2 = WorkerRegistry(p).assign({5: _emb(0), 6: _emb(1)})
    assert r2[5] == r1[1] and r2[6] == r1[2]
    # a third appearance gets a new id
    r3 = WorkerRegistry(p).assign({7: _emb(9)})
    assert r3[7] == "W-03"
    assert WorkerRegistry(p).size == 3


def test_registry_one_to_one_per_run(tmp_path):
    # two very similar in-run workers: must NOT share one W-id
    e = _emb(0)
    out = WorkerRegistry(tmp_path / "r.json").assign({1: e, 2: e.copy()})
    assert out[1] != out[2]


# -------------------------------------------- PPE grace auto-cap (fix 2) --
def test_cap_ppe_grace():
    assert cap_ppe_grace(120.0, 45.0) == 45.0 * 0.25    # capped to 11.25 s
    assert cap_ppe_grace(10.0, 45.0) == 10.0            # below the cap
    assert cap_ppe_grace(0.0, 45.0) == 0.0              # Immediate stays 0
    assert cap_ppe_grace(120.0, None) == 120.0          # webcam: unknown dur
    assert cap_ppe_grace(120.0, 4.0) == 2.0             # floor applies


# --- PPE smoothing must be symmetric and stride-independent (regression) ---
def test_ppe_off_window_is_stride_independent():
    """The old code latched an item ON after one detection and only released
    it after 20 consecutive misses. Because PPE runs every 2nd processed
    frame, at stride 5 one lucky hit marked a worker compliant for ~8 s of
    video, so no PPE alert could ever fire. The window is now defined in
    seconds and must stay ~constant in wall-clock terms across strides."""
    import pipeline as P

    def off_seconds(fps, stride):
        interval = P.PPE_EVERY * max(stride, 1) / fps
        n = max(2, round(P.PPE_OFF_S / max(interval, 1e-6)))
        return n * interval

    for stride in (1, 2, 3, 5, 10):
        secs = off_seconds(25.0, stride)
        assert secs <= 3.0, f"stride {stride}: {secs:.1f}s latch is too long"
    # and the old failure mode is gone: 20 misses at stride 5 was ~8 s
    assert off_seconds(25.0, 5) < 4.0


def test_ppe_conf_thresholds_are_per_class():
    """A single flat threshold cannot serve prompts whose score distributions
    differ by 10x (helmet median 0.75 vs vest median 0.065 on sample crops)."""
    from vision.detectors import PPE_CONF, PPE_MODEL_FLOOR
    assert PPE_CONF["helmet"] > PPE_CONF["vest"]
    assert all(v >= PPE_MODEL_FLOOR for v in PPE_CONF.values())
