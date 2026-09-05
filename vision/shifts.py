"""Shift windows: split a session into clock-time shifts (morning /
afternoon / night crews) and summarise alerts, compliance and activity
per shift.

Footage time is mapped onto the wall clock by anchoring the session's
start time (stats.started_at) and adding each row's time_s. For recorded
videos this is a presentation convention (video time ≠ recording time);
for live/webcam runs it is the actual clock. Windows may cross midnight
(e.g. Night 22:00-06:00).
"""
from datetime import datetime, timedelta

DEFAULT_SHIFTS = [{"name": "Morning", "start": "06:00", "end": "14:00"},
                  {"name": "Afternoon", "start": "14:00", "end": "22:00"},
                  {"name": "Night", "start": "22:00", "end": "06:00"}]


def parse_hhmm(s):
    """'HH:MM' -> minutes since midnight ('24:00' = end of day)."""
    h, m = str(s).strip().split(":")
    h, m = int(h), int(m)
    if h == 24 and m == 0:            # '24:00' midnight end marker
        return 24 * 60
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad time {s!r}")
    return h * 60 + m


def shift_for_minutes(mins, shifts=None):
    """Name of the shift containing minute-of-day `mins`, or None.

    Windows are half-open [start, end) and may wrap past midnight."""
    for s in shifts or DEFAULT_SHIFTS:
        a, b = parse_hhmm(s["start"]), parse_hhmm(s["end"])
        if a == b:
            continue
        if a < b:
            if a <= mins < b:
                return s["name"]
        elif mins >= a or mins < b:        # crosses midnight
            return s["name"]
    return None


def normalise_shifts(shifts):
    """Validate a shift config list; fall back to the defaults if empty/bad."""
    clean = []
    for s in shifts or []:
        try:
            clean.append({"name": str(s.get("name") or "Shift")[:24],
                          "start": str(s["start"]), "end": str(s["end"])})
            parse_hhmm(s["start"])
            parse_hhmm(s["end"])
        except Exception:
            continue
    return clean or [dict(s) for s in DEFAULT_SHIFTS]


def _start_dt(data):
    """Wall-clock session start: prefer stats.started_at; legacy runs fall
    back to stats.date minus the session duration."""
    st = data.get("stats", {}) or {}
    for k in ("started_at", "date"):
        v = st.get(k)
        if not v:
            continue
        try:
            return datetime.strptime(str(v)[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
    if st.get("date"):
        try:
            start = datetime.strptime(str(st["date"])[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
        try:
            dur = data.get("total_frames", 0) * max(st.get("stride", 1), 1) \
                / max(st.get("video_fps", 25) or 25, 1)
            return start - timedelta(seconds=dur)
        except Exception:
            return start
    return None


def shift_summary(data, shifts=None):
    """Per-shift aggregates over person_frames / incidents / activity.

    Returns {"config": [shift dicts], "rows": [one summary per shift]} with
    rows always present for every configured shift (zeros where empty), so
    tables render stably. Rows carry: person_frames, workers, helmet/vest
    wear-rate, incidents, by_type, activity_s."""
    shifts = normalise_shifts(shifts)
    start = _start_dt(data)
    buckets = {s["name"]: {"shift": s["name"],
                           "window": f"{s['start']}-{s['end']}",
                           "person_frames": 0, "workers": set(),
                           "helmet": 0, "vest": 0, "incidents": 0,
                           "by_type": {}, "activity_s": {}}
               for s in shifts}
    if start is None:
        return {"config": shifts,
                "rows": [{k: (list(v[k]) if isinstance(v[k], set) else v[k])
                          for k in ("shift", "window", "person_frames",
                                    "incidents")} for v in buckets.values()]}

    def bname(row):
        t = row.get("time_s")
        if t is None:
            return None
        clock = start + timedelta(seconds=float(t))
        return shift_for_minutes(clock.hour * 60 + clock.minute, shifts)

    for r in data.get("person_frames") or []:
        b = buckets.get(bname(r))
        if not b:
            continue
        b["person_frames"] += 1
        b["workers"].add(r.get("person"))
        b["helmet"] += bool(r.get("helmet"))
        b["vest"] += bool(r.get("vest"))
    for i in data.get("incidents") or []:
        b = buckets.get(bname(i))
        if not b:
            continue
        b["incidents"] += 1
        b["by_type"][i.get("type")] = b["by_type"].get(i.get("type"), 0) + 1
    for sgm in (data.get("activity") or {}).get("segments") or []:
        b = buckets.get(bname(sgm))
        if b:
            a = sgm.get("activity")
            b["activity_s"][a] = b["activity_s"].get(a, 0) \
                + (sgm.get("duration_s", 0) or 0)

    rows = []
    for s in shifts:
        b = buckets[s["name"]]
        n = b["person_frames"]
        rows.append({"shift": s["name"], "window": b["window"],
                     "person_frames": n, "workers": len(b["workers"]),
                     "helmet_pct": round(100 * b["helmet"] / n, 1) if n else None,
                     "vest_pct": round(100 * b["vest"] / n, 1) if n else None,
                     "incidents": b["incidents"], "by_type": b["by_type"],
                     "activity_s": {k: round(v, 1)
                                    for k, v in b["activity_s"].items()}})
    return {"config": shifts, "rows": rows}
