"""Generate a one-page PDF safety report from output/events.json."""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from vision.alerting import TYPE_LABEL, severity_map
except Exception:      # standalone use outside the project root
    TYPE_LABEL = {"no_helmet": "No Helmet", "no_vest": "No Vest",
                  "possible_fall": "Possible Fall", "danger_zone": "Danger Zone",
                  "proximity": "Unsafe Proximity"}

    def severity_map(overrides=None):
        out = {"no_helmet": "High", "no_vest": "High",
               "possible_fall": "Critical", "danger_zone": "Critical",
               "proximity": "Critical"}
        for k, v in (overrides or {}).items():
            if k in out and v in ("Critical", "High", "Medium", "Low"):
                out[k] = v
        return out

DARK = (22, 27, 43)
ACCENT = (0, 150, 199)
RED = (220, 53, 69)
ORANGE = (255, 159, 64)
GREEN = (40, 167, 69)
GREY = (120, 130, 150)

SEV_COLOR = {"Critical": RED, "High": ORANGE, "Medium": (200, 160, 30),
             "Low": GREY}


def compliance(data):
    """True per-frame wear-rates from person_frames (helmet/vest matched on
    each person-frame). Falls back to the old alert-count estimate for legacy
    JSONs without per-frame data."""
    rows = data.get("person_frames") or []
    if rows:
        helm = 100.0 * sum(bool(r.get("helmet")) for r in rows) / len(rows)
        vest = 100.0 * sum(bool(r.get("vest")) for r in rows) / len(rows)
    else:
        st = data.get("stats", {})
        pf = max(st.get("person_frames", 0), 1)
        ac = data.get("alert_counts", {})
        helm = max(0.0, 100 * (1 - ac.get("no_helmet", 0) / pf))
        vest = max(0.0, 100 * (1 - ac.get("no_vest", 0) / pf))
    return round(helm, 1), round(vest, 1), round((helm + vest) / 2, 1)


def _mmss(sec):
    sec = float(sec or 0)
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def worker_log(data):
    """Per-worker incident summary: {worker: [(type, start_s, end_s), ...]}."""
    from collections import defaultdict

    per = defaultdict(list)
    for i in data.get("incidents", []):
        per[i["person"]].append((i["type"], i.get("start_s", 0), i.get("end_s", 0)))
    return dict(per)


def zone_summary(data):
    """Per-zone aggregates: incidents, workers entered, person-frames inside.

    Handles legacy single-zone runs (no 'zones'/'zone' fields on rows) by
    mapping everything to the single zone defined in the run."""
    zs = data.get("zones")
    if not zs and data.get("zone"):
        zs = [{"name": "Danger Zone", "poly": data["zone"]}]
    if not zs:
        return []          # two-camera runs: camera zones are reported instead
    legacy = not any("zones" in (r or {}) for r in data.get("person_frames") or [])
    rows = data.get("person_frames") or []
    inc = [i for i in data.get("incidents", []) if i["type"] == "danger_zone"]
    out = []
    for z in zs:
        n = z["name"]
        if legacy:
            pf = sum(1 for r in rows if r.get("in_zone"))
            workers = len({r["person"] for r in rows if r.get("in_zone")})
            zi = [i for i in inc if not i.get("zone")]
        else:
            inside = [r for r in rows if n in (r.get("zones") or [])]
            pf = len(inside)
            workers = len({r["person"] for r in inside})
            zi = [i for i in inc if i.get("zone") == n]
        out.append({"name": n, "person_frames": pf, "workers": workers,
                    "incidents": len(zi),
                    "longest_s": max((i.get("duration_s", 0) for i in zi),
                                     default=0)})
    return out


def worker_details(data):
    """Per-worker presence / PPE / position aggregates from person_frames.

    {worker: {frames, first_s, last_s, helmet..boots counts, zone frames,
               activity_s: {activity: seconds}}}
    """
    rows = data.get("person_frames") or []
    per = {}
    for r in rows:
        w = per.setdefault(r["person"], {
            "frames": 0, "helmet": 0, "vest": 0, "gloves": 0, "boots": 0,
            "zone_frames": {}, "first_s": r.get("time_s", 0),
            "last_s": r.get("time_s", 0), "activity_s": {}})
        w["frames"] += 1
        for k in ("helmet", "vest", "gloves", "boots"):
            if r.get(k):
                w[k] += 1
        zn = r.get("zones") or (["Danger Zone"] if r.get("in_zone") else [])
        for name in zn:
            w["zone_frames"][name] = w["zone_frames"].get(name, 0) + 1
        t = r.get("time_s", 0)
        w["first_s"] = min(w["first_s"], t)
        w["last_s"] = max(w["last_s"], t)
    for wid, w in per.items():
        w["activity_s"] = ((data.get("activity") or {})
                           .get("per_worker_s", {}).get(str(wid), {}))
    return per


def worker_ids_map(data):
    """{person_id: worker_id ('W-03')} from whichever rows carry one."""
    out = {}
    for r in data.get("person_frames") or []:
        w = r.get("worker_id")
        if w:
            out[r.get("person")] = w
    for i in data.get("incidents") or []:
        w = i.get("worker_id")
        if w:
            out[i.get("person")] = w
    return out


def make_pdf(events_json, pdf_path, severity=None):
    from fpdf import FPDF

    data = json.load(open(events_json))
    inc = data.get("incidents", [])
    st = data.get("stats", {})
    multi = bool(data.get("multi_zone"))
    zsrc = data.get("zone_sources") or []
    wz = data.get("worker_zones") or {}
    wids = worker_ids_map(data)
    sev = severity_map(severity if severity is not None
                       else st.get("severity"))
    helm, vest, overall = compliance(data)
    n_crit = sum(1 for i in inc if sev.get(i["type"]) == "Critical")

    pdf = FPDF()
    pdf.add_page()
    W = pdf.w

    # header band
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, W, 34, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 19)
    pdf.set_xy(12, 9)
    pdf.cell(0, 9, "CONSTRUCTION SITE SAFETY REPORT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_xy(12, 20)
    if multi and zsrc:
        src = " | ".join(f"{z['name']}: {Path(z.get('video', '?')).stem}"
                         for z in zsrc)
    else:
        src = Path(data.get("video", "?")).name
    pdf.cell(0, 6, f"Source: {src}    Generated: {datetime.now():%d %b %Y %H:%M}")

    pdf.set_auto_page_break(True, margin=14)

    # KPI boxes
    y0 = 42
    kpis = [(f"{len(inc)}", "Incidents", RED if inc else GREEN),
            (f"{n_crit}", "Critical", RED if n_crit else GREEN),
            (f"{st.get('workers_seen', 0)}", "Workers seen", ACCENT),
            (f"{overall}%", "PPE compliance", GREEN if overall >= 70 else ORANGE)]
    bw, gap = (W - 24 - 3 * 6) / 4, 6
    for i, (val, lab, col) in enumerate(kpis):
        x = 12 + i * (bw + gap)
        pdf.set_fill_color(245, 247, 250)
        pdf.set_draw_color(225, 230, 238)
        pdf.set_xy(x, y0)
        pdf.rect(x, y0, bw, 26, "DF")
        pdf.set_xy(x, y0 + 3)
        pdf.set_font("helvetica", "B", 17)
        pdf.set_text_color(*col)
        pdf.cell(bw, 9, val, align="C")
        pdf.set_xy(x, y0 + 14)
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(*GREY)
        pdf.cell(bw, 6, lab, align="C")

    # session meta
    y = y0 + 36
    pdf.set_xy(12, y)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "Session Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(60, 70, 90)
    dur = data.get("total_frames", 0) * max(st.get("stride", 1), 1) / max(st.get("video_fps", 25), 1)
    eq = ", ".join(f"{k} ({v} frames)" for k, v in st.get("equipment_seen", {}).items()) or "none"
    grace = st.get("ppe_grace_s", 0) or 0
    dz_grace = st.get("danger_grace_s", 1.0) or 0
    meta = [f"Analysed {data.get('total_frames', 0)} frames ({dur:.0f}s of footage) at {data.get('processing_fps', 0)} fps processing",
            f"Workers on site: {st.get('max_concurrent', st.get('workers_seen', 0))} peak in frame"
            f" · {st.get('workers_seen', 0)} unique tracked"
            + (f" (merged from {st.get('track_ids_raw', 0)} raw IDs)" if st.get("track_ids_raw") else ""),
            f"Equipment detected: {eq}",
            f"Helmet compliance: {helm}%  |  Vest compliance: {vest}%"]
    if st.get("gate_mode"):
        n_gate = len(data.get("gate_checks") or [])
        n_bad = sum(1 for g in data.get("gate_checks") or []
                    if not g.get("compliant"))
        meta.append(f"PPE checked once at entry (gate mode): {n_gate} workers"
                    f" checked, {n_bad} flagged")
    else:
        meta.append(f"Alerting: helmet/vest after {grace:.0f}s sustained non-compliance"
                    f" · danger zone after {dz_grace:.0f}s inside the zone")
    if st.get("ppe_zone_only"):
        meta.append("PPE rules apply inside the defined zones only")
    if st.get("low_light"):
        meta.append("Low-light enhancement enabled during detection")
    if st.get("registry_known") or st.get("registry_new"):
        meta.append(f"Worker identities: {st.get('registry_known', 0)} known"
                    f" · {st.get('registry_new', 0)} newly registered"
                    f" (persistent W-IDs)")
    for m in meta:
        pdf.set_x(12)
        pdf.cell(4, 6.5, "-", align="C")
        pdf.cell(0, 6.5, m, new_x="LMARGIN", new_y="NEXT")

    # zone summary — two-camera runs: one row per zone (video)
    if multi and zsrc:
        y = pdf.get_y() + 6
        pdf.set_xy(12, y)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 8, "Zone Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_fill_color(*DARK)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 9.5)
        heads = ["Zone", "Video", "Workers", "Incidents", "Helmet", "Vest"]
        colw = [24, 52, 24, 26, 24, 24]
        for h, c in zip(heads, colw):
            pdf.cell(c, 8, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(60, 70, 90)
        pdf.set_font("helvetica", "", 9.5)
        for z in zsrc:
            row = [z["name"], Path(z.get("video", "?")).stem,
                   str(z.get("workers", 0)), str(z.get("incidents", 0)),
                   f"{z.get('helmet_pct', 0)}%", f"{z.get('vest_pct', 0)}%"]
            for v, c in zip(row, colw):
                pdf.cell(c, 8, v, border=1, align="C")
            pdf.ln()

    # danger-zone summary — per zone
    zsum = zone_summary(data)
    if zsum:
        y = pdf.get_y() + 6
        pdf.set_xy(12, y)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 8, "Danger Zones", new_x="LMARGIN", new_y="NEXT")
        pdf.set_fill_color(*DARK)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 9.5)
        heads = ["Zone", "Incidents", "Workers entered", "Frames in zone",
                 "Longest (s)"]
        colw = [34, 28, 36, 32, 30]
        for h, c in zip(heads, colw):
            pdf.cell(c, 8, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(60, 70, 90)
        pdf.set_font("helvetica", "", 9.5)
        for z in zsum:
            row = [z["name"], str(z["incidents"]), str(z["workers"]),
                   str(z["person_frames"]), f"{z['longest_s']:.1f}"]
            for v, c in zip(row, colw):
                pdf.cell(c, 8, v, border=1, align="C")
            pdf.ln()

    # shift summary — per clock-time shift (morning / afternoon / night)
    shift_rows = ((data.get("shifts") or {}).get("rows") or [])
    if any(r.get("person_frames") or r.get("incidents") for r in shift_rows):
        y = pdf.get_y() + 6
        pdf.set_xy(12, y)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 8, "Shift Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_fill_color(*DARK)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 9.5)
        heads = ["Shift", "Window", "Person-frames", "Workers",
                 "Helmet", "Vest", "Incidents"]
        colw = [26, 30, 30, 24, 24, 24, 28]
        for h, c in zip(heads, colw):
            pdf.cell(c, 8, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(60, 70, 90)
        pdf.set_font("helvetica", "", 9.5)
        for r in shift_rows:
            row = [r.get("shift", ""), r.get("window", ""),
                   str(r.get("person_frames", 0)), str(r.get("workers", 0)),
                   f"{r['helmet_pct']}%" if r.get("helmet_pct") is not None else "-",
                   f"{r['vest_pct']}%" if r.get("vest_pct") is not None else "-",
                   str(r.get("incidents", 0))]
            for v, c in zip(row, colw):
                pdf.cell(c, 8, v, border=1, align="C")
            pdf.ln()

    # gate PPE checks — one check per worker at entry (gate mode runs)
    gate = data.get("gate_checks") or []
    if gate:
        y = pdf.get_y() + 6
        pdf.set_xy(12, y)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        n_ok = sum(1 for g in gate if g.get("compliant"))
        pdf.cell(0, 8, f"Gate PPE Checks - {n_ok}/{len(gate)} passed",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(60, 70, 90)
        pdf.set_font("helvetica", "", 9.5)
        for g in sorted(gate, key=lambda g: g.get("time_s", 0)):
            wlab = f" ({g['worker_id']})" if g.get("worker_id") else ""
            zlab = f" · {g['zone']}" if g.get("zone") else ""
            if g.get("compliant"):
                res = "compliant - full PPE"
            else:
                res = "missing " + " + ".join(g.get("missing") or ["PPE"])
            pdf.set_x(12)
            pdf.cell(4, 6.5, "-", align="C")
            pdf.cell(0, 6.5,
                     f"Worker {g.get('person')}{wlab}{zlab} at "
                     f"{_mmss(g.get('time_s', 0))} - {res}",
                     new_x="LMARGIN", new_y="NEXT")

    # worker activity
    act = data.get("activity") or {}
    tot = act.get("totals_s", {})
    per_w = act.get("per_worker_s", {})
    if tot:
        tracked = sum(tot.values())
        y = pdf.get_y() + 6
        pdf.set_xy(12, y)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 8, "Worker Activity", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(60, 70, 90)
        shares = "  |  ".join(f"{k.title()}: {v:.0f}s ({100 * v / tracked:.0f}%)"
                              for k, v in sorted(tot.items(),
                                                 key=lambda kv: -kv[1]))
        pdf.set_x(12)
        pdf.multi_cell(0, 6.5, f"{tracked:.0f}s of tracked worker time - " + shares)
        if per_w:
            y = pdf.get_y() + 3
            pdf.set_xy(12, y)
            pdf.set_fill_color(*DARK)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("helvetica", "B", 9)
            heads = ["Worker", "Working", "Idle", "Walking", "Sitting",
                     "Standing", "Fallen", "Total"]
            colw = [22, 25, 22, 25, 24, 27, 22, 19]
            for h, c in zip(heads, colw):
                pdf.cell(c, 7.5, h, border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_text_color(60, 70, 90)
            pdf.set_font("helvetica", "", 9)
            order = sorted(per_w.items(),
                           key=lambda kv: -sum(kv[1].values()))[:10]
            for wid, dd in order:
                pdf.set_x(12)
                cells = [f"W{wid}",
                         f"{dd.get('working', 0):.0f}",
                         f"{dd.get('idle', 0):.0f}",
                         f"{dd.get('walking', 0):.0f}",
                         f"{dd.get('sitting', 0):.0f}",
                         f"{dd.get('standing', 0):.0f}",
                         f"{dd.get('fallen', 0):.0f}",
                         f"{sum(dd.values()):.0f}"]
                for v, c in zip(cells, colw):
                    pdf.cell(c, 7, v, border=1, align="C")
                pdf.ln()
            if len(per_w) > 10:
                pdf.set_x(12)
                pdf.set_font("helvetica", "I", 8.5)
                pdf.set_text_color(*GREY)
                pdf.cell(0, 6, f"(+{len(per_w) - 10} more workers)",
                         new_x="LMARGIN", new_y="NEXT")

    # worker details — per person: presence, PPE wear-rates, position mix
    wd = worker_details(data)
    if wd:
        y = pdf.get_y() + 6
        pdf.set_xy(12, y)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 8, "Worker Details", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(60, 70, 90)
        order = sorted(wd.items(), key=lambda kv: -kv[1]["frames"])[:12]
        for wid, w in order:
            n = max(w["frames"], 1)
            zf = w.get("zone_frames") or {}
            ztxt = " | ".join(f"{k} {v}f" for k, v in sorted(zf.items())) \
                or "never"
            zlab = f" ({wz.get(str(wid))})" if wz.get(str(wid)) else ""
            wlab = f" · {wids[wid]}" if wids.get(wid) else ""
            pdf.set_x(12)
            pdf.set_font("helvetica", "B", 9.5)
            pdf.cell(0, 6.5,
                     f"Worker {wid}{zlab}{wlab} | "
                     f"{_mmss(w['first_s'])}-{_mmss(w['last_s'])}"
                     f" | {w['frames']} frames | in zones: {ztxt}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 9)
            rates = (f"helmet {100 * w['helmet'] / n:.0f}%"
                     f" | vest {100 * w['vest'] / n:.0f}%"
                     f" | gloves {100 * w['gloves'] / n:.0f}%"
                     f" | boots {100 * w['boots'] / n:.0f}%")
            pdf.set_x(18)
            pdf.cell(4, 6, "-", align="C")
            pdf.cell(0, 6, f"PPE worn: {rates}", new_x="LMARGIN", new_y="NEXT")
            acts = w.get("activity_s") or {}
            pos = " | ".join(f"{k} {v:.0f}s"
                             for k, v in sorted(acts.items(),
                                                key=lambda kv: -kv[1]))
            pdf.set_x(18)
            pdf.cell(4, 6, "-", align="C")
            pdf.cell(0, 6, f"Position: {pos}" if pos else "Position: no data",
                     new_x="LMARGIN", new_y="NEXT")
        if len(wd) > 12:
            pdf.set_x(12)
            pdf.set_font("helvetica", "I", 8.5)
            pdf.set_text_color(*GREY)
            pdf.cell(0, 6, f"(+{len(wd) - 12} more workers)",
                     new_x="LMARGIN", new_y="NEXT")

    # incidents by type
    y = pdf.get_y() + 6
    pdf.set_xy(12, y)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "Incidents by Type", new_x="LMARGIN", new_y="NEXT")
    by_type = defaultdict(list)
    for i in inc:
        by_type[i["type"]].append(i)
    pdf.set_font("helvetica", "", 9.5)
    colw = [58, 26, 26, 30, 30]
    heads = ["Type", "Count", "Severity", "Helmet/Vest", "Other"]
    pdf.set_fill_color(*DARK)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 9.5)
    for h, c in zip(heads, colw):
        pdf.cell(c, 8, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(60, 70, 90)
    pdf.set_font("helvetica", "", 9.5)
    if not by_type:
        pdf.cell(sum(colw), 8, "No incidents recorded", border=1, align="C")
        pdf.ln()
    for t, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        is_ppe = t in ("no_helmet", "no_vest")
        row = [TYPE_LABEL.get(t, t), str(len(items)), sev.get(t, "-"),
               f"{helm}% / {vest}%" if is_ppe else "-",
               "-"]
        for v, c in zip(row, colw):
            pdf.cell(c, 8, v, border=1, align="C")
        pdf.ln()

    # top offenders
    y = pdf.get_y() + 6
    pdf.set_xy(12, y)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "Top Workers by Violations", new_x="LMARGIN", new_y="NEXT")
    per = defaultdict(lambda: [0, 0.0])
    for i in inc:
        per[i["person"]][0] += 1
        per[i["person"]][1] = max(per[i["person"]][1], i.get("duration_s", 0) or 0)
    top = sorted(per.items(), key=lambda kv: -kv[1][0])[:8]
    pdf.set_fill_color(*DARK)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 9.5)
    for h, c in zip(["Worker ID", "Incidents", "Longest violation (s)"], [45, 45, 65]):
        pdf.cell(c, 8, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(60, 70, 90)
    pdf.set_font("helvetica", "", 9.5)
    if not top:
        pdf.cell(155, 8, "No violations", border=1, align="C")
        pdf.ln()
    for wid, (n, longest) in top:
        wl = f"{wid} ({wids[wid]})" if wids.get(wid) else str(wid)
        for v, c in zip([wl, str(n), f"{longest:.1f}"], [45, 45, 65]):
            pdf.cell(c, 8, v, border=1, align="C")
        pdf.ln()

    # worker incident log — every worker, their incidents and when
    y = pdf.get_y() + 6
    pdf.set_xy(12, y)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "Worker Incident Log", new_x="LMARGIN", new_y="NEXT")
    per = worker_log(data)
    pdf.set_font("helvetica", "", 9.5)
    pdf.set_text_color(60, 70, 90)
    if not per:
        pdf.set_x(12)
        pdf.cell(0, 7, "No incidents recorded for any worker.", new_x="LMARGIN", new_y="NEXT")
    for wid in sorted(per, key=lambda w: -len(per[w])):
        items = per[wid]
        n = len(items)
        wl = f" ({wids[wid]})" if wids.get(wid) else ""
        pdf.set_x(12)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.cell(0, 6.5, f"Worker {wid}{wl} - {n} incident{'s' if n != 1 else ''}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 9)
        by_type = defaultdict(list)
        for t, s, e in items:
            by_type[t].append((s, e))
        for t, spans in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            when = ", ".join(f"{_mmss(s)}-{_mmss(e)}" for s, e in spans[:6])
            extra = f" (+{len(spans) - 6} more)" if len(spans) > 6 else ""
            pdf.set_x(18)
            pdf.cell(4, 6, "-", align="C")
            pdf.cell(0, 6, f"{TYPE_LABEL.get(t, t)} x{len(spans)}: {when}{extra}",
                     new_x="LMARGIN", new_y="NEXT")

    # footer note
    y = pdf.get_y() + 6
    pdf.set_xy(12, y)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    clips = [i for i in inc if i.get("clip")]
    note = (f"Note: {len(clips)} incident clips saved in this run's clips/ folder. "
            if clips else "Note: no clips saved. ")
    note += "Generated from pretrained models (YOLO26, PPE, YOLO-World, pose) - no model training."
    pdf.multi_cell(0, 6, note)

    pdf.output(pdf_path)
    return pdf_path
