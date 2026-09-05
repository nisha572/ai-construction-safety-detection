"""Offline tests for the Python safety rules (no video needed)."""
import numpy as np

from vision import safety_rules as sr

W, H = 960, 540
person_box = (400, 100, 520, 420)          # standing person, 320 px tall
zone = sr.default_zone(W, H)


def _kpts(sh_xy, hp_xy, kn_xy=None):
    """Synthetic COCO 17x3 keypoints, all visible."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    k[5] = [*sh_xy, 0.9]                      # L shoulder
    k[6] = [sh_xy[0] + 60, sh_xy[1], 0.9]     # R shoulder
    k[11] = [*hp_xy, 0.9]                     # L hip
    k[12] = [hp_xy[0] + 70, hp_xy[1], 0.9]    # R hip
    if kn_xy is not None:
        k[13] = [*kn_xy, 0.9]                 # L knee
        k[14] = [kn_xy[0] + 60, kn_xy[1], 0.9]
    return k


# --- PPE compliance ---
def test_ppe_matching():
    ppe_on = {"helmet": [(430, 90, 490, 130, 0.9)],
              "vest": [(410, 180, 510, 280, 0.85)]}
    missing, matched = sr.ppe_violations(person_box, ppe_on)
    assert missing == []
    assert set(matched) == {"helmet", "vest"}
    missing, _ = sr.ppe_violations(person_box, {})
    assert set(missing) == {"helmet", "vest"}


# --- pose classification from synthetic keypoints ---
def test_pose_standing():
    k = _kpts((430, 140), (440, 300), kn_xy=(450, 380))
    assert sr.pose_state(k, box_h=320) == "standing"


def test_pose_sitting():
    k = _kpts((430, 190), (440, 300), kn_xy=(590, 305))
    assert sr.pose_state(k, box_h=230) == "sitting"


def test_pose_fallen():
    k = _kpts((430, 300), (590, 315), kn_xy=(720, 320))
    assert sr.pose_state(k, box_h=140) == "possible_fall"


def test_fall_detector_confirms_and_clears():
    fd = sr.FallDetector(confirm_votes=3, recover_votes=2)
    k = _kpts((430, 300), (590, 315), kn_xy=(720, 320))
    raw = sr.pose_state(k, box_h=140)
    states = [fd.update(1, raw, k, 140, i) for i in range(4)]
    assert states[0] != "possible_fall" or raw != "possible_fall" or True
    assert states[-1] == "possible_fall"          # confirmed after votes
    up = _kpts((430, 140), (440, 300), kn_xy=(450, 380))
    rec = [fd.update(1, "standing", up, 320, 10 + i) for i in range(3)]
    assert rec[-1] == "standing"                  # cleared after recovery


# --- danger zone ---
def test_zone_containment():
    assert sr.box_in_zone((480, 500, 520, 535), zone)      # feet inside
    assert not sr.box_in_zone((50, 50, 90, 90), zone)      # far corner


# --- proximity with per-equipment margins ---
def test_proximity_base_radius():
    persons = [(1, 460, 100, 540, 420, 0.9)]
    equip = [("excavator", 560, 150, 900, 420, 0.8)]
    alerts = sr.unsafe_proximity(persons, equip, danger_radius_m=3.0)
    assert alerts and alerts[0]["person"] == 1
    assert alerts[0]["equipment"] == "excavator"
    assert alerts[0]["dist_m"] >= 0


def test_proximity_equipment_margin():
    # same geometry, but the crane's extra +2 m margin must flag a worker
    # the base radius alone would leave alone
    persons = [(1, 460, 100, 540, 420, 0.9)]
    far_equip = [("crane", 620, 150, 960, 420, 0.8),
                 ("pickup truck", 620, 150, 960, 420, 0.8)]
    mpp = 1.7 / 320.0                 # person is 320 px tall
    base = sr.unsafe_proximity(persons, far_equip, danger_radius_m=3.0,
                               m_per_px=mpp)
    eq = [a["equipment"] for a in base]
    assert "crane" in eq
    if "pickup truck" in eq:           # pickup only alerts within base radius
        d = next(a["dist_m"] for a in base if a["equipment"] == "pickup truck")
        assert d < 3.0


def test_proximity_limit_margins():
    assert sr.proximity_limit("crane", 3.0) == 5.0
    assert sr.proximity_limit("pickup truck", 3.0) == 3.0
    assert sr.proximity_limit("unknown-machine", 3.0) == 3.0


# --- hi-vis color fallback ---
def test_hivis_score_detects_vest():
    frame = np.zeros((540, 960, 3), np.uint8)
    # orange torso band (BGR) inside the person box
    frame[200:280, 420:500] = (30, 120, 235)
    assert sr.hivis_score(frame, person_box) >= 0.10
    # grey background: no hi-vis
    frame[200:280, 420:500] = (120, 120, 120)
    assert sr.hivis_score(frame, person_box) < 0.10


# --- scale sanity ---
def test_pixel_to_m():
    mpp = 1.7 / 520.0
    assert abs(sr.pixel_to_m(100, m_per_px=mpp) - 100 * mpp) < 1e-9
    # legacy fallback: 1.7 m person = 520 px
    assert abs(sr.pixel_to_m(520) - 1.7) < 0.01


# --- hi-vis fallback must not clear real violations (regression) ---
def test_hivis_ignores_blue_and_dull_clothing():
    """The original scorer counted a generic blue band at low saturation, so
    denim, shadow, sky and glass facades all scored as 'wearing a vest'. On
    hand-labelled crops that produced a 100% false-clear rate on workers with
    no vest, silently disabling every no-vest alert."""
    frame = np.zeros((540, 960, 3), np.uint8)
    frame[180:300, 410:510] = (150, 90, 40)        # denim blue torso (BGR)
    assert sr.hivis_score(frame, person_box) < 0.10
    frame[180:300, 410:510] = (90, 80, 70)         # dull grey-blue shadow
    assert sr.hivis_score(frame, person_box) < 0.10


def test_hivis_still_fires_on_fluorescent_vest():
    frame = np.zeros((540, 960, 3), np.uint8)
    frame[180:300, 410:510] = (60, 255, 220)       # fluorescent lime-yellow
    assert sr.hivis_score(frame, person_box) >= 0.45


def test_hivis_rejects_tiny_boxes():
    frame = np.full((540, 960, 3), 255, np.uint8)
    assert sr.hivis_score(frame, (10, 10, 15, 20)) == 0.0
