"""Entry/exit gate PPE check: verify each worker ONCE, when first seen
stably, instead of alerting continuously for the whole session.

Useful when the camera covers a gate/entrance: every worker is checked on
the way in, the result is logged (compliant / missing items), and the run
stays quiet about PPE afterwards. Other alert types (danger zone, falls,
proximity) are unaffected.
"""
from collections import defaultdict

GATE_MIN_FRAMES = 10   # processed frames seen before the check fires
GATE_REQUIRED = ("helmet", "vest")


class GateChecker:
    """Feed each tracked person once per processed frame via check();
    returns a gate-check record the first time a track becomes stable,
    and None afterwards (one check per track per run)."""

    def __init__(self, min_frames=GATE_MIN_FRAMES, required=GATE_REQUIRED):
        self.min_frames = max(1, int(min_frames))
        self.required = tuple(required)
        self.seen = defaultdict(int)
        self.done = set()

    def check(self, tid, time_s, ppe_matched):
        """tid: track id; time_s: session time of this frame;
        ppe_matched: set of PPE items currently detected on this track.
        Returns {'person', 'time_s', 'compliant', 'missing'} or None."""
        if tid is None or tid < 0 or tid in self.done:
            return None
        self.seen[tid] += 1
        if self.seen[tid] < self.min_frames:
            return None
        self.done.add(tid)
        missing = [i for i in self.required
                   if i not in (ppe_matched or set())]
        return {"person": tid, "time_s": time_s,
                "compliant": not missing, "missing": missing}
