"""Shared alert metadata: display labels + tunable severity levels.

One canonical home for the alert-type vocabulary used by the pipeline,
report generator and dashboard, so severity can be configured once and
consumed everywhere (stored in each run's events.json stats).
"""

TYPE_LABEL = {"no_helmet": "No Helmet", "no_vest": "No Vest",
              "possible_fall": "Possible Fall", "danger_zone": "Danger Zone",
              "proximity": "Unsafe Proximity"}

SEV_LEVELS = ["Critical", "High", "Medium", "Low"]

DEFAULT_SEVERITY = {"no_helmet": "High", "no_vest": "High",
                    "possible_fall": "Critical", "danger_zone": "Critical",
                    "proximity": "Critical"}


def severity_map(overrides=None):
    """Default severity per alert type, updated by user overrides.

    Overrides may map alert types to any of SEV_LEVELS; unknown keys and
    unknown values are ignored, so a stale/invalid config can never break
    report generation.
    """
    out = dict(DEFAULT_SEVERITY)
    for k, v in (overrides or {}).items():
        if k in out and v in SEV_LEVELS:
            out[k] = v
    return out


def count_severe(incidents, severity=None, levels=("Critical",)):
    """How many incidents sit at or above one of the given severity levels."""
    sev = severity_map(severity)
    return sum(1 for i in incidents if sev.get(i.get("type")) in levels)
