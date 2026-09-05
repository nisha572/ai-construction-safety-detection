"""Persistent worker registry: stable W-01/W-02... identities that survive
across runs AND across cameras, matched by ReID appearance embeddings.

The pipeline already computes per-track ReID embeddings for its track
merging; this module reuses them after merging as worker "signatures".
At the end of a run each merged worker's average embedding is compared
(1:1 greedy cosine matching) against the registered centroids: known
workers keep their W-id, new ones are registered. The registry is a small
JSON file (output/worker_registry.json) written atomically under a lock,
so two-camera runs can assign through it concurrently from both threads.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np

MATCH_SIM = 0.55     # cosine similarity linking a track to a known worker

_lock = threading.Lock()


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


class WorkerRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {"next_id": 1, "workers": {}}
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                if isinstance(d, dict) and isinstance(d.get("workers"), dict):
                    self.data = d
            except Exception:
                pass

    @property
    def size(self):
        return len(self.data.get("workers", {}))

    def assign(self, embeddings):
        """embeddings: {key: np.ndarray}, keys opaque (merged person ids),
        values L2-normalised appearance vectors.

        Greedy 1:1 matching against registered centroids; every key gets a
        W-id (new workers are registered). Returns {key: "W-xx"}.
        """
        keys = list(embeddings.keys())
        out = {}
        with _lock:
            workers = self.data["workers"]
            pairs = []
            for key in keys:
                emb = np.asarray(embeddings[key], np.float32)
                for wid, w in workers.items():
                    sim = _cos(emb, np.asarray(w["embedding"], np.float32))
                    if sim >= MATCH_SIM:
                        pairs.append((round(sim, 6), str(key), wid))
            pairs.sort(reverse=True)
            taken_keys, taken_wids = set(), set()
            for _sim, skey, wid in pairs:
                if skey in taken_keys or wid in taken_wids:
                    continue
                taken_keys.add(skey)
                taken_wids.add(wid)
                for key in keys:
                    if str(key) == skey:
                        out[key] = wid
                        break
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for key in keys:
                if key in out:
                    continue
                wid = f"W-{int(self.data['next_id']):02d}"
                self.data["next_id"] = int(self.data["next_id"]) + 1
                workers[wid] = {"embedding": [float(x) for x in embeddings[key]],
                                "first_seen": stamp, "last_seen": stamp,
                                "sightings": 0}
                out[key] = wid
            for key in keys:
                w = workers[out[key]]
                w["last_seen"] = stamp
                w["sightings"] = int(w.get("sightings", 0)) + 1
                # running-mean centroid, weighted by sightings so far
                old = np.asarray(w["embedding"], np.float32)
                n = max(int(w["sightings"]) - 1, 1)
                m = old * (n / (n + 1)) + np.asarray(embeddings[key],
                                                     np.float32) / (n + 1)
                nrm = np.linalg.norm(m)
                if nrm > 1e-6:
                    w["embedding"] = [float(x) for x in (m / nrm)]
            self._save()
        return out

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            tmp.replace(self.path)
        except Exception:
            pass
