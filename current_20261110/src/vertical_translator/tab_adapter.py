"""Hand a translated target-floor map to the existing, unchanged Tab and GP+Tab paths (inference only).

The translator output is a position-level map of the TARGET floor. For Tab it is passed as one more
"history" whose floor key is the target floor itself, so its rows carry the target floor feature and
look like dense calibration rows; the real S scans are passed as support exactly as before.

    from tab_adapter import load_event_map, map_as_scans
    z = load_event_map(out_dir, event['id'])                       # written by run_nine_floor.py
    translated = map_as_scans(z['cand_xy'], z['map_HS'], z['aps']) # dict(rss int16 with 100 = not detected, xy)

    # context T: translated map + real S                     (history rows replaced)
    model.fit({event['floor']: translated}, support, event['floor'], aps, bounds, seed)
    # context RT: raw other-floor histories + translated + S (history rows kept, translated rows added)
    model.fit({**histories, event['floor']: translated}, support, event['floor'], aps, bounds, seed)

`aps` and `bounds` stay the project's H/S-only values (z['aps'] equals the project's AP rule).
FrozenHistoryRegressor.fit reads only h['rss'] and h['xy'], so no other field is needed; call
uji.group_scans(**translated) first if a caller needs the grouped statistics.
"""
import os
import numpy as np


def load_event_map(out_dir, event_id):
    with np.load(os.path.join(out_dir, event_id + '.npz')) as z:
        return {k: z[k] for k in z.files}


def map_as_scans(cand_xy, enc_map, aps, n_ap=520, detect_threshold=-100.0):
    """Encoded map [n, len(aps)] (dBm, -105 = not detected) -> raw-scan layout [n, 520] with 100 = not detected.
    Cells predicted below detect_threshold are written as not detected (fixed a priori, not tuned)."""
    rss = np.full((len(cand_xy), n_ap), 100, dtype=np.int16)
    val = np.rint(enc_map).astype(np.int16)
    block = np.where(enc_map > detect_threshold, val, 100).astype(np.int16)
    rss[:, np.asarray(aps)] = block
    return dict(rss=rss, xy=np.asarray(cand_xy, dtype=np.float64))
