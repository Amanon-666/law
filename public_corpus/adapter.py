"""UAH2016/Pisa2019 raw trajectory scans for LAW source tasks; no network I/O.

The parser and metric conversion extend check_mf2.py:parse_ipin/load_ipin_year.
Unlike that statistics loader, retain all APs, raw scans and acquisition IDs,
exclude official evaluation, and never pool a spatial grid into repeated scans.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from data import group_scans


SITES = {
    "UAH2016": (2016, "01-Logifles", ("01-Training",), "logfile_UAH_*.txt"),
    "Pisa2019": (2019, "01-Logfiles", ("01-Training", "02-Validation"), "*.txt"),
}


def parse_log(path):
    """Existing parse_ipin interpolation with acquisition/landmark provenance.

    A scan is one original WIFI AppTimestamp; duplicate BSSIDs at that timestamp
    retain the last record, as in the existing loader. Sensor timestamps remain
    available in the original file, addressed by file + AppTimestamp.
    """
    posi, wifi = [], {}
    with open(path, encoding="latin-1") as fh:
        for line_number, line in enumerate(fh, 1):
            if line.startswith("POSI;"):
                t = line.strip().rstrip(";").split(";")
                posi.append((float(t[1]), float(t[3]), float(t[4]),
                             float(t[5]), float(t[6]), line_number))
            elif line.startswith("WIFI;"):
                t = line.strip().rstrip(";").split(";")
                wifi.setdefault(float(t[1]), {})[t[4].lower()] = float(t[-1])
    P = sorted(posi)
    ts = np.array(sorted(wifi))
    out = []
    for a, b in zip(P[:-1], P[1:]):
        # Existing loader's same-floor, same-building and <=120 s rule.
        if a[3] != b[3] or a[4] != b[4] or b[0] - a[0] > 120:
            continue
        for t in ts[(ts >= a[0]) & (ts < b[0])]:
            w = (t - a[0]) / (b[0] - a[0])
            out.append(dict(lat=a[1] + w * (b[1] - a[1]),
                            lon=a[2] + w * (b[2] - a[2]),
                            floor=int(a[3]), building=int(a[4]), rss=wifi[t],
                            app_timestamp=float(t), landmark_lines=(a[5], b[5]),
                            landmark_times=(a[0], b[0]), fraction=float(w)))
    return out


def convert(site, output, datasets=None, partitions=None):
    """Write one directory per building, one NPZ per floor and a manifest.

    Coordinates use the existing loader's equirectangular metric rule and
    per-building mean lat/lon reference, now recorded with the raw lat/lon.
    partitions may restrict Pisa to ('01-Training',) for a future protocol.
    """
    year, logs, allowed, pattern = SITES[site]
    partitions = tuple(allowed if partitions is None else partitions)
    if not partitions or not set(partitions) <= set(allowed):
        raise ValueError(f"{site}: allowed public partitions are {allowed}")
    datasets = Path.home() / "datasets" if datasets is None else Path(datasets)
    root = datasets / f"ipin_t3_{year}" / f"{year}_IPIN_Competition_Track03"
    files = sorted(p for part in partitions for p in (root / logs / part).rglob(pattern)
                   if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No public logs for {site} under {root}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources, records = [], []
    for file_index, path in enumerate(files):
        rows = parse_log(path)
        sources.append(dict(path=str(path), relative_path=str(path.relative_to(root)),
                            bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            accepted_scans=len(rows)))
        for r in rows:
            r["file_index"] = file_index
        records.extend(rows)
    manifest = dict(schema="law-public-raw-v1", site=site, year=year,
                    partitions=list(partitions), sources=sources,
                    missing_rss=100, rss_unit="dBm", buildings=[],
                    position_kind="time_interpolated_POSI_landmarks",
                    max_supported_R=1,
                    repeat_policy="R=1 only: trajectory labels do not establish static repeated measurements",
                    parser_basis="multifloor-check2-20261026/check_mf2.py:parse_ipin/load_ipin_year",
                    ap_policy="all original BSSIDs, no frequency filter; episode selection uses H/S only")
    for bid in sorted({r["building"] for r in records}):
        rows = [r for r in records if r["building"] == bid]
        name = f"{site}-b{bid}"
        dest = output / name
        dest.mkdir(exist_ok=True)
        lat0, lon0 = np.mean([[r["lat"], r["lon"]] for r in rows], axis=0)
        aps = sorted({a for r in rows for a in r["rss"]})
        col = {a: i for i, a in enumerate(aps)}
        floors = {}
        for floor in sorted({r["floor"] for r in rows}):
            fr = [r for r in rows if r["floor"] == floor]
            latlon = np.array([[r["lat"], r["lon"]] for r in fr])
            xy = np.column_stack(((latlon[:, 1] - lon0) * 111320 * np.cos(np.radians(lat0)),
                                  (latlon[:, 0] - lat0) * 111320))
            rss = np.full((len(fr), len(aps)), 100., np.float32)
            for i, r in enumerate(fr):
                for a, value in r["rss"].items():
                    rss[i, col[a]] = value
            ids = [f"{sources[r['file_index']]['relative_path']}@{r['app_timestamp']!r}" for r in fr]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate acquisition IDs in {name} F{floor}")
            np.savez_compressed(dest / f"F{floor}.npz", xy=xy, raw_latlon=latlon, rss=rss,
                                scan_ids=np.array(ids), file_indices=np.array([r["file_index"] for r in fr]),
                                app_timestamps=np.array([r["app_timestamp"] for r in fr]),
                                landmark_lines=np.array([r["landmark_lines"] for r in fr]),
                                landmark_times=np.array([r["landmark_times"] for r in fr]),
                                interpolation_fraction=np.array([r["fraction"] for r in fr]))
            _, counts = np.unique(xy, axis=0, return_counts=True)
            floors[str(floor)] = dict(file=f"F{floor}.npz", scans=len(fr),
                                      exact_label_positions=len(counts),
                                      positions_with_ge3_labels=int((counts >= 3).sum()),
                                      max_labels_at_exact_position=int(counts.max()),
                                      observed_APs=int((rss != 100).any(0).sum()),
                                      xy_min=xy.min(0).tolist(), xy_max=xy.max(0).tolist())
        meta = dict(site=site, name=name, building_id=bid, ap_namespace=name, ap_ids=aps, floors=floors,
                    max_supported_R=1, position_kind=manifest["position_kind"],
                    transform=dict(input="WGS84 latitude,longitude in degrees", output="x east,y north in metres",
                                   ref_lat=float(lat0), ref_lon=float(lon0), metres_per_degree=111320,
                                   rule="x=(lon-ref_lon)*111320*cos(ref_lat); y=(lat-ref_lat)*111320",
                                   origin_policy="mean of retained public source scan labels; no RSS fitted registration",
                                   floor_alignment="shared geodetic frame; no per-floor fitted transform"))
        (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        manifest["buildings"].append(dict(name=name, directory=name, floors=floors, APs=len(aps)))
    (output / f"{site}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_building(path):
    """Load converted raw arrays; AP identities are scoped to this building/site."""
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    floors = {}
    for f, info in meta["floors"].items():
        with np.load(path / info["file"], allow_pickle=False) as a:
            floors[int(f)] = {key: a[key] for key in a.files}
    return dict(metadata=meta, ap_ids=np.array(meta["ap_ids"]), floors=floors)


def source_task(building, target, histories, K, R, Q_count, seed=0, history_rows=None):
    """Return a variable-H raw source episode in existing group_scans format.

    K and Q_count count distinct exact label positions. All scans at S positions
    are excluded from Q. R>1 is deliberately unsupported for these walking logs.
    H retains all raw scans unless history_rows supplies explicit budget rows.
    This changes H only: S/Q always sample the complete target floor.
    Q.xy/Q stats are source-training labels; deployment must pass only Q.rss.
    No network/model fitting, floor pairing, spatial snapping or fake repeats.
    """
    floors = building["floors"]
    histories = list(histories)
    if target in histories or len(set(histories)) != len(histories):
        raise ValueError("H must contain distinct floors other than the target")
    if R != 1:
        raise ValueError("R must be 1: IPIN trajectory labels do not verify static repeated scans")
    f = floors[target]
    hrows = {h: np.arange(len(floors[h]["xy"])) if history_rows is None
             else np.asarray(history_rows[h], dtype=int) for h in histories}
    positions, group = np.unique(f["xy"], axis=0, return_inverse=True)
    group = group.ravel()
    if K < 0 or Q_count < 1 or K + Q_count > len(positions):
        raise ValueError("Need K>=0, Q_count>=1 and enough disjoint label positions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positions))
    s_pos, q_pos = order[:K], order[K:K + Q_count]
    sr = np.array([rng.choice(np.flatnonzero(group == g)) for g in s_pos], dtype=int)
    qr = np.array([rng.choice(np.flatnonzero(group == g)) for g in q_pos], dtype=int)
    seen = (f["rss"][sr] != 100).any(0)
    for h in histories:
        seen |= (floors[h]["rss"][hrows[h]] != 100).any(0)
    ap_indices = np.flatnonzero(seen)
    # H-only centring matches existing LAW history construction; common shift
    # across every floor. Empty H leaves the documented corpus frame unchanged.
    origin = (np.concatenate([floors[h]["xy"][hrows[h]] for h in histories]).mean(0)
              if histories else np.zeros(2))

    def scans(raw, rows):
        out = group_scans(raw["rss"][rows][:, ap_indices], raw["xy"][rows] - origin)
        out.update(scan_ids=raw["scan_ids"][rows], raw_latlon=raw["raw_latlon"][rows],
                   ap_namespace=building["metadata"]["ap_namespace"],
                   row_indices=rows, ap_ids=building["ap_ids"][ap_indices])
        return out

    S, Q = scans(f, sr), scans(f, qr)
    Q["scan_weights"] = 1 / (Q_count * Q["stats"]["n"][Q["group"]])
    return dict(H=[scans(floors[h], hrows[h]) for h in histories],
                S=S, Q=Q, target=target, histories=histories,
                dz=[h - target for h in histories], ap_indices=ap_indices,
                ap_ids=building["ap_ids"][ap_indices],
                s_position_indices=s_pos, q_position_indices=q_pos,
                metadata=dict(site=building["metadata"]["site"], building=building["metadata"]["name"],
                              K=K, requested_R=R, actual_R=1, Q_count=Q_count, seed=seed,
                              position_kind=building["metadata"]["position_kind"],
                              origin=origin.tolist(), ap_selection="H union S; Q signal excluded"))
