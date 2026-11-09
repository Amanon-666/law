"""Budgeted public source history GPs and one-episode LAW case loading.

No downloads, neural training, target localization or coordinate rescaling.
Prepare on ray-web with existing base/obs; max 128 distinct H positions/floor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from data import group_scans
import inference
from multi_history import single_history
from run import ap_stats, tensor
from transfer import History
from public_corpus.adapter import load_building, source_task


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def farthest_rows(raw, limit, seed):
    """FPS of exact label positions; seeded start/ties fixed by sorted positions.

    RSS never enters the selection. Choose one actual raw row per retained
    position using the same fixed local generator, without replacing records.
    """
    positions, group = np.unique(raw["xy"], axis=0, return_inverse=True)
    group = group.ravel()
    rng = np.random.default_rng(seed)
    keep, nearest = [], np.full(len(positions), np.inf)
    j = int(rng.integers(len(positions)))
    for _ in range(min(limit, len(positions))):
        keep.append(j)
        nearest = np.minimum(nearest, ((positions - positions[j]) ** 2).sum(1))
        nearest[keep] = -1
        j = int(nearest.argmax())
    return np.array([rng.choice(np.flatnonzero(group == g)) for g in keep], dtype=int)


def prepare(data_root, output, parameters, pilot=False):
    """Pilot 3 observed APs/site, then resume the exact same all-floor cache."""
    data_root, output, parameters = map(Path, (data_root, output, parameters))
    output.mkdir(parents=True, exist_ok=True)
    base_obs = json.loads(parameters.read_text())
    base, obs = base_obs["base"], base_obs["obs"]
    names = ["UAH2016-b20", "Pisa2019-b60"]
    inputs = {str(p): digest(p) for name in names
              for p in sorted((data_root / name).iterdir()) if p.suffix in (".json", ".npz")}
    core = [Path(inference.__file__), Path(__file__), Path(single_history.__code__.co_filename),
            Path(History.__init__.__code__.co_filename), Path(source_task.__code__.co_filename)]
    freeze = dict(schema="law-public-history-v1", limit_positions=128, scans_per_position=1,
                  seed=261026, VI_maxiter=640, base=base, obs=obs,
                  parameters=str(parameters), parameters_sha256=digest(parameters),
                  input_sha256=inputs, code_sha256={str(p): digest(p) for p in core},
                  coordinate_policy="unscaled corpus metric xy; cached z in shared public frame",
                  selection="deterministic coordinate-only FPS; seeded first position and raw row",
                  purpose="source-history preprocessing; not neural training or localization")
    frozen_path = output / "freeze.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != freeze:
            raise ValueError("Frozen cache inputs/code/parameters changed; use a new cache directory")
    else:
        write(frozen_path, freeze)
    events_path = output / "events.jsonl"

    def event(**value):
        value["UTC"] = datetime.now(timezone.utc).isoformat()
        with events_path.open("a") as f:
            f.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    started = time.perf_counter()
    building_summaries = []
    for name in names:
        building = load_building(data_root / name)
        floor_summaries = []
        for floor, raw in sorted(building["floors"].items()):
            dest = output / name / f"F{floor}"
            dest.mkdir(parents=True, exist_ok=True)
            seed = 261026 + (2016 if name.startswith("UAH") else 2019) * 100 + floor
            rows = farthest_rows(raw, 128, seed)
            scans = group_scans(raw["rss"][rows], raw["xy"][rows])
            selection = dict(available_scans=len(raw["rss"]),
                             available_positions=len(np.unique(raw["xy"], axis=0)),
                             used_positions=len(rows), used_scans=len(rows), seed=seed,
                             row_indices=rows.tolist(), scan_ids=raw["scan_ids"][rows].tolist(),
                             observed_APs=int((scans["stats"]["k"].sum(0) > 0).sum()))
            selection_path = dest / "selection.json"
            if selection_path.exists():
                if json.loads(selection_path.read_text()) != selection:
                    raise ValueError(f"History selection changed: {dest}")
            else:
                write(selection_path, selection)
            observed = np.flatnonzero(scans["stats"]["k"].sum(0) > 0)
            if pilot:
                if floor != min(building["floors"]):
                    continue
                ranked = observed[np.argsort(scans["stats"]["k"].sum(0)[observed], kind="stable")]
                aps = np.unique(ranked[[0, len(ranked) // 2, -1]])
            else:
                aps = range(len(building["ap_ids"]))
            stage_started = time.perf_counter()
            for ap in aps:
                ap = int(ap)
                cache, status = dest / f"ap{ap}.npz", dest / f"ap{ap}.json"
                if status.exists() and cache.exists():
                    continue
                details = {}
                original_minimize = inference.minimize

                def observed_minimize(*args, **kwargs):
                    result = original_minimize(*args, **kwargs)
                    details.update(optimizer_success=bool(result.success), iterations=int(result.nit),
                                   evaluations=int(result.nfev), message=str(result.message),
                                   objective=float(result.fun), max_gradient=float(np.abs(result.jac).max()))
                    return result

                t0 = time.perf_counter()
                try:
                    with patch("inference.minimize", observed_minimize):
                        h = single_history(dict(scans=scans), ap, base, obs, steps=640)
                    arrays = dict(empty=np.array(h is None))
                    if h is not None:
                        if not torch.isfinite(h.m).all() or not torch.isfinite(h.c).all():
                            raise RuntimeError("Nonfinite cached historical posterior")
                        arrays.update(z=h.z.numpy(), mean=h.m.numpy(), cov=h.c.numpy())
                    np.savez_compressed(cache, **arrays)
                    write(status, dict(AP=ap, empty=h is None, seconds=time.perf_counter() - t0,
                                       optimizer=details, finite=True))
                except Exception as error:
                    event(stage="failure", building=name, floor=floor, AP=ap,
                          seconds=time.perf_counter() - t0, error=repr(error))
                    raise
                if pilot or ap % 50 == 0:
                    event(stage="pilot" if pilot else "progress", building=name, floor=floor, AP=ap,
                          empty=h is None, seconds=time.perf_counter() - t0, optimizer=details)
            statuses = [json.loads(p.read_text()) for p in dest.glob("ap*.json")]
            summary = dict(floor=floor, **{k: selection[k] for k in ("available_scans", "available_positions", "used_scans", "used_positions", "observed_APs")},
                           cached_APs=len(statuses), nonempty=sum(not x["empty"] for x in statuses),
                           no_detection=sum(x["empty"] for x in statuses),
                           optimizer_not_success=sum(bool(x["optimizer"]) and not x["optimizer"]["optimizer_success"] for x in statuses),
                           AP_fit_seconds=sum(x["seconds"] for x in statuses),
                           stage_seconds=time.perf_counter() - stage_started)
            floor_summaries.append(summary)
            event(stage="floor_done", building=name, **summary)
        building_summaries.append(dict(name=name, floors=floor_summaries))
    result = dict(pilot=pilot, complete=not pilot, buildings=building_summaries,
                  stage_seconds=time.perf_counter() - started,
                  total_AP_fit_seconds=sum(f["AP_fit_seconds"] for b in building_summaries for f in b["floors"]),
                  cache_file_bytes=sum(p.stat().st_size for p in output.rglob("ap*.npz")),
                  no_downloads=True, no_network_training=True)
    write(output / ("pilot.json" if pilot else "summary.json"), result)
    return result


def load_cases(building, cache_root, target, histories, K=5, R=1, Q_count=12, seed=261026):
    """Return (conditional_operator cases, source metadata), one scenario at a time.

    Global AP integers index cache files only. The network receives existing
    history content/dz/positions/stats, never the AP integer or BSSID itself.
    """
    cache_root = Path(cache_root)
    frozen = json.loads((cache_root / "freeze.json").read_text())
    base = frozen["base"]
    name = building["metadata"]["name"]
    selections = {f: json.loads((cache_root / name / f"F{f}" / "selection.json").read_text()) for f in histories}
    task = source_task(building, target, histories, K, R, Q_count, seed,
                       history_rows={f: v["row_indices"] for f, v in selections.items()})
    origin = tensor(task["metadata"]["origin"])
    S, Q = task["S"], task["Q"]
    cases = {}
    for local, ap in enumerate(task["ap_indices"]):
        hs = []
        for floor in histories:
            with np.load(cache_root / name / f"F{floor}" / f"ap{ap}.npz", allow_pickle=False) as a:
                h = None if bool(a["empty"]) else History(tensor(a["z"]) - origin,
                    dict(mean=tensor(a["mean"]), cov=tensor(a["cov"])), base)
            hs.append(h)
        cases[int(ap)] = dict(histories=hs, sx=tensor(S["positions"]), qx=tensor(Q["positions"]),
                              stats=ap_stats(S, local), dz=task["dz"], rss=tensor(Q["rss"][:, local]),
                              group=torch.tensor(Q["group"]), query_weights=tensor(Q["scan_weights"]))
    meta = dict(**task["metadata"], target=target, histories=list(histories),
                AP_indices=task["ap_indices"].tolist(), AP_ids=task["ap_ids"].tolist(),
                support_scan_ids=S["scan_ids"].tolist(), query_scan_ids=Q["scan_ids"].tolist(),
                history_selection={str(f): s for f, s in selections.items()},
                base=frozen["base"], obs=frozen["obs"], cache_root=str(cache_root))
    return cases, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(261026)
    print(json.dumps(prepare(a.data, a.output, a.parameters, a.pilot)), flush=True)
