"""One real multi-H source scenario/site through unchanged LAW case interfaces.

This checks cache/coordinate/AP plumbing and fits S-only source weights. It
does not score queries, train a shared network or run target localization.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from conditional_operator_learning import fit_episode, prior_for
from public_corpus.adapter import load_building, source_task
from public_corpus.history_cache import farthest_rows, load_cases, write
from run import tensor
from transfer import History


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    results = []
    for name, histories in (("UAH2016-b20", [0, 2, 3]), ("Pisa2019-b60", [0, 2])):
        start = time.perf_counter()
        building = load_building(a.data / name)
        cases, meta = load_cases(building, a.cache, 1, histories)
        target = building["floors"][1]
        assert len(cases) == len(meta["AP_indices"])
        # The target is sampled from its full source floor, independently of H
        # subset/J. Only the historical scans are FPS-budgeted.
        for h in (histories, histories[:1]):
            full = source_task(building, 1, h, 5, 1, 12, 261026)
            assert full["S"]["scan_ids"].tolist() == meta["support_scan_ids"]
            assert full["Q"]["scan_ids"].tolist() == meta["query_scan_ids"]
        lookup = {s: i for i, s in enumerate(target["scan_ids"])}
        sr = [lookup[s] for s in meta["support_scan_ids"]]
        seen = (target["rss"][sr] != 100).any(0)
        for floor in histories:
            selection = meta["history_selection"][str(floor)]
            rows = np.array(selection["row_indices"])
            raw = building["floors"][floor]
            np.testing.assert_array_equal(rows, farthest_rows(raw, 128, selection["seed"]))
            assert len(rows) == len(np.unique(raw["xy"][rows], axis=0)) <= 128
            assert raw["scan_ids"][rows].tolist() == selection["scan_ids"]
            seen |= (raw["rss"][rows] != 100).any(0)
        assert list(cases) == np.flatnonzero(seen).tolist()
        assert not set(meta["support_scan_ids"]) & set(meta["query_scan_ids"])
        weights, trials = fit_episode(cases, meta["base"], meta["obs"])
        assert len(weights) == len(histories)
        assert abs(sum(weights) - 1) < 1e-8
        translation_error, checked = 0., False
        for ap, case in cases.items():
            assert case["sx"].shape == (5, 2) and case["qx"].shape == (12, 2)
            assert len(case["histories"]) == len(histories)
            assert case["rss"].shape == case["group"].shape == case["query_weights"].shape == (12,)
            assert not (case["sx"][:, None] == case["qx"][None]).all(-1).any()
            assert (case["stats"]["n"] == 1).all()
            prior = prior_for(None, case, weights, meta["base"])
            m, features = prior.features(case["sx"])
            cov = prior.covariance(case["sx"], case["sx"], features, features)
            assert torch.isfinite(m).all() and torch.isfinite(cov).all()
            if not checked:
                for floor, shifted in zip(histories, case["histories"]):
                    if shifted is None:
                        continue
                    with np.load(a.cache / name / f"F{floor}" / f"ap{ap}.npz") as z:
                        raw_h = History(tensor(z["z"]), dict(mean=tensor(z["mean"]), cov=tensor(z["cov"])), meta["base"])
                    left = raw_h.sample_mean(case["sx"] + tensor(meta["origin"]))
                    right = shifted.sample_mean(case["sx"])
                    torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-9)
                    translation_error = float((left - right).abs().max())
                    checked = True
                    break
        assert checked
        write(a.cache / f"{name}_example.json", meta)
        result = dict(building=name, target=1, histories=histories, K=5, R=1, Q_positions=12,
                      target_available_positions=len(np.unique(target["xy"], axis=0)),
                      APs=len(cases), support_only_APs=sum(all(h is None for h in c["histories"]) for c in cases.values()),
                      history_scans=[s["used_scans"] for s in meta["history_selection"].values()],
                      weights=weights, adaptation_trials=trials,
                      translation_max_error=translation_error, seconds=time.perf_counter() - start,
                      query_signal_used_for_adaptation=False, query_scored=False,
                      shared_network_updates=0, localization=False, passed=True)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    write(a.cache / "case_check.json", dict(passed=True, scenarios=results))


if __name__ == "__main__":
    main()
