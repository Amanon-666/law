"""Offline conversion + bounded raw-data interface checks, no training.

Run from the LAW root: python -m public_corpus.check_adapter --output PATH
--loader points at the existing check_mf2.py for parser and coordinate parity.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from public_corpus.adapter import convert, load_building, parse_log, source_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--loader", type=Path, default=Path.home() / "projects/panyushuo/multifloor-check2-20261026/check_mf2.py")
    args = p.parse_args()
    # Lift the exact pure parser without importing its command-line statistics
    # program, reading other years/datasets, or writing its shared cache.
    text = args.loader.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "parse_ipin")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.loader), "exec"), namespace)
    legacy = namespace["parse_ipin"]
    summary = dict(training=False, GPU_used=False, network_downloads=0, sites=[],
                   loader=str(args.loader), loader_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   checks="legacy parser parity; metric transform parity; raw IDs; disjoint S/Q; variable H; H/S-only AP selection; K=0/1; R=3 rejection")
    for site in ("UAH2016", "Pisa2019"):
        manifest = convert(site, args.output)
        parity_scans = 0
        for source in manifest["sources"]:
            old, new = legacy(source["path"]), parse_log(source["path"])
            assert len(old) == len(new)
            for a, b in zip(old, new):
                assert a == (b["lat"], b["lon"], b["floor"], b["building"], b["rss"])
            parity_scans += len(new)
        result = dict(site=site, parser_parity_scans=parity_scans, files=len(manifest["sources"]),
                      raw_input_file_bytes=sum(s["bytes"] for s in manifest["sources"]), buildings=[])
        for b in manifest["buildings"]:
            building = load_building(args.output / b["directory"])
            meta = building["metadata"]
            episodes = []
            for target, raw in building["floors"].items():
                assert np.isfinite(raw["xy"]).all() and np.isfinite(raw["rss"]).all()
                assert len(set(raw["scan_ids"])) == len(raw["xy"])
                assert raw["rss"].shape[1] == len(building["ap_ids"])
                tr = meta["transform"]
                np.testing.assert_array_equal(raw["xy"][:, 0], (raw["raw_latlon"][:, 1] - tr["ref_lon"]) * 111320 * np.cos(np.radians(tr["ref_lat"])))
                np.testing.assert_array_equal(raw["xy"][:, 1], (raw["raw_latlon"][:, 0] - tr["ref_lat"]) * 111320)
                other = sorted(set(building["floors"]) - {target})
                fixed_ids = None
                for J in range(1, len(other) + 1):
                    task = source_task(building, target, other[:J], K=5, R=1, Q_count=12, seed=261026)
                    S, Q = task["S"], task["Q"]
                    assert len(S["positions"]) == 5 and len(Q["positions"]) == 12
                    assert len(S["rss"]) == 5 and len(Q["rss"]) == 12
                    assert not set(S["scan_ids"]) & set(Q["scan_ids"])
                    assert not (S["positions"][:, None] == Q["positions"][None]).all(-1).any()
                    seen = (raw["rss"][S["row_indices"]] != 100).any(0)
                    for h in other[:J]:
                        seen |= (building["floors"][h]["rss"] != 100).any(0)
                    np.testing.assert_array_equal(task["ap_indices"], np.flatnonzero(seen))
                    np.testing.assert_array_equal(task["ap_ids"], building["ap_ids"][task["ap_indices"]])
                    assert S["ap_namespace"] == Q["ap_namespace"] == meta["ap_namespace"]
                    assert len(task["H"]) == J
                    ids = (S["scan_ids"].tolist(), Q["scan_ids"].tolist())
                    assert fixed_ids is None or ids == fixed_ids
                    fixed_ids = ids
                    episodes.append(dict(target=target, histories=other[:J], APs=len(task["ap_ids"]),
                                         support_scans=len(S["rss"]), query_scans=len(Q["rss"]),
                                         K=5, actual_R=1, disjoint_positions=True))
                for K in (0, 1):
                    task = source_task(building, target, other[:1], K=K, R=1, Q_count=12)
                    assert len(task["S"]["positions"]) == K
                try:
                    source_task(building, target, other[:1], K=5, R=3, Q_count=12)
                except ValueError as error:
                    assert "R must be 1" in str(error)
                else:
                    raise AssertionError("R=3 must not synthesize repeated measurements")
            result["buildings"].append(dict(name=b["name"], APs=b["APs"], floors=b["floors"], episodes=episodes))
        summary["sites"].append(result)
        print(json.dumps(result), flush=True)
    summary["passed"] = True
    summary["derived_file_bytes"] = sum(p.stat().st_size for p in args.output.rglob("*.npz"))
    (args.output / "check_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(passed=True, derived_file_bytes=summary["derived_file_bytes"])), flush=True)


if __name__ == "__main__":
    main()
