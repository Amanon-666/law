"""Verify paired score tables, report floor/building means and conditional draw uncertainty."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import BUDGETS, SEEDS

INDEX = ["building", "floor", "draw", "K"]


def table(path, methods):
    df = pd.read_csv(path).rename(columns={"seed":"draw"})
    wide = df[df.method.isin(methods)].pivot(index=INDEX, columns="method", values="mde")
    if len(wide) != 650 or set(wide) != set(methods) or not np.isfinite(wide.values).all():
        raise ValueError(f"Incomplete 13-floor comparison in {path}")
    return wide


def intervals(frame, reference, name):
    # These intervals condition on the three buildings, every floor and its
    # official queries. Only random support choices are resampled.
    rng = np.random.default_rng(20261101)
    n = 5000
    resamples, centers = [], []
    for b, part in frame.groupby(level="building"):
        byfloor = []
        for _, floor in part.groupby(level="floor"):
            a = floor.droplevel(["building","floor"]).unstack("K").sort_index()
            if tuple(a.index) != SEEDS:
                raise ValueError("Mismatched draw IDs")
            v = a["core"][list(BUDGETS)].to_numpy()
            delta = np.diff(v, axis=1) if reference is None else v - a[reference][list(BUDGETS)].to_numpy()
            boot = delta[rng.integers(len(SEEDS), size=(n,len(SEEDS)))].mean(1)
            byfloor.append((delta.mean(0),boot))
        centers.append(np.mean([x[0] for x in byfloor], axis=0))
        resamples.append(np.mean([x[1] for x in byfloor], axis=0))
    center = np.stack(centers)
    noise = np.stack(resamples, axis=1) - center
    # One-sided, nonstudentized, centered bootstrap; one family per comparison.
    critical = np.quantile((-noise).max(axis=(1,2)), .95)
    return [(name,b,k,center[b,i],center[b,i]+critical)
            for b in range(len(center)) for i,k in enumerate(BUDGETS if reference else BUDGETS[1:])]


def main(d):
    core = table(d/"forward_official.csv", ["history","anchors","fixed","learned"])
    np.testing.assert_allclose(core.fixed, core.learned, atol=0)
    wk = table(d/"wknn_events.csv", ["raw_fixed3","prototype_fixed3","joint_tuned"])
    inv = table(d/"inverse_events.csv", ["pooled3","fixed_rule","transported","S_only3"])
    noalign = table(d/"propagation.csv", ["constant","permuted","aligned"])
    np.testing.assert_allclose(core.fixed, noalign.aligned, atol=1e-8, rtol=0)
    np.testing.assert_allclose(wk.raw_fixed3, inv.pooled3, atol=1e-8, rtol=0)
    shape = table(d/"shape_events.csv", ["identity","smooth","sharpen","global_fit","content_fit"])
    np.testing.assert_allclose(core.fixed, shape.identity, atol=1e-8, rtol=0)
    main = pd.concat([core.rename(columns={"fixed":"core"}).drop(columns="learned"),wk,
                      inv.drop(columns="pooled3"),noalign.drop(columns="aligned"),shape.drop(columns="identity")], axis=1)
    main = main.rename(columns={"core": "uncorrected_shape", "smooth": "core"})
    matched = pd.read_csv(d/"map_readout_events.csv").set_index(INDEX)
    main = main.join(matched[["tobit_off","wknn_off","wknn_on","tobit_on"]], validate="one_to_one")
    np.testing.assert_allclose(main.core, main.tobit_on, atol=1e-8, rtol=0.)
    main = main.drop(columns="tobit_on")
    floor = main.groupby(level=["building","floor","K"]).mean()
    floor.to_csv(d/"floor_means.csv")
    building = floor.groupby(level=["building","K"]).mean()
    building.to_csv(d/"building_means.csv")
    building.groupby(level="K").mean().to_csv(d/"macro_curve.csv")
    changes = []
    for ref in ("joint_tuned","prototype_fixed3","wknn_on","tobit_off","uncorrected_shape",None):
        changes += intervals(main, ref, "next_K_minus_previous" if ref is None else f"core_minus_{ref}")
    stress = table(d/"device_stress_events.csv", ["fixed_tobit","strong_wknn"]).rename(columns={"fixed_tobit":"core"})
    phone = table(d/"smooth_scale_phone.csv", ["half10","selected"])
    stress["core"] = phone.half10
    changes += intervals(stress, "strong_wknn", "phone_core_minus_strong_wknn")
    changes += intervals(stress, None, "phone_next_K_minus_previous")
    stress.groupby(level=["building","K"]).mean().to_csv(d/"phone_building_means.csv")
    pd.DataFrame(changes, columns=["contrast","building","K","mean_delta","simultaneous_upper95_given_queries"]).to_csv(d/"paired_intervals.csv", index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    main(p.parse_args().results)
