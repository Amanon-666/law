"""Reduced-step synthetic regression for the reviewed coordinate-cache correction."""
import json
import os
from pathlib import Path
import numpy as np
from law_bench.methods_dl import FineTuneDNN, MAMLDNN

os.nice(10)
xy = np.column_stack((np.arange(12, dtype=float), np.arange(12, dtype=float)%3))
rss = np.full((12, 3), -70.0)
H = {0: {'rss': rss, 'xy': xy}, 1: {'rss': rss-5, 'xy': xy+1}}
supports = [{'rss': rss[:1], 'xy': np.array([[x, x]])} for x in (2., 30.)]
result = {'kind': 'synthetic_cache_regression_only', 'methods': {}}
for cls, hp in ((FineTuneDNN, (2, .001, 1, .0003)), (MAMLDNN, (2, .001, 1, .01, 3))):
    model = cls(seed=7, hp=hp)
    outputs = [model._run(H, S, rss[:2], hp, ('synthetic', 'B', 0)) for S in supports]
    assert len(model._cache) == 2
    assert all(np.isfinite(v).all() and v.shape == (2, 2) for v in outputs)
    result['methods'][cls.__name__] = {'cache_entries': len(model._cache), 'finite_shape_2x2': True, 'hp': hp}
Path('results/bench_round4_cache_regression_20261111.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
