"""Task adapter for the unchanged original UJI A recipe; source list supplied by fit_source."""
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

from frozen_hurdle_translator import FrozenHurdleTranslator
from law_bench.run import _data_identity


class ReFi:
    name = 'refi'

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def fit_source(self, sources):
        if not sources or any(s['dataset'] != 'uji' for s in sources):
            raise ValueError('Original A training entry supports UJI source buildings only')
        self.sources = [s['building'] for s in sources]
        identity = _data_identity({s['building']: s['floors'] for s in sources},
                                  {s['building']: s['ap_ids'] for s in sources})
        root = Path(__file__).resolve().parents[1]
        package = root/'claude_hurdle_20261110'
        code = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), root/'frozen_hurdle_translator.py',
                          *sorted(package.glob('*.py'))]}
        provenance = dict(sources=self.sources, data=identity, code=code,
                          seeds=[1, 2, 3, 4, 5], draws=4,
                          reader=dict(beta=128, eps=.02, offsets=True),
                          evaluation_status='exposed_development; reader selected on UJI B0')
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache = self.cache_dir/key
        cache.mkdir(parents=True, exist_ok=True)
        (cache/'provenance.json').write_text(json.dumps(provenance, indent=2))
        self.translator = FrozenHurdleTranslator.__new__(FrozenHurdleTranslator)
        self.translator.models = []
        H = None
        for seed in range(1, 6):
            path = cache/f'seed_{seed}.joblib'
            tick = time.monotonic()
            if path.exists():
                model = joblib.load(path)
                state = 'reused'
            else:
                if H is None:
                    sys.path.insert(0, str(package))
                    import hurdle as H
                    C = H.C
                    for source in sources:
                        b = int(source['building'][1:])
                        assert list(source['ap_ids']) == C.WAPS
                        assert sorted(source['floors']) == H.V.FLOORS[b]
                        for f, d in source['floors'].items():
                            mask = (C.BLD == b) & (C.FLR == f)
                            assert np.array_equal(d['rss'], C.RSS[mask])
                            assert np.array_equal(d['xy'], C.XY[mask])
                field = H.HurdleField([int(b[1:]) for b in self.sources], seed, draws=4)
                model = dict(cols=field.cols, **{name: dict(estimator=getattr(field, name).est,
                            cols=getattr(field, name).cols) for name in ('det', 'mean', 'scale')})
                temporary = path.with_suffix('.tmp')
                joblib.dump(model, temporary, compress=3)
                temporary.replace(path)
                state = 'trained'
                del field
            self.translator.models.append(model)
            print(json.dumps(dict(refi_seed=seed, state=state, seconds=time.monotonic()-tick,
                                  sources=self.sources)), flush=True)
        self.provenance = provenance
        return self

    def predict(self, task):
        if task.dataset == 'uji' and task.building in self.sources:
            raise ValueError('Target building is a training source')
        prepared = self.translator.prepare(
            {f'{task.building}F{f}': d for f, d in task.H.items()}, task.S,
            f'{task.building}F{task.floor}')
        self.diagnostics = dict(recipe='verbatim HurdleField', seeds=[1, 2, 3, 4, 5],
                                sources=self.sources, candidate_rows=len(prepared.cand_xy))
        return prepared.predict(task.Q_rss)['projected']
