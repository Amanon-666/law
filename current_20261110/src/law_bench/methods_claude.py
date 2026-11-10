"""law_bench Method wrappers for Claude's modules: literature GP baselines and our method (+ ablations).
Installed as law_bench/methods_claude.py next to law_bench_ours.py and law_bench_gp_lit.py (renamed ours.py, gp_lit.py).
"""
import time
import numpy as np
from .task import Method
from . import ours as O
from . import gp_lit as G


def _floors(src):
    return {f: dict(rss=d['rss'], xy=d['xy']) for f, d in src['floors'].items()}


def _hs_aps(task):
    """AP columns detected in H or the current S; Query never extends or selects the axis"""
    seen = (task.S['rss'] != 100).any(0)
    for d in task.H.values():
        seen |= (d['rss'] != 100).any(0)
    return np.flatnonzero(seen)


def _bounds_xy(task):
    return np.vstack([d['xy'] for d in task.H.values()] + [task.S['xy']])


class GpS(Method):
    name = 'gp_S'

    def fit_source(self, sources):
        fl = []
        for src in sources:
            for f, d in src['floors'].items():
                fl.append((d['xy'], G.enc(d['rss'])))                 # raw scans, not position means
        self.hyper = G.fit_hyper(fl)

    def predict(self, task):
        a = _hs_aps(task)
        return G.gp_S(task.S['rss'][:, a], task.S['xy'], task.Q_rss[:, a], _bounds_xy(task), self.hyper)


class GpHS(GpS):
    """SIMPLIFIED history-mean GP: nearest lower/upper history floor's position mean as GP prior mean + GP on S
    residuals. Not the project's retained GP (kept as its own comparator); non-learned rival of ours."""
    name = 'gp_histmean'

    def predict(self, task):
        a = _hs_aps(task)
        maps = O.sub_maps(O.make_maps(task.H), a)
        ref_fn = lambda X: O.T.history_features(X, task.floor, {f: m[:3] for f, m in maps.items()})[1]
        return G.gp_HS(task.S['rss'][:, a], task.S['xy'], task.Q_rss[:, a], _bounds_xy(task), ref_fn, self.hyper)


class OursMethod(Method):
    """variant: full | noM2 | C1 | v1 ; seeds: operator ensemble; theta selected on sources by held-out NLL"""
    def __init__(self, variant='full', seeds=(1, 2, 3), informative=False, fixed_theta=None):
        self.variant, self.seeds, self.inf, self.fixed = variant, seeds, informative, fixed_theta
        self.name = 'ours' if variant == 'full' else f'ours_{variant}'
        if informative:
            self.name += '_Ainfo'

    def fit_source(self, sources):
        t0 = time.time()
        b = [_floors(s) for s in sources]
        rel = rnd = self.variant != 'v1'
        theta = self.fixed
        if theta is None and self.variant == 'full':
            theta, self.theta_scores = O.select_theta(b, seed=self.seeds[0], reliability=rel, randomise=rnd)
        self.model = O.Ours(theta=theta or (16.0, 0.0, 1.0, 1.0), seeds=self.seeds, variant=self.variant,
                            informative=self.inf)
        self.model.ops = [O.Operator(b, s, reliability=rel, randomise=rnd) for s in self.seeds]
        self.theta = self.model.theta; self.fit_seconds = time.time() - t0

    def predict(self, task):
        return self.model.predict(task.H, task.floor, task.S['rss'], task.S['xy'], task.Q_rss)
