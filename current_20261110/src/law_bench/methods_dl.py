"""Standard deep-learning transfer baselines for law_bench (Claude, 2026-11-11).

Both use the SAME fingerprint DNN localiser (AP-vector input -> planar xy), the standard representation of the
fingerprinting literature. Because AP columns are building-specific, such models can only use history of the SAME
building (the other floors H share its AP axis); cross-building source buildings cannot be consumed directly - this
is the representation contrast of ablation A2.

  FineTuneDNN  pre-train on all H scans (other floors), then fine-tune all layers on the target-floor S scans.
  MAMLDNN      first-order MAML (Finn et al., ICML 2017) meta-trained on H: each task = one history floor, support =
               K random positions (<=R scans each), query = other scans of that floor; then adapt on S with the same
               inner-loop rule. (This is also the structure of MetaLoc-style localisation meta-learning.)
  PretrainDNN  pre-train on H only, no adaptation (zero-shot reference).
Hyper-parameters: fixed small grid chosen on SOURCE buildings in fit_source by leave-one-floor-out (same protocol
shape), never on the task building. Pre-training/meta-training depends only on (building, target floor, H) and is
cached across K and repeats of the same floor.
"""
import copy
import numpy as np
import torch
from torch import nn
from .task import Method

torch.set_num_threads(2)
ND = -105.0


def _x(rss):
    return torch.tensor((np.where(rss == 100, ND, rss) - ND) / 105.0, dtype=torch.float32)


class _Net(nn.Module):
    def __init__(self, a, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(a, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.f(x)


def _fit(net, X, Y, steps, lr, bs=256, seed=0):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (min(bs, len(X)),), generator=g)
        loss = ((net(X[idx]) - Y[idx]) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return net


def _norm(xy, lo, sc):
    return torch.tensor((xy - lo) / sc, dtype=torch.float32)


def _hs_aps(H, S):
    seen = (S['rss'] != 100).any(0)
    for d in H.values():
        seen |= (d['rss'] != 100).any(0)
    return np.flatnonzero(seen)


class _Base(Method):
    grid = ()

    def __init__(self, seed=0, hp=None):
        self.seed, self.hp, self._cache = seed, hp, {}

    def fit_source(self, sources):
        """choose hp on sources: for each source building, leave-one-floor-out with K=3 random positions as S,
        score = mean error on the held-out floor's other scans; pick the best grid entry"""
        if self.hp is not None or not self.grid:
            self.hp = self.hp or (self.grid[0] if self.grid else None); return
        rng = np.random.default_rng(self.seed); scores = {g: [] for g in self.grid}
        for src in sources[:2]:
            fl = sorted(src['floors'])
            for t in fl[:2]:
                H = {f: src['floors'][f] for f in fl if f != t}; T = src['floors'][t]
                P, inv = np.unique(T['xy'], axis=0, return_inverse=True); inv = inv.ravel()
                sp = rng.choice(len(P), min(3, len(P)), replace=False)
                srows = np.concatenate([np.flatnonzero(inv == k)[:3] for k in sp])
                qrows = rng.choice(np.flatnonzero(~np.isin(inv, sp)), min(300, int((~np.isin(inv, sp)).sum())), replace=False)
                S = dict(rss=T['rss'][srows], xy=T['xy'][srows])
                for g in self.grid:
                    pred = self._run(H, S, T['rss'][qrows], g, key=None)
                    scores[g].append(float(np.linalg.norm(pred - T['xy'][qrows], axis=1).mean()))
        self.hp = min(scores, key=lambda g: np.mean(scores[g])); self.hp_scores = {str(k): float(np.mean(v)) for k, v in scores.items()}

    def predict(self, task):
        key = (task.dataset, task.building, task.floor)
        return self._run(task.H, task.S, task.Q_rss, self.hp, key)

    def _prep(self, H, S):
        aps = _hs_aps(H, S)
        hx = np.vstack([d['rss'] for d in H.values()])[:, aps]; hy = np.vstack([d['xy'] for d in H.values()])
        allxy = np.vstack([hy, S['xy']]); lo = allxy.min(0); sc = np.maximum(allxy.max(0) - lo, 1.0)
        return aps, hx, hy, lo, sc


class FineTuneDNN(_Base):
    name = 'finetune_dnn'
    grid = ((3000, 1e-3, 100, 3e-4), (3000, 1e-3, 300, 3e-4), (3000, 1e-3, 100, 1e-3))   # pre steps, lr, ft steps, ft lr

    def _pre(self, H, S, hp, key):
        aps, hx, hy, lo, sc = self._prep(H, S)
        # Support can extend the coordinate bounds even when the AP axis is unchanged.
        ck = (key, tuple(aps), tuple(lo), tuple(sc), hp[:2]) if key else None
        if ck and ck in self._cache:
            return self._cache[ck]
        torch.manual_seed(self.seed); net = _Net(len(aps))
        _fit(net, _x(hx), _norm(hy, lo, sc), hp[0], hp[1], seed=self.seed)
        out = (net, aps, lo, sc)
        if ck:
            self._cache[ck] = out
        return out

    def _run(self, H, S, q_rss, hp, key):
        net0, aps, lo, sc = self._pre(H, S, hp, key)
        net = copy.deepcopy(net0)
        if hp[2] > 0 and len(S['xy']):
            _fit(net, _x(S['rss'][:, aps]), _norm(S['xy'], lo, sc), hp[2], hp[3], bs=64, seed=self.seed + 1)
        net.eval()
        with torch.no_grad():
            return net(_x(q_rss[:, aps])).numpy() * sc + lo


class PretrainDNN(FineTuneDNN):
    name = 'pretrain_dnn'
    grid = ((3000, 1e-3, 0, 0.0),)


class MAMLDNN(_Base):
    """first-order MAML; hp = (meta steps, meta lr, inner steps, inner lr, K_meta)"""
    name = 'maml_dnn'
    grid = ((1500, 1e-3, 5, 1e-2, 3), (1500, 1e-3, 10, 1e-2, 3), (1500, 1e-3, 5, 3e-3, 3))

    def _meta(self, H, S, hp, key):
        aps, hx, hy, lo, sc = self._prep(H, S)
        # Cache the exact coordinate system used by the trained network.
        ck = (key, tuple(aps), tuple(lo), tuple(sc), hp) if key else None
        if ck and ck in self._cache:
            return self._cache[ck]
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        net = _Net(len(aps)); opt = torch.optim.Adam(net.parameters(), lr=hp[1])
        floors = [(_x(d['rss'][:, aps]), _norm(d['xy'], lo, sc), d['xy']) for d in H.values() if len(d['xy']) > 10]
        for _ in range(hp[0]):
            X, Y, xy = floors[rng.integers(len(floors))]
            P, inv = np.unique(xy, axis=0, return_inverse=True); inv = inv.ravel()
            sp = rng.choice(len(P), min(hp[4], len(P) - 1), replace=False)
            srows = np.concatenate([np.flatnonzero(inv == k)[:3] for k in sp]); qpool = np.flatnonzero(~np.isin(inv, sp))
            qrows = rng.choice(qpool, min(64, len(qpool)), replace=False)
            fast = copy.deepcopy(net); fopt = torch.optim.SGD(fast.parameters(), lr=hp[3])
            for _ in range(hp[2]):
                l = ((fast(X[srows]) - Y[srows]) ** 2).sum(1).mean(); fopt.zero_grad(); l.backward(); fopt.step()
            lq = ((fast(X[qrows]) - Y[qrows]) ** 2).sum(1).mean(); fast.zero_grad(); lq.backward()
            for p, fp in zip(net.parameters(), fast.parameters()):          # first-order MAML meta-gradient
                p.grad = fp.grad.clone()
            opt.step(); opt.zero_grad()
        out = (net, aps, lo, sc)
        if ck:
            self._cache[ck] = out
        return out

    def _run(self, H, S, q_rss, hp, key):
        net0, aps, lo, sc = self._meta(H, S, hp, key)
        fast = copy.deepcopy(net0); fopt = torch.optim.SGD(fast.parameters(), lr=hp[3])
        X, Y = _x(S['rss'][:, aps]), _norm(S['xy'], lo, sc)
        fast.train()
        for _ in range(hp[2]):
            l = ((fast(X) - Y) ** 2).sum(1).mean(); fopt.zero_grad(); l.backward(); fopt.step()
        fast.eval()
        with torch.no_grad():
            return fast(_x(q_rss[:, aps])).numpy() * sc + lo
