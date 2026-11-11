"""Original 4 m aggregated full-H GP, restricted to SAH1 F2 <- F1/F3.

Reuses the original adapted_bank functions; no history fitting or operator net.
The old H-dependent binning, source parameters, M-PMC and mean reader remain
unchanged. S and Query RSS come only from Task; Query XY is never supplied.
An isolated child avoids collisions between the old data/run/inference modules
and the current benchmark. Requires the existing old LAW dependencies (pypmc).
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np


LEGACY_ROOT = Path('/home/panyushuo/projects/panyushuo/'
    'LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
PARAMETERS = 'results/uji_history_bank_v1_20261026/parameters.json'
CACHE = 'results/tampere_operator_confirm_v1_20261026/histories/SAH1_F2'
PARAMETERS_SHA = '4ba43ecbc988f9031e1345160be92f53326540c0bec5384906607aa2fdf0ff4d'
CODE_MODULES = ('calibration_relation check_support_integral data full_ap_study '
    'history_bank inference multi_history operator_prior reference_pmc run '
    'run_calibration_learning run_change run_change_control run_multi_history '
    'run_operator_prior run_operator_tampere source_weight_mixture support_integral '
    'transfer uji_probability_study').split()
CODE_SHA = '359e4f013fbcfc4cb910520ff1eee09c621600aaad8da74defc36037c5956f07'
# Read-only binding on 2026-11-11: full original CSV H rows (one-based IDs),
# XY float64, RSS float64, in original order. Both original and bench parsers agree.
H_SHA = {1: 'a9fd4df7828a63b2792be90f54809bc266c31d2280d363bf199c28375541d19c',
         3: 'c19aafb00304b8949097c741ae50911f39d379f7179408ac7540dab428eda2cc'}
FIELD_SHA = {1: '048c0f8dc09f9a1dd1e5e08c77b420fadf6906a143e98bb04a119dbc188bb173',
             3: '6ba47b2f6c271319de37ef0bf2b72a8a7d74fd1748a39d89fe5fa152dfa51362'}
J = {1: 189, 3: 179}


def _check_code(root):
    digest = hashlib.sha256()
    for name in CODE_MODULES:
        digest.update(name.encode() + b'\0' + (root / (name + '.py')).read_bytes())
    if digest.hexdigest() != CODE_SHA:
        raise ValueError('Original Tampere GP dependency version changed')
    if hashlib.sha256((root / PARAMETERS).read_bytes()).hexdigest() != PARAMETERS_SHA:
        raise ValueError('Original UJI B0 F0-F2 parameters changed')


class TampereLegacyGP:
    name = 'tampere_legacy_gp'

    def __init__(self, *, legacy_root=LEGACY_ROOT):
        self.legacy_root = Path(legacy_root).resolve()
        self.allowed_sources = None
        self.diagnostics = {}

    def fit_source(self, sources):
        self.allowed_sources = None
        allowed = {(s['dataset'], s['building']) for s in sources}
        if ('uji', 'B0') not in allowed or ('tampere', 'SAH1') in allowed:
            raise ValueError('Requires permitted UJI B0 and held-out Tampere SAH1')
        _check_code(self.legacy_root)
        self.allowed_sources = allowed
        return self

    def predict(self, task):
        self.diagnostics = dict(task_id=task.task_id, completed=False,
            identity='original_4m_aggregated_full_H_GP', history_fit=False,
            parameters_sha256=PARAMETERS_SHA, source_floors=['B0F0', 'B0F1', 'B0F2'],
            code_sha256=CODE_SHA, cache=str(self.legacy_root / CACHE),
            H_sha256=H_SHA, fields_sha256=FIELD_SHA,
            history_solver_status='not recorded in legacy cache; not re-fitted')
        try:
            if self.allowed_sources is None:
                raise ValueError('Call fit_source first')
            if ((task.dataset, task.building, task.floor) != ('tampere', 'SAH1', 2)
                    or set(task.H) != {1, 3} or task.K not in (1, 3, 5)
                    or task.meta.get('protocol') != 'static-resample-v2'):
                raise ValueError('This asset supports only static SAH1 F2, H=F1/F3, K=1/3/5')
            if list(task.ap_ids) != [f'tampere/SAH1/column{i+1}' for i in range(775)]:
                raise ValueError('Original SAH1 AP axis required')
            arrays = dict(srss=task.S['rss'], sxy=task.S['xy'], qrss=task.Q_rss)
            for floor in (1, 3):
                h = task.H[floor]
                digest = hashlib.sha256()
                for key, dtype in [('row_ids', '<i8'), ('xy', '<f8'), ('rss', '<f8')]:
                    digest.update(np.ascontiguousarray(h[key], dtype=dtype).tobytes())
                if digest.hexdigest() != H_SHA[floor]:
                    raise ValueError(f'Full raw H rows/XY/RSS differ from legacy binding: F{floor}')
                arrays[f'hxy{floor}'], arrays[f'hrss{floor}'] = h['xy'], h['rss']
            if (len(task.S['rss']) != task.K
                    or len(np.unique(task.S['xy'], axis=0)) != task.K):
                raise ValueError('Original Tampere comparison requires K distinct positions, R=1')
            with tempfile.TemporaryDirectory(prefix='tampere-legacy-gp-') as temporary:
                directory = Path(temporary)
                np.savez(directory / 'task.npz', **arrays)
                (directory / 'request.json').write_text(json.dumps(dict(
                    legacy_root=str(self.legacy_root), repeat=task.repeat)))
                try:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    str(directory)], check=True)
                finally:
                    if (directory / 'diagnostics.json').exists():
                        self.diagnostics.update(json.loads((directory / 'diagnostics.json').read_text()))
                prediction = np.load(directory / 'prediction.npy', allow_pickle=False)
            self.diagnostics['completed'] = True
            return prediction
        except Exception as error:
            self.diagnostics['error'] = repr(error)
            raise


def _predict_original(directory, diagnostics):
    request = json.loads((directory / 'request.json').read_text())
    root = Path(request['legacy_root'])
    _check_code(root)
    sys.path.insert(0, str(root))
    import torch
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    import run_operator_tampere as old

    with np.load(directory / 'task.npz', allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    base, obs = old.params()
    hxy = np.concatenate([data['hxy1'], data['hxy3']])
    origin = hxy.mean(0)
    histories = {}
    for floor in (1, 3):
        folder = root / CACHE / f'SAH1F{floor}'
        done = json.loads((folder / 'done.json').read_text())
        grouped = old.cells(data[f'hxy{floor}'] - origin, data[f'hrss{floor}'])
        if (not np.array_equal(origin, done['origin'])
                or len(grouped['positions']) != done['cells'] or done['cells'] != J[floor]):
            raise ValueError(f'Legacy H origin/4m grouping mismatch: F{floor}')
        paths = sorted(folder.glob('ap*.npz'))
        expected = np.flatnonzero((data[f'hrss{floor}'] != 100).any(0)).tolist()
        if sorted(int(p.stem[2:]) for p in paths) != expected:
            raise ValueError(f'Legacy detected AP coverage mismatch: F{floor}')
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode() + b'\0' + path.read_bytes())
        if digest.hexdigest() != FIELD_SHA[floor]:
            raise ValueError(f'Legacy historical field files changed: F{floor}')
        histories[floor] = {}
        for path in paths:
            field = old.load_history(path, base)
            if field is None or not np.array_equal(field.z.numpy(), grouped['positions']):
                raise ValueError(f'Legacy field coordinate binding mismatch: {path}')
            histories[floor][int(path.stem[2:])] = field

    support = old.group_scans(data['srss'], data['sxy'] - origin)
    sx = old.tensor(support['positions'])
    hpos = hxy - origin
    seen = ((data['hrss1'] != 100).any(0) | (data['hrss3'] != 100).any(0)
            | (support['stats']['k'].sum(0) > 0))
    aps = np.flatnonzero(seen).tolist()
    lo = np.minimum(hpos.min(0), sx.numpy().min(0))
    hi = np.maximum(hpos.max(0), sx.numpy().max(0))
    axes = [np.linspace(lo[d], hi[d], int(np.ceil((hi[d] - lo[d]) / 2)) + 1) for d in range(2)]
    grid = old.tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))
    hist = {a: [histories[f].get(a) for f in (1, 3)] for a in aps}
    stats = {key: torch.stack([old.ap_stats(support, a)[key] for a in aps])
             for key in ('n', 'k', 's1', 's2')}
    # Original adapted_bank assembly (run_operator_tampere.py:133-150).
    means, covs, residuals = [], [], []
    with torch.no_grad():
        for a in aps:
            mm, cc = [], []
            for h in hist[a]:
                prior = old.Prior(old.Transfer('identity'), h, dict(base, drift=0.))
                m, f = prior.features(sx)
                mm.append(m)
                cc.append(prior.covariance(sx, sx, f, f))
            if any(h is not None for h in hist[a]):
                residual = old.rbf(sx, sx, base['drift'], base['length'])
            else:
                cc = [torch.zeros_like(x) for x in cc]
                residual = old.rbf(sx, sx, base['amplitude'], base['length'])
            means.append(torch.stack(mm)); covs.append(torch.stack(cc)); residuals.append(residual)
    weights, trials = old.fit_two_source_weight(
        torch.stack(means), torch.stack(covs), torch.stack(residuals), stats, obs)
    diagnostics.update(origin=origin.tolist(), J=J, H_rows={'1': 1951, '3': 2047},
        APs=aps, grid_positions=len(grid), weights=weights, weight_trials=trials,
        seed_rule='261026 + 10000 * Task.repeat + original_AP_index',
        field_ess={}, output='original posterior mean, absolute XY')
    fields = {}
    with torch.no_grad():
        for a in aps:
            hs = hist[a]
            prior = (old.BankPrior(hs, weights, base) if any(h is not None for h in hs)
                     else old.BankPrior([], [], base))
            result = old.field_from_prior(prior, obs, sx, old.ap_stats(support, a), grid,
                261026 + 10000 * request['repeat'] + a, defensive_fraction=.1)
            fields[a] = result['mean'], result['variance']
            diagnostics['field_ess'][str(a)] = result['effective_samples']
        predictions = []
        for start in range(0, len(data['qrss']), 64):
            scans = old.tensor(data['qrss'][start:start + 64].astype(float))
            scores = scans.new_zeros((len(scans), len(grid)))
            for a in aps:
                mean, variance = fields[a]
                scores += old.observation_log_prob(scans[:, a, None], mean[None], variance[None], obs)
            predictions.append((torch.softmax(scores, 1) @ grid).numpy())
    low = sum(v is not None and v < 1000 for v in diagnostics['field_ess'].values())
    diagnostics.update(low_ess_fields=low, low_ess_fraction=low / len(aps),
                       numerically_reliable=low / len(aps) <= .02)
    np.save(directory / 'prediction.npy', np.concatenate(predictions) + origin)


if __name__ == '__main__':
    work = Path(sys.argv[1])
    diagnostics = {}
    try:
        _predict_original(work, diagnostics)
    finally:
        (work / 'diagnostics.json').write_text(json.dumps(diagnostics, allow_nan=False))
