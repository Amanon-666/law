"""Post-hoc cache-only G1/G4 diagnosis; never fits or selects a model on Q."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

import run_k_curve as curve
from inference import observation_log_prob
from run import tensor


OUT = curve.ROOT / 'results/base_path_audit_20261027'


def interval(values):
    """Conditional-on-corpus, normal-draw sensitivity interval, not OOD evidence."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not n:
        return dict(n=0, mean=None, ci95=None)
    mean = float(x.mean())
    half = float(t.ppf(.975, n-1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, ci95=[mean-half, mean+half] if half is not None else None,
                draws=x.tolist(), assumption='Independent randomized draws conditional on fixed corpus; normal draw-mean differences. Three draws cannot verify normality; exploratory only.')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    frozen = curve.verify()
    manifest = curve.manifest()
    OUT.mkdir(exist_ok=True)
    protocol = dict(UTC=datetime.now(timezone.utc).isoformat(), posthoc=True,
        script_sha256=curve.digest(Path(__file__)), source_freeze_sha256=curve.digest(curve.OUT/'freeze.json'),
        questions=['Does the posterior place mass near truth, or is the mean readout the failure?',
                   'At the SAME wrong competitor, do detected or missing APs oppose truth?',
                   'Are the nested-K differences distinguishable from draw variability?'],
        rules=['No refitting, no hyperparameter search, no change to frozen official scores.',
               'Q coordinates only diagnose. Nearest truth grid is a discretization diagnostic, not a deployable predictor.',
               'Competitor is full-score maximum outside radius 10 in each dataset native units; threshold is diagnostic, not a tuned model.',
               'MAP is a fixed readout diagnostic; no choosing mean or MAP using Q.',
               'Keep all events and explicit missing predictions; do not treat failures as wins.',
               'CI unit is randomized S/Q draw, not AP or query; no pooling distance units/buildings.',
               'CIs are pointwise, exploratory t intervals with only three draws; cannot certify G1/G4 or generalization.'],
        CI_reference='https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm')
    write(OUT/'protocol.json', protocol)
    events = []
    for meta in manifest['events']:
        dest = curve.OUT/meta['id']
        row = dict(id=meta['id'], building=meta['building'], target=meta['target'],
                   K=meta['K'], seed=meta['seed'], methods={})
        adaptation = curve.read(dest/'adaptation.json')
        row.update(weights=adaptation['weights'], intercept_std=adaptation['sigma']['intercept_std'])
        task = curve.episodes.build_task(meta)
        xy, rss = task['Q']['xy'], task['Q']['rss']
        points = curve.grid(meta).numpy()
        distances = np.linalg.norm(xy[:, None]-points[None], axis=-1)
        nearest = distances.argmin(1)
        row['query_scan_ids'] = task['Q']['scan_ids'].tolist()
        row['distance_to_S'] = np.linalg.norm(xy[:, None]-task['S']['xy'][None], axis=-1).min(1).tolist()
        for name in ('H_base', 'GP_S'):
            path = dest/name/'predictions.npz'
            if not path.exists():
                row['methods'][name] = dict(status='missing_prediction')
                continue
            with np.load(path, allow_pickle=False) as a:
                scores, predictions = a['scores'], a['predictions']
                np.testing.assert_allclose(a['query_xy'], xy, atol=1e-10)
                assert np.array_equal(a['query_scan_ids'], task['Q']['scan_ids'])
            probabilities = torch.softmax(tensor(scores), 1).numpy()
            np.testing.assert_allclose(probabilities@points, predictions, atol=1e-8)
            best = scores.argmax(1)
            far = np.where(distances > 10, scores, -np.inf).argmax(1)
            assert (distances[np.arange(len(xy)), far] > 10).all()
            # Recompute only the two queried candidate locations/AP; no field inference.
            parts = np.zeros((len(xy), 2, 2))  # query, truth/far, detected/missing
            for column, ap in enumerate(meta['AP_indices']):
                with np.load(dest/name/f'ap{ap}.npz', allow_pickle=False) as a:
                    indexes = np.stack((nearest, far), axis=1)
                    mean, variance = a['mean'][indexes], a['variance'][indexes]
                logs = observation_log_prob(tensor(rss[:, column, None]), tensor(mean),
                                            tensor(variance), manifest['obs']).numpy()
                group = (rss[:, column] == 100).astype(int)
                for q in range(len(xy)):
                    parts[q, :, group[q]] += logs[q]
            expected = np.stack((scores[np.arange(len(xy)), nearest], scores[np.arange(len(xy)), far]), 1)
            np.testing.assert_allclose(parts.sum(-1), expected, rtol=1e-10, atol=1e-8)
            mean_errors = np.linalg.norm(predictions-xy, axis=1)
            official = curve.read(dest/f'{name}.json')
            np.testing.assert_allclose(mean_errors, official['errors'], atol=1e-9)
            row['methods'][name] = dict(status='complete', predictions_sha256=curve.digest(path),
                mean_errors=mean_errors.tolist(), map_errors=distances[np.arange(len(xy)), best].tolist(),
                mean_mde=float(mean_errors.mean()), map_mde=float(distances[np.arange(len(xy)), best].mean()),
                top_probability=probabilities.max(1).tolist(),
                effective_grid_points=np.exp(-(probabilities*np.log(np.maximum(probabilities,1e-300))).sum(1)).tolist(),
                mass_within10=(probabilities*(distances<=10)).sum(1).tolist(),
                nearest_truth_grid_distance=distances[np.arange(len(xy)), nearest].tolist(),
                truth_minus_far_margin=(parts[:,0].sum(1)-parts[:,1].sum(1)).tolist(),
                detected_margin=(parts[:,0,0]-parts[:,1,0]).tolist(),
                missing_margin=(parts[:,0,1]-parts[:,1,1]).tolist(),
                margin_far_xy=points[far].tolist())
        for name in ('WKNN_HS','WKNN_S','H_support'):
            path = dest/f'{name}.json'
            row['methods'][name] = dict(status='complete', **curve.read(path)) if path.exists() else dict(status='missing_prediction')
        events.append(row)
        write(OUT/f'{meta["id"]}.json', row)
        print(json.dumps(dict(event=meta['id'], complete=True)), flush=True)
    groups = []
    for building in sorted({e['building'] for e in events}):
        for k in sorted({e['K'] for e in events}):
            rows = [e for e in events if e['building']==building and e['K']==k]
            comparisons = {}
            for other in ('GP_S','WKNN_HS','H_support'):
                delta=[]
                for e in rows:
                    base, ref = e['methods']['H_base'], e['methods'][other]
                    if base['status']=='complete' and ref['status']=='complete':
                        ref_mde=ref['mean_mde'] if other=='GP_S' else ref['mde']
                        delta.append(base['mean_mde']-ref_mde)
                comparisons['H_base_minus_'+other]=interval(delta)
            comparisons['MAP_minus_mean']=interval([e['methods']['H_base']['map_mde']-e['methods']['H_base']['mean_mde'] for e in rows if e['methods']['H_base']['status']=='complete'])
            groups.append(dict(building=building,K=k,comparisons=comparisons))
    write(OUT/'summary.json', dict(protocol=protocol, events=events, groups=groups,
          complete=True, model_fits=0, downloads=0, failed_events_not_imputed=True))


if __name__ == '__main__':
    run()
