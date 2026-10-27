"""T1 post-hoc discrimination on frozen initial, g-free S weights. No fitting."""
import csv
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from inference import observation_log_prob
from uji_probability_study import load_raw

OUT = ROOT / 'results/b1f1_official_diag_20261026'
CACHE = ROOT / 'results/uji_conditional_operator_v1_20261026/evaluation/initial'
OLD = ROOT / 'results/uji_history_bank_v1_20261026'
METHODS = ['adapted_bank', 'operator_bank']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def position_weights(xy):
    _, inverse, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inverse.ravel()])


def selected(floor, rows):
    lookup = {int(r): i for i, r in enumerate(floor['row_ids'])}
    ix = [lookup[r] for r in rows]
    return floor['rss'][ix], floor['xy'][ix]


def detection_rates(rss, xy, peak):
    near = np.linalg.norm(xy - peak, axis=1) <= 10
    if not near.any():
        return None
    w = position_weights(xy[near])
    return dict(rate=float(w @ (rss[near] != 100)), scans=int(near.sum()),
                positions=len(np.unique(xy[near], axis=0)))


def main():
    assert socket.gethostname() == 'ray-web', 'Numerical work stays on ray-web'
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'summary.json').exists(), 'Preserve existing delivery'
    training = load_raw('trainingData.csv', {'B1F0', 'B1F1', 'B1F3'})
    official = load_raw('validationData.csv', {'B1F1'})['B1F1']
    obs = json.loads((OLD / 'parameters.json').read_text())['obs']
    events = [e for e in json.loads((OLD / 'manifest.json').read_text())['events'] if e['floor'] == 'B1F1']
    summaries, detection, provenance = [], [], {}
    for event in events:
        name = event['id']; source = CACHE / name
        adaptation = json.loads((source / 'adaptation.json').read_text())
        assert adaptation['weights'] == adaptation['baseline_weights']
        assert 'g-free' in adaptation['policy']
        parts = [selected(training['B1F1'], event['internal_rows']), selected(official, event['official_rows'])]
        rss = np.concatenate([p[0] for p in parts]); xy_raw = np.concatenate([p[1] for p in parts])
        with np.load(source / 'predictions.npz') as stored:
            origin = stored['origin']; xy = xy_raw - origin
            np.testing.assert_array_equal(xy, stored['query_xy'])
            predictions = {m: stored[m] for m in METHODS}
        bounds = np.array(event['grid_bounds']) - origin
        axes = [np.linspace(bounds[0, k], bounds[1, k], int(np.ceil((bounds[1, k]-bounds[0, k])/2))+1) for k in range(2)]
        grid = np.stack(np.meshgrid(*axes), -1).reshape(-1, 2)
        grid_error, true_ix = cKDTree(grid).query(xy)
        ap = event['ap_columns']; fields = {m: [[], []] for m in METHODS}
        paths = [source / 'adaptation.json', source / 'predictions.npz', CACHE / 'freeze.json', OLD / 'parameters.json', OLD / 'manifest.json', ROOT / 'inference.py']
        for a in ap:
            path = source / f'ap{a}.npz'; paths.append(path)
            with np.load(path) as f:
                for m in METHODS:
                    fields[m][0].append(f[m+'_mean']); fields[m][1].append(f[m+'_variance'])
        fields = {m: (torch.tensor(np.array(f[0])), torch.tensor(np.array(f[1]))) for m, f in fields.items()}
        provenance[name] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
        scores = {}
        # Exact reuse of original likelihood; compute unique RSS values per AP once.
        with torch.no_grad():
            for m in METHODS:
                score = torch.zeros((len(rss), len(grid)))
                for j, a in enumerate(ap):
                    values, inverse = np.unique(rss[:, a], return_inverse=True)
                    likelihood = observation_log_prob(torch.tensor(values[:, None]), fields[m][0][j][None], fields[m][1][j][None], obs)
                    score += likelihood[torch.tensor(inverse)]
                estimate = score.softmax(1) @ torch.tensor(grid)
                np.testing.assert_allclose(estimate.numpy(), predictions[m], rtol=0, atol=1e-9)
                scores[m] = score
                print(json.dumps(dict(event=name, stage='scores', method=m)), flush=True)
            distances = torch.tensor(np.linalg.norm(grid[None] - xy[:, None], axis=-1))
            near = distances <= 10
            row = torch.arange(len(rss)); true_index = torch.tensor(true_ix)
            competitor = scores['operator_bank'].masked_fill(near, -torch.inf).argmax(1)
            metrics = {}; per_ap = {}
            detected = torch.tensor(rss[:, ap] != 100)
            for m in METHODS:
                s = scores[m]; truth = s[row, true_index]
                best_far = s.masked_fill(near, -torch.inf).max(1).values
                logp = s.log_softmax(1); prob = logp.exp()
                means, variances = fields[m]
                contributions = []
                for j, a in enumerate(ap):
                    r = torch.tensor(rss[:, a])
                    t = observation_log_prob(r, means[j, true_index], variances[j, true_index], obs)
                    c = observation_log_prob(r, means[j, competitor], variances[j, competitor], obs)
                    contributions.append(t-c)
                contribution = torch.stack(contributions, 1)
                fixed = s[row, true_index] - s[row, competitor]
                residual = (contribution.sum(1) - fixed).abs().max().item()
                assert residual < 1e-9, residual
                per_ap[m] = contribution.numpy()
                metrics[m] = dict(true_loglik=truth.numpy(), own_best_far_margin=(truth-best_far).numpy(),
                    fixed_operator_far_margin=fixed.numpy(), missing_margin=(contribution * ~detected).sum(1).numpy(),
                    detected_margin=(contribution * detected).sum(1).numpy(),
                    mass_within_10=(prob*near).sum(1).numpy(), true_position_logprob=logp[row, true_index].numpy(),
                    mde=np.linalg.norm(predictions[m] - xy, axis=1),
                    true_grid_distance=grid_error)
            ni = len(parts[0][0]); split_labels = np.array(['internal']*ni + ['official']*(len(rss)-ni))
            for split in ['internal', 'official']:
                keep = split_labels == split; w = position_weights(xy[keep])
                result = dict(event=name, split=split, scans=int(keep.sum()), positions=len(np.unique(xy[keep],axis=0)), methods={})
                for m in METHODS:
                    result['methods'][m] = {k: float(w @ v[keep]) for k,v in metrics[m].items()}
                    result['methods'][m]['positive_own_margin_fraction'] = float(w @ (metrics[m]['own_best_far_margin'][keep] > 0))
                delta = {k: metrics['operator_bank'][k]-metrics['adapted_bank'][k] for k in metrics[METHODS[0]]}
                result['operator_minus_baseline'] = {k: float(w @ v[keep]) for k,v in delta.items()}
                deteriorated = delta['fixed_operator_far_margin'][keep] < 0
                result['margin_worsened_position_weight_fraction'] = float(w @ deteriorated)
                result['reconstruction_max_abs_error'] = float(np.max(np.abs(delta['fixed_operator_far_margin']-delta['missing_margin']-delta['detected_margin'])))
                summaries.append(result)
            records=[]
            for i in range(len(rss)):
                r = dict(row_id=int((event['internal_rows']+event['official_rows'])[i]), split=str(split_labels[i]),
                    x=float(xy_raw[i,0]), y=float(xy_raw[i,1]), true_grid_index=int(true_ix[i]),
                    operator_far_index=int(competitor[i]), operator_far_distance=float(distances[i,competitor[i]]))
                for m in METHODS:
                    r.update({m+'_'+k:float(v[i]) for k,v in metrics[m].items()})
                records.append(r)
            with (OUT / (name+'_queries.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            np.savez_compressed(OUT/(name+'_ap_margins.npz'),ap_columns=ap,detected=detected.numpy(),
                true_grid_index=true_ix,operator_far_index=competitor.numpy(),grid=grid,origin=origin,**per_ap)
            # Observable group: >=5 dB positive posterior correction at the H peak.
            # Peak is maximum detected position-mean RSS in the two raw histories.
            for j,a in enumerate(ap):
                peaks=[]
                for h in ['B1F0','B1F3']:
                    f=training[h];count=f['stats']['k'][:,a];ok=count>0
                    if ok.any():
                        ids=np.flatnonzero(ok); ix=ids[np.argmax(f['stats']['s1'][ok,a]/count[ok])]
                        peaks.append((float(f['stats']['s1'][ix,a]/count[ix]),f['positions'][ix]))
                if not peaks:continue
                peak=max(peaks,key=lambda p:p[0])[1]
                ix=int(cKDTree(grid).query(peak-origin)[1])
                correction=float(fields['operator_bank'][0][j,ix]-fields['adapted_bank'][0][j,ix])
                ri=detection_rates(parts[0][0][:,a],parts[0][1],peak)
                ro=detection_rates(parts[1][0][:,a],parts[1][1],peak)
                # Exact shared coordinates, equal positions and then scans within positions.
                pi=np.unique(parts[0][1],axis=0);po=np.unique(parts[1][1],axis=0)
                common=np.array([p for p in pi if np.any(np.all(po==p,axis=1))]).reshape(-1,2)
                ci=np.array([any(np.all(common==p,axis=1)) for p in parts[0][1]]) if len(common) else np.zeros(len(parts[0][1]),bool)
                co=np.array([any(np.all(common==p,axis=1)) for p in parts[1][1]]) if len(common) else np.zeros(len(parts[1][1]),bool)
                detection.append(dict(event=name,ap_column=a,peak_raw=peak.tolist(),posterior_correction_at_H_peak=correction,
                    group='positive_correction_ge5' if correction>=5 else 'other_H_observed',internal=ri,official=ro,
                    exact_matched_internal=detection_rates(parts[0][0][ci,a],parts[0][1][ci],peak),
                    exact_matched_official=detection_rates(parts[1][0][co,a],parts[1][1][co],peak)))
        print(json.dumps(dict(event=name,stage='done')),flush=True)
    aggregate={}
    for split in ['internal','official']:
        rows=[r for r in summaries if r['split']==split]
        aggregate[split]={m:{k:float(np.mean([r['methods'][m][k] for r in rows])) for k in rows[0]['methods'][m]} for m in METHODS}
        aggregate[split]['operator_minus_baseline']={k:float(np.mean([r['operator_minus_baseline'][k] for r in rows])) for k in rows[0]['operator_minus_baseline']}
    detection_summary={}
    for group in ['positive_correction_ge5','other_H_observed']:
        subset=[r for r in detection if r['group']==group and r['internal'] is not None and r['official'] is not None]
        matched=[r for r in subset if r['exact_matched_internal'] is not None and r['exact_matched_official'] is not None]
        detection_summary[group]=dict(ap_event_pairs_with_both_neighborhoods=len(subset),
            internal_mean_AP_rate=float(np.mean([r['internal']['rate'] for r in subset])) if subset else None,
            official_mean_AP_rate=float(np.mean([r['official']['rate'] for r in subset])) if subset else None,
            exact_matched_ap_event_pairs=len(matched),
            exact_matched_internal_mean_AP_rate=float(np.mean([r['exact_matched_internal']['rate'] for r in matched])) if matched else None,
            exact_matched_official_mean_AP_rate=float(np.mean([r['exact_matched_official']['rate'] for r in matched])) if matched else None)
    write(OUT/'detection.json',dict(summary=detection_summary,records=detection))
    write(OUT/'summary.json',dict(created_unix=time.time(),scope=__doc__,script_sha256=digest(Path(__file__)),
        cache=str(CACHE),obs=obs,aggregation='Equal scans within position; equal positions within event; equal events. AP rate summary additionally equal AP-event pairs.',
        margin='loglik(nearest true grid)-max loglik(candidate distance >10 raw units). Fixed decomposition uses operator best-far for BOTH methods.',
        limits='Post-hoc labels used only for diagnosis. Detected term includes both RSS density and detection factor. No AP installation truth; unmatched detection rates cannot exclude spatial sampling, devices or temporal acquisition changes. No causal proof or independent validation. s=0 not tested here.',
        aggregate=aggregate,events=summaries,detection_summary=detection_summary,provenance=provenance))


if __name__=='__main__':
    main()
