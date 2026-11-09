"""Posthoc signed-cache analysis only; no fitting, inference or target selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
METHODS = ('GP', 'ordinary', 'full24', 'GP_ordinary_half', 'GP_full24_half', 'fusion')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root=ROOT):
    results = root/'results'
    sources = {}

    def read(path):
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    def arrays(path, expected):
        assert digest(path) == expected, path
        sources[str(path)] = expected
        return np.load(path)

    ordinary = read(results/'ordinary_context_target_20261109/complete.json')
    fusion = read(results/'position_fusion_20261109_continued/target/complete.json')
    tabdir = results/'building_information_tab_20261109'
    public = read(tabdir/'complete.json')
    manifest = read(results/'building_information_assets_20261109/manifest.json')
    for folder, complete in ((results/'ordinary_context_target_20261109', ordinary),
                             (results/'position_fusion_20261109_continued', fusion), (tabdir, public)):
        assert digest(folder/'freeze.json') == complete['freeze_sha256']
        sources[str(folder/'freeze.json')] = complete['freeze_sha256']
    assets_complete = read(results/'building_information_assets_20261109/complete.json')
    assert digest(results/'building_information_assets_20261109/manifest.json') == assets_complete['manifest_sha256']
    events = {e['id']: e for e in manifest['events']}
    assert set(events) == {r['event'] for r in ordinary['records']} == {r['event'] for r in fusion['records']}
    assert len(events) == 54
    assert {b: len({e['floor'] for e in events.values() if e['building'] == b}) for b in ('B1','B2')} == {'B1':4,'B2':5}
    fused = {r['event']: r for r in fusion['records']}
    publics = {r['event']: r for r in public['records']
               if r['method'] == 'public' and r['information'] == 'HS' and r['K'] == 1}
    samples = {}
    for r in ordinary['records']:
        eid = r['event']; e = events[eid]; f = fused[eid]
        fp = Path(f['prediction_path'])
        fs = fp.with_name(fp.name.replace('_unscored', '_scored'))
        with arrays(Path(r['prediction_path']), r['prediction_sha256']) as a, \
             arrays(Path(r['scored_path']), r['scored_sha256']) as b, \
             arrays(fp, f['prediction_sha256']) as c, arrays(fs, f['scored_sha256']) as d:
            assert np.array_equal(a['query_rows'], e['query_rows'])
            assert np.array_equal(c['query_rows'], a['query_rows'])
            assert np.array_equal(b['truth'], d['truth'])
            truth = b['truth']; pred = {m: a[m] for m in METHODS if m != 'fusion'}
            pred['fusion'] = c['fusion']
            assert np.array_equal(a['GP'], c['GP']) and np.array_equal(a['full24'], c['full24'])
            for m in METHODS:
                stored = d[m] if m == 'fusion' else b[m]
                assert np.allclose(np.linalg.norm(pred[m]-truth, axis=1), stored, rtol=0, atol=1e-12)
            if e['K'] == 1:
                pub = publics[eid]
                with arrays(tabdir/pub['path'], pub['sha256']) as p:
                    assert np.array_equal(p['query_rows'], a['query_rows'])
                    assert np.array_equal(p['truth'], truth)
                    pred['public'] = p['projected']
                    assert np.allclose(np.linalg.norm(pred['public']-truth, axis=1), p['errors'], rtol=0, atol=1e-12)
                    pred['GP_public_half'] = .5*(pred['GP']+pred['public'])
            xy, group = np.unique(truth, axis=0, return_inverse=True)
            errors = {m: np.array([np.linalg.norm(v-truth, axis=1)[group == i].mean()
                                   for i in range(len(xy))]) for m, v in pred.items()}
            samples[eid] = dict(event=e, xy=xy, errors=errors)

    def weight(s):
        e = s['event']
        return .5/(4 if e['building'] == 'B1' else 5)/2/len(s['xy'])

    def aggregate(subset, methods):
        out = {}
        for m in methods:
            selected = [s for s in subset if m in s['errors']]
            v = np.concatenate([s['errors'][m] for s in selected])
            w = np.concatenate([np.full(len(s['xy']), weight(s)) for s in selected])
            w /= w.sum(); order = np.argsort(v)
            out[m] = dict(mean=float(w@v), position_mean_p90=float(v[order][np.searchsorted(np.cumsum(w[order]), .9)]))
        gp = out['GP']['mean']
        for m in out:
            out[m]['difference_vs_GP'] = out[m]['mean']-gp
            out[m]['vs_GP'] = 'win' if out[m]['mean'] < gp else ('tie' if out[m]['mean'] == gp else 'loss')
        return out

    all_samples = list(samples.values())
    per_k = {str(k): aggregate([s for s in all_samples if s['event']['K'] == k],
                 (*METHODS, 'public', 'GP_public_half') if k == 1 else METHODS) for k in (1, 3, 5)}
    floors = sorted({s['event']['floor'] for s in all_samples})
    floor_scores = {f: {str(k): aggregate([s for s in all_samples if s['event']['floor'] == f and s['event']['K'] == k], (*METHODS, 'public', 'GP_public_half') if k == 1 else METHODS)
                       for k in (1, 3, 5)} for f in floors}
    building_scores = {b: {str(k): aggregate([s for s in all_samples if s['event']['building'] == b and s['event']['K'] == k], (*METHODS, 'public', 'GP_public_half') if k == 1 else METHODS)
                          for k in (1, 3, 5)} for b in ('B1', 'B2')}
    for f in floors:
        floor_scores[f]['all_K'] = aggregate([s for s in all_samples if s['event']['floor'] == f], METHODS)
    for b in ('B1', 'B2'):
        building_scores[b]['all_K'] = aggregate([s for s in all_samples if s['event']['building'] == b], METHODS)
    paired = {}
    for lo, hi in ((1, 3), (3, 5), (1, 5)):
        changes = {m: [] for m in METHODS}; weights = []; contracts = True
        for s in all_samples:
            e = s['event']
            if e['K'] != lo:
                continue
            other = samples[f"{e['floor']}_{e['direction']}_K{hi}"]; q = other['event']
            contracts &= set(e['support_rows']) <= set(q['support_rows']) and e['query_rows'] == q['query_rows'] and np.array_equal(s['xy'], other['xy'])
            for m in METHODS:
                changes[m].extend(other['errors'][m]-s['errors'][m])
            weights.extend([weight(s)]*len(s['xy']))
        if not contracts:
            paired[f'{lo}->{hi}'] = dict(eligible=False, reason='Support nesting or identical Query contract failed; no paired claim')
            continue
        w = np.array(weights); w /= w.sum()
        paired[f'{lo}->{hi}'] = dict(eligible=True, methods={m: dict(mean_change=float(w@v), worse_position_fraction=float(w@(np.array(v)>0))) for m,v in changes.items()})
    monotonic = {f: {m: all(floor_scores[f][str(a)][m]['mean'] >= floor_scores[f][str(b)][m]['mean'] for a,b in ((1,3),(3,5))) for m in METHODS} for f in floors}
    decomposition = {}
    for label, subset in [('all', all_samples)]+[(str(k), [s for s in all_samples if s['event']['K'] == k]) for k in (1,3,5)]:
        decomposition[label] = {}
        for tab in ('ordinary','full24','public'):
            ss = [s for s in subset if tab in s['errors']]
            if not ss or (tab == 'public' and label == 'all'):
                continue
            vals = []; ww = []
            for s in ss:
                g,t,h = (s['errors'][x] for x in ('GP',tab,f'GP_{tab}_half'))
                strength = .5*(g-t); cancel = .5*(g+t)-h
                assert cancel.min() >= -1e-12
                assert np.allclose(g-h,strength+cancel,rtol=0,atol=1e-12)
                vals.extend(zip(g-h,strength,cancel)); ww.extend([weight(s)]*len(g))
            w = np.array(ww); w /= w.sum()
            decomposition[label][tab] = dict(zip(('total_half_gain','single_branch_strength','coordinate_cancellation'),map(float,w@np.array(vals))))
    practical = []
    for s in all_samples:
        e = s['event']
        if e['K'] != 1:
            continue
        other = samples[f"{e['floor']}_{e['direction']}_K5"]
        assert set(e['support_rows']) <= set(other['event']['support_rows'])
        assert e['query_rows'] == other['event']['query_rows'] and np.array_equal(s['xy'],other['xy'])
        practical.append(dict(event=e, xy=s['xy'], errors={'GP':other['errors']['GP'], 'GP_ordinary_half':s['errors']['GP_ordinary_half']}))
    practical_comparison = dict(overall=aggregate(practical,('GP','GP_ordinary_half')),
        by_floor={f:aggregate([s for s in practical if s['event']['floor']==f],('GP','GP_ordinary_half')) for f in floors},
        scope='Fixed ordinary half K1 versus GP K5 on identical Query, nested Support: 1 position x 3 scans versus 5 x 3 in this development protocol only; no general saving or no-loss guarantee')
    # The practical comparison requests means and wins only.
    for block in [practical_comparison['overall'],*practical_comparison['by_floor'].values()]:
        for row in block.values(): row.pop('position_mean_p90')
    out = dict(scope='Development/posthoc existing-cache description; no inference, selection, bootstrap or statistical guarantee',
               position_unit='Within each K, mean scan error at each XY. Overall mixes repeated position-event-budget entries across K, not independent locations.',
               weighting='Equal building, floor, direction, K, then position; scans at same XY averaged first',
               overall=aggregate(all_samples,METHODS),by_K=per_k,by_floor=floor_scores,by_building=building_scores,
               paired_K_changes=paired,floor_K_monotonic=monotonic,half_gain_decomposition=decomposition,
               decomposition_scope='Development/posthoc error cancellation is descriptive, not learning-specific contribution or deployable Query selection',
               K1_half_vs_K5_GP=practical_comparison,sources=sources)
    for m in METHODS[:-1]:
        assert abs(out['overall'][m]['mean']-ordinary['summary'][m]['mean']) <= 1e-9
    out['sources'][str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    target = results/'ordinary_context_target_20261109/system_stability.json'
    target.write_text(json.dumps(out,indent=2))
    return out


if __name__ == '__main__':
    run()
