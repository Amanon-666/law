"""B2F4 K3/r0 landscape export; CPU, existing caches only, no fitting."""
import copy
import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from paper_bench_20261111.extras import ROOT, ReFi, cache_only
from law_bench.data import load
from law_bench.protocol import make_tasks
from law_bench import ours

def main():
    os.nice(10)
    out = ROOT/'results/paper_landscape_20261111'
    out.mkdir(exist_ok=True)
    buildings, aps = load('uji')
    task, truth = make_tasks('uji', 'B2', buildings['B2'], aps['B2'], 4,
                             budgets=(1,3,5), repeat=0)[1]
    old = json.loads((ROOT/'results/paper_uji_B2_20261111/events/refi/uji/B2/F4/K3/r0.json').read_text())
    assert old['query_row_indices'] == truth['row_indices'].tolist()
    model = ReFi(cache_dir=ROOT/'results/paper_refi_models_20261111')
    cache_only(model, [dict(dataset='uji', building=b, floors=buildings[b], ap_ids=aps[b]) for b in ('B0','B1')])
    refi = model.translator.prepare({f'B2F{f}':d for f,d in task.H.items()}, task.S, 'B2F4')
    xy = refi.cand_xy
    maps = ours.make_maps(task.H)
    n = len(np.unique(np.concatenate([v[0] for v in maps.values()]), axis=0))
    # Existing naive field defaults (64+25 mRSS variance), no new source fitting.
    # Keep empirical S candidates identical, replace only historical distributions.
    field = ours.naive_field(maps, 4, refi.aps, xy[:n])
    copied = copy.deepcopy(refi)
    for name in ('p','m','s2'):
        getattr(copied,name)[:n] = field[name]
    outputs = {k:p.predict(task.Q_rss) for k,p in [('copied',copied),('refi',refi)]}
    errors = {k:np.linalg.norm(v['projected']-truth['xy'],axis=1) for k,v in outputs.items()}
    np.testing.assert_allclose(errors['refi'], old['scores']['scan_errors'], rtol=1e-7, atol=1e-7)
    for v in outputs.values():
        np.testing.assert_allclose(logsumexp(v['log_position_mass'],axis=1), 0, atol=1e-10)
    eligible = np.flatnonzero(errors['copied']>10)
    selected = []
    for q in eligible[np.argsort(errors['refi'][eligible],kind='stable')]:
        if all(not np.array_equal(truth['xy'][q], truth['xy'][j]) for j in selected):
            selected.append(int(q))
        if len(selected)==2:
            break
    assert len(selected)==2
    worst = np.minimum(errors['copied'],errors['refi']).copy()
    for q in selected:
        worst[np.all(truth['xy']==truth['xy'][q],axis=1)] = -np.inf
    selected.append(int(np.argmax(worst)))
    def save(name, columns, rows):
        with (out/name).open('w',newline='') as f:
            w=csv.writer(f); w.writerow(columns); w.writerows(rows)
    save('candidates.csv',['x','y'],xy)
    save('support.csv',['x','y'],np.unique(task.S['xy'],axis=0))
    save('footprint.csv',['x','y'],np.unique(buildings['B2'][4]['xy'],axis=0))
    rows=[]
    for i,q in enumerate(selected):
        pred={k:v['projected'][q] for k,v in outputs.items()}
        rows.append([int(truth['row_indices'][q]),*truth['xy'][q],*pred['copied'],*pred['refi'],errors['copied'][q],errors['refi'][q]])
        for k,v in outputs.items():
            save(f'q{i}_{k}.csv',['x','y','logmass'],np.column_stack([xy,v['log_position_mass'][q]]))
    save('queries.csv',['qid','true_x','true_y','map_copied_x','map_copied_y','map_refi_x','map_refi_y','err_copied','err_refi'],rows)
    peaks=[]
    for i,q in enumerate(selected):
        for k,v in outputs.items():
            p=xy[v['log_position_mass'][q].argmax()]
            peaks.append([i,k,*p,float(np.linalg.norm(p-truth['xy'][q]))])
    save('peaks.csv',['query_file_index','method','peak_x','peak_y','peak_error'],peaks)
    (out/'metadata.json').write_text(json.dumps(dict(task_id=task.task_id,source_buildings=model.sources,
        query_scan_indices=selected,query_row_ids=[r[0] for r in rows],
        selection='Two lowest ReFi scan errors among copied>10m at distinct true positions, then largest min(copied,refi) error among remaining scans; post-hoc illustration, not representative performance',
        coordinates='map_* columns are original reader posterior-mean predictions; peaks.csv contains argmax',
        copied='Existing naive_field nearest floor B2F3; variance=64+25; identical empirical S candidates and original frozen reader',
        duplicates='H/S coordinate duplicates retained as in original reader',
        validation='Query rows and all ReFi scan errors match original paper task; log masses normalized'),indent=2))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    main()
