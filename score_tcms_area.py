"""Post-hoc information-parity correction, preserving original TC predictions."""
import json
import socket
import time

import numpy as np

from full_ap_study import metrics
from run_multi_history import digest, write
from run_tcms_history_bank import OUT, BANK, VARIANTS


def main():
    if socket.gethostname()!='ray-web': raise RuntimeError('Server only')
    rule=OUT/'area_comparison_amendment.json'
    if not rule.exists():
        write(rule,dict(created_unix=time.time(),post_hoc=True,
            reason='LAW enforces a historical-H-plus-S rectangle; coordinate regressors must also receive this allowed-area constraint',
            operation='Euclidean projection onto the SAME frozen rectangle for EVERY TC case and variant',
            query_used_for_rule=False,extra_parameters=0,raw_predictions_preserved=True,
            qualification='This is a corrected bounded-area comparison, not original paper output or preregistered initial readout'))
    manifest=json.loads((OUT/'manifest.json').read_text());rows=[];diagnostics=[]
    for variant in VARIANTS:
        for meta in manifest['events']:
            directory=OUT/variant/meta['id']
            complete=json.loads((directory/'completed.json').read_text())
            path=directory/'predictions.npz'
            if digest(path)!=complete['prediction_sha256']: raise RuntimeError('Changed raw prediction')
            with np.load(path) as a:
                pred,xy,n=a['prediction'],a['query_xy'],int(a['internal_count'])
                support=a['support_xy']
            if ((xy[:,None]==support[None]).all(-1)).any(): raise RuntimeError('S/Q position overlap')
            lo,hi=np.array(meta['grid_bounds']);projected=np.clip(pred,lo,hi)
            np.savez_compressed(directory/'predictions_area_matched.npz',prediction=projected,query_xy=xy,internal_count=n)
            for split,sl in [('internal',slice(0,n)),('official',slice(n,None))]:
                _,g,count=np.unique(xy[sl],axis=0,return_inverse=True,return_counts=True);g=g.ravel()
                sw=1/(len(count)*count[g])
                for readout,p in [('raw',pred),('area_matched',projected)]:
                    rows.append(dict(method=variant,readout=readout,event=meta['id'],floor=meta['floor'],split=split,
                        **metrics(p[sl],dict(xy=xy[sl],scan_weights=sw))))
            for axis in range(2):
                fitted=json.loads((directory/f'axis{axis}.json').read_text());d=fitted['diagnostics']
                chosen=d['runs'][d['chosen_restart']]
                diagnostics.append(dict(event=meta['id'],method=variant,axis=axis,
                    selected_optimizer_converged=chosen['optimizer_converged'],selected_objective=chosen['objective'],
                    selected_evaluations=chosen['evaluations'],selected_gradient_max=chosen['gradient_max'],
                    boundary_parameters=chosen['near_bounds'],converged_restarts=sum(r['optimizer_converged'] for r in d['runs'])))
    summary=[]
    for variant in VARIANTS:
        for readout in ('raw','area_matched'):
            for split in ('internal','official'):
                selected=[r for r in rows if (r['method'],r['readout'],r['split'])==(variant,readout,split)]
                summary.append(dict(method=variant,readout=readout,split=split,
                    mean_error=float(np.mean([r['mde'] for r in selected])),events=len(selected)))
    result=dict(post_hoc=True,summary=summary,rows=rows,optimization=diagnostics,
        law_reference=str(BANK/'summary.json'),qualification='Fixed finite-budget method adaptation; not an optimized upper bound or exact paper reproduction')
    write(OUT/'area_matched_comparison.json',result)
    print(json.dumps(dict(summary=summary,selected_converged=sum(d['selected_optimizer_converged'] for d in diagnostics),models=len(diagnostics))),flush=True)


if __name__=='__main__': main()
