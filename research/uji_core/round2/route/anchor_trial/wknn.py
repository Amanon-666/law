"""Source-only neighbor selection under the frozen early-library anchor design."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import fit
from research.uji_core.next import anchor_measure as anchors
from research.uji_core.inverse import order,squared,vote,signal
from research.uji_core.next.paired_map_wknn import group_ids
from research.uji_core.protocol import position_weights

EXPECTED='ed3ae84fd08d2286b7c754937d0d1c50b9304e8c339c3a3a5487689eea481162'
NEIGHBORS=(1,3,7,15)
ARMS=('original_map','forecast_map','raw','prototype')


def mdes(q_signal, ref_signal, xy, ids, query_xy, w):
    distance=squared(q_signal,ref_signal)
    ranked=order(distance,ids,max(NEIGHBORS))
    ds=np.take_along_axis(distance,ranked,axis=1)
    loc=xy[ranked]-xy[0]
    return {n:float(w@np.linalg.norm(vote(ds[:,:n],loc[:,:n],q_signal.shape[1])[0]+xy[0]-query_xy,axis=1))
            for n in NEIGHBORS}


def main(archive,out):
    frozen=out/'frozen.json'
    if fit.digest(frozen)!=EXPECTED:raise ValueError('Frozen anchor rule changed')
    gains=json.loads(frozen.read_text())['g_by_K']
    pairs,_=fit.old_pairs(archive)
    supports=json.loads((out/'support_ids.json').read_text())
    paths={(d['pair'],d['floor'],d['draw']):d['S_order'] for d in supports}
    rows=[]
    for a,b in pairs:
        pair=a['date']+'-'+b['date']
        earlier,later=fit.load(archive,a),fit.load(archive,b)
        for floor,other in [(3,5),(5,3)]:
            h=earlier[other];t=earlier[floor];q=later[floor]
            prior=anchors.history([h],scale=fit.UNIT)
            hids=h['ids'].astype(np.int64);tids=t['ids'].astype(np.int64)
            hxy,hid=group_ids(h['xy'],hids)
            ref_xy=hxy
            # Pool measured scans in signal space, undetected = 0 after +110 shift.
            hsig=signal(h['rss'])
            h_proto=np.stack([hsig[np.all(h['xy']==pt,axis=1)].mean(0) for pt in ref_xy])
            qsig=signal(q['rss'])
            rowno={num:i for i,num in enumerate(t['ids'])}
            for seed in fit.SEEDS:
                ids=[rowno[num] for num in paths[(pair,floor,seed)]]
                for k in fit.BUDGETS:
                    ix=ids[:k];s={f:t[f][ix] for f in ('rss','xy')};sx,sid=group_ids(s['xy'],tids[ix])
                    support_positions=set(map(tuple,sx))
                    keep=np.array([tuple(pt) not in support_positions for pt in q['xy']])
                    n_query=len(set(map(tuple,q['xy'][keep])))
                    if n_query!=24-k or keep.sum()!=12*n_query:raise ValueError('Query site overlap or count')
                    truth=q['xy'][keep];w=position_weights(truth)
                    field=anchors.build(prior,s,scale=fit.UNIT)
                    np.testing.assert_allclose(field.xy,np.vstack((hxy,sx)),rtol=0,atol=1e-11)
                    curr=field.visible;obs=qsig[keep][:,curr]
                    ab={'original_map':field.raw, 'forecast_map':field.raw+gains[str(k)]*(field.posterior-field.raw)}
                    rates={}
                    for method,anchor in ab.items():
                        rs=np.vstack((field.historical,anchor))+110
                        rates[method]=mdes(obs,rs,field.xy,np.r_[hid,sid],truth,w)
                    for method,hfeat,xy,ref_id in [('raw',hsig,h['xy'],hids),('prototype',h_proto,ref_xy,hid)]:
                        rs=np.vstack((hfeat[:,curr],signal(s['rss'])[:,curr]))
                        rates[method]=mdes(obs,rs,np.vstack((xy,s['xy'])),np.r_[ref_id,tids[ix]],truth,w)
                    for arm in ARMS:
                        for n,err in rates[arm].items():
                            rows.append(dict(pair=pair,floor=floor,draw=seed,K=k,n_neighbors=n,reference=arm,
                                             query_sites=n_query,query_scans=int(keep.sum()),mde_m=err))
    table=pd.DataFrame(rows)
    if len(table)!=700*4*4 or not np.isfinite(table.mde_m.values).all():raise ValueError('Incomplete tuning grid')
    # Every pair contributes both floors, all five budgets and all ten draws.
    risk=table.groupby(['reference','n_neighbors']).mde_m.mean()
    chosen={arm:min(NEIGHBORS,key=lambda n:(risk.loc[arm,n],n)) for arm in ARMS}
    direct=min(('raw','prototype'),key=lambda arm:(risk.loc[arm,chosen[arm]],arm!='raw',chosen[arm]))
    conf=dict(anchor_freeze_sha256=EXPECTED,source_protocol_sha256=fit.digest(out/'protocol.json'),
        script_sha256=fit.digest(__file__),
        k_original_map=chosen['original_map'],k_forecast_map=chosen['forecast_map'],
        direct_rep=direct,k_direct=chosen[direct],direct_alpha=0.,
        k_raw_direct=chosen['raw'],k_prototype_direct=chosen['prototype'],
        source='Exactly same seven pre-2017 disjoint visit pairs, 2 adjacent floors, 10 nested draws and 5 K. Never loads any UJIIndoorLoc, April route,July panel or external position results.',
        Q='Only later-date target-floor real query scans at coordinates outside *all* current S. For K=1/3/5/10/20 there are respectively 23/21/19/14/4 Q positions, 12 scans per position. Q position repeated scans count as one site in risk. Registered other-floor H can cover these positions.',
        arms='Both map WKNNs retain the H/S candidate multiplicities of the localization method, including H/S at the same XY. g=0 in original_map and the frozen g[K] in forecast_map. Direct raw/prototype WKNN uses unmodified H scans or per-floor per-XY filled means and raw S. No floor penalty (one physically adjacent H floor), no transport. All arms use same allowed H union current S APs, RMS distance, inverse-distance vote, deterministic input-row ID tie breaking.',
        choice_rule='One k for all K, minimizing labelled future-query mean position error. First average scans within XY, then equal Q sites, draws, K, floors and distinct date pairs. Fixed grid 1,3,7,15; smaller k wins exact risk ties. Direct additionally chooses between raw/prototype, prefer raw on equal risk.',
        note='These source selections reuse the old source campaigns that trained g. Source scores describe fitting. Each current K uses its own held-Q complement; this source design does not estimate a fixed-Q K-trend.')
    fit.dump(out/'wknn_frozen.json',conf)
    risk.to_csv(out/'wknn_source_risk.csv');table.to_csv(out/'wknn_source_events.csv.gz',index=False)
    print(fit.digest(out/'wknn_frozen.json'),chosen,direct,flush=True)
    print(risk.unstack().round(4).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();main(args.archive,args.out)
