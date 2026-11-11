"""Account for real source labels kept by candidate-nearest supervision.

One fixed K3 episode per floor/mode on the already available B0/B2 sources.
No fitting, localization, source selection or target evaluation. This asks
whether legal source supervision is discarded because training locations are
chosen from H/S rather than the labelled source Query positions. Distinguish
unavailable information from information present but unused by this sampler.
"""
from candidates import unified_source_tasks_20261111 as S
import argparse
from pathlib import Path
import socket

import numpy as np
from threadpoolctl import threadpool_limits


def run(csv_path,output):
    _,V,_,views=S.load_sources(csv_path,[0,2],[1]);records=[]
    for building in V.FLOORS:
        C,H=views[building]
        for floor in V.FLOORS[building]:
            for draw in range(4):
                _,record,data=S.U.task(C,H,floor,3,draw,np.random.default_rng([20261111,building,floor,draw]))
                _,_,context,base,sr,qr,hr=data
                assert not np.intersect1d(base,qr).size
                Q,n,*_=H.det_stats(C.RSS[qr],C.XY[qr])
                target=H.STATS[(0,floor)][0]
                distance=np.linalg.norm(context['cand_xy'][:,None]-target[None],axis=2)
                nearest=distance.argmin(1)
                lookup={tuple(x):i for i,x in enumerate(Q)}
                labels=np.array([lookup.get(tuple(target[i]),-1) for i in nearest])
                kept=(distance.min(1)<=S.U.T.R_LABEL)&(labels>=0)
                used=np.unique(labels[kept]);matching=distance.min(1)[kept]
                records.append(dict(building=building,floor=floor,mode=S.U.E.MODES[draw],K=3,
                    source_query_positions=len(Q),source_query_scans=len(qr),
                    represented_label_positions=len(used),represented_label_scans=int(n[used].sum()),
                    discarded_label_positions=len(Q)-len(used),
                    candidate_cells=int(kept.sum()),distinct_APs=len(context['aps']),
                    matched_at_exact_coordinate=int((matching<1e-8).sum()),
                    nonzero_label_offsets=int((matching>=1e-8).sum()),
                    mean_label_offset_m=float(matching.mean()) if len(matching) else None,
                    max_label_offset_m=float(matching.max()) if len(matching) else None,
                    source_episode_feature_sha256=record['features_sha256']))
    summary={}
    for building in V.FLOORS:
        for kind,draws in (('full_history',(0,1)),('coherent_thin_history',(2,3))):
            selected=[r for r in records if r['building']==building and r['mode'] in [S.U.E.MODES[d] for d in draws]]
            available=sum(r['source_query_positions'] for r in selected)
            used=sum(r['represented_label_positions'] for r in selected)
            summary[f'B{building}_{kind}']=dict(episode_position_instances=available,
                used_position_instances=used,retained_fraction=used/available,
                candidate_cells=sum(r['candidate_cells'] for r in selected),
                nonzero_label_offsets=sum(r['nonzero_label_offsets'] for r in selected))
    output.parent.mkdir(parents=True,exist_ok=True);assert not output.exists()
    S.A.write(output,dict(complete=True,created=S.A.datetime.now(S.A.timezone.utc).isoformat(),
        code_sha256=S.A.digest(__file__),source_adapter_sha256=S.A.digest(S.__file__),
        source_csv_sha256=S.U.E.CSV_SHA,source_buildings=[0,2],excluded_targets=[1],
        records=records,summary=summary,fits=0,models_loaded=0,localization_scores=0,
        interpretation='Episode-position instances may repeat source positions; one fixed draw per mode, descriptive mechanism evidence only'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert socket.gethostname()=='ray-web'
    with threadpool_limits(limits=2):run(args.csv,args.output)
