"""Choose existing TPM field/match settings using B1 leave-floor-out tasks only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from external_baselines import tpm_adapter as adapter
from external_baselines.tpm_source_calibration import FLOORS, OUT as CALIBRATION, ROOT, sha, write
from uji_probability_study import load_raw, load_legacy, DATA, LEGACY

OUT=ROOT/'results/tpm_source_selection_20261027'
FIELDS=[(ell,noise) for ell in (20.,40.,60.,100.) for noise in (.1,.3,1.,3.)]
MATCHES=[(sigma,threshold) for sigma in (8.,12.,16.,24.) for threshold in (-90.,-85.,-80.,-75.)]
BUDGETS=(1,3,5,10,20)
DRAWS=20


def prepare():
    execution=json.loads((CALIBRATION/'execution.json').read_text())
    assert execution['state']=='complete_source_fits'
    raw=load_raw('trainingData.csv',set(FLOORS))
    events=[]
    for fi,floor in enumerate(FLOORS):
        f=raw[floor]
        for draw in range(DRAWS):
            rng=np.random.default_rng(271000+fi*100+draw)
            order=rng.permutation(len(f['positions']))
            support=[int(rng.choice(np.flatnonzero(f['group']==p))) for p in order[:20]]
            query=[int(rng.choice(np.flatnonzero(f['group']==p))) for p in order[20:52]]
            assert not set(map(tuple,f['xy'][support])) & set(map(tuple,f['xy'][query]))
            events.append(dict(id=f'{floor}_draw{draw}',target=floor,histories=[h for h in FLOORS if h!=floor],
                               draw=draw,support_rows=support,query_rows=query,
                               support_csv_rows=f['row_ids'][support].tolist(),query_csv_rows=f['row_ids'][query].tolist()))
    files=[CALIBRATION/'freeze.json',*(CALIBRATION/f'{floor}.json' for floor in (*FLOORS,'all_B1'))]
    for p in files[1:]:assert json.loads(p.read_text())['usable']
    OUT.mkdir(exist_ok=False)
    write(OUT/'manifest.json',dict(events=events))
    write(OUT/'freeze.json',dict(UTC=datetime.now(timezone.utc).isoformat(),purpose=__doc__,
        source_building='UJI_B1',excluded_development_buildings=['UJI_B0','UJI_B2','UAH2016-b20','Pisa2019-b60'],
        fields=FIELDS,matches=MATCHES,K=BUDGETS,draws_per_floor=DRAWS,R=1,query_cap=32,
        objective='Mean MDE, equal weights for source floor, draw and K. Select TPM-pool and SCM-T-pool independently from identical original 16x16 parameter grids.',
        validation='Target B1 floor excluded from transfer/observation fitting. H contains other B1 floors. Same fixed Q and nested S for all K; current H-or-S-visible APs only.',
        information='No development-building Q or independent-validation data used. Source validation labels choose only fixed field/matcher settings.',
        limits=['One source building, not independent-building performance evidence.',
                'Twenty resamples share a finite source corpus; selected scores are selection diagnostics, not unbiased performance.',
                'Calibration uses original three alternating sweeps and reported optimizer stops; not global-optimum guarantee.',
                'Multi-H exact-coordinate pooling is an explicit extension of the single-H TPM.'],
        code_sha256={str(p):sha(p) for p in [Path(__file__),Path(adapter.__file__),ROOT/'uji_probability_study.py',ROOT/'data.py',LEGACY/'cross_floor.py']},
        input_sha256={str(p):sha(p) for p in [*files,DATA/'trainingData.csv',OUT/'manifest.json']}))


def run():
    frozen=json.loads((OUT/'freeze.json').read_text())
    for section in ('code_sha256','input_sha256'):
        for path,digest in frozen[section].items():
            if sha(path)!=digest:raise ValueError(f'Changed frozen file: {path}')
    raw=load_raw('trainingData.csv',set(FLOORS))
    cross=load_legacy('cross_floor')
    events=json.loads((OUT/'manifest.json').read_text())['events']
    all_scores={method:[] for method in ['TPM_pool','SCMT_pool']}
    for floor in FLOORS:
        path=OUT/f'{floor}.json'
        if path.exists():raise FileExistsError(f'Existing source-selection output: {path}')
        transfer=json.loads((CALIBRATION/f'{floor}.json').read_text())['transfer']
        transfer={k:np.asarray(v) if isinstance(v,list) else v for k,v in transfer.items()}
        rows=[e for e in events if e['target']==floor]
        score={method:np.zeros((DRAWS,len(BUDGETS),len(FIELDS),len(MATCHES))) for method in all_scores}
        f=raw[floor]
        hnames=rows[0]['histories']
        hseen=np.logical_or.reduce([(raw[h]['rss']!=100).any(0) for h in hnames])
        origin=np.concatenate([raw[h]['positions'] for h in hnames]).mean(0)
        for e in rows:
            qrows=e['query_rows']
            for ki,k in enumerate(BUDGETS):
                sr=e['support_rows'][:k]
                visible=hseen | (f['rss'][sr]!=100).any(0)
                histories=[dict(rss=raw[h]['rss'][:,visible],xy=raw[h]['xy']-origin) for h in hnames]
                support=dict(rss=f['rss'][sr][:,visible],xy=f['xy'][sr]-origin)
                q=f['rss'][qrows][:,visible]
                truth=f['xy'][qrows]-origin
                for method in all_scores:
                    prior=adapter.history_map(histories,transfer if method=='TPM_pool' else None,cross)
                    for field_index,field in enumerate(FIELDS):
                        prototypes,positions=adapter.calibrate(prior,support,field)
                        for match_index,match in enumerate(MATCHES):
                            predictions=adapter.tobit_weights(q,prototypes,*match)@positions
                            value=np.linalg.norm(predictions-truth,axis=1).mean()
                            if not np.isfinite(value):raise FloatingPointError(f'Nonfinite {e["id"]} {method}')
                            score[method][e['draw'],ki,field_index,match_index]=value
            if (e['draw']+1)%5==0:
                print(json.dumps(dict(floor=floor,completed_draws=e['draw']+1,total_draws=DRAWS)),flush=True)
        write(path,dict(target=floor,scores={method:v.tolist() for method,v in score.items()},
                        axes=['draw','K','field','match'],freeze_sha256=sha(OUT/'freeze.json')))
        for method in all_scores:all_scores[method].append(score[method])
    selections={}
    for method,values in all_scores.items():
        values=np.stack(values)
        mean=values.mean((0,1,2))
        fi,mi=np.unravel_index(mean.argmin(),mean.shape)
        selections[method]=dict(field=FIELDS[fi],match=MATCHES[mi],source_selection_MDE=float(mean[fi,mi]),
            selected_indices=[int(fi),int(mi)],all_candidate_mean_MDE=mean.tolist(),
            selected_per_floor_K=values[:,:,:,fi,mi].mean(1).tolist())
    write(OUT/'selection.json',dict(selections=selections,freeze_sha256=sha(OUT/'freeze.json'),
        target_transfer_fit=str(CALIBRATION/'all_B1.json'),
        interpretation='Source-selected settings only; no claim of development or independent-building performance.'))
    write(OUT/'execution.json',dict(state='complete',source_floors=4,draws=80,budgets=5,methods=2,
        field_match_candidates=256,development_predictions=0,HTTP_downloads=0))
    print(json.dumps(dict(stage='source_selection_complete',selections={m:{k:v for k,v in d.items() if k not in ['all_candidate_mean_MDE','selected_per_floor_K']} for m,d in selections.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    (prepare if args.stage=='prepare' else run)()
