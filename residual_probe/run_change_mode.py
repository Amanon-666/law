"""Two frozen spatial change directions, calibrated on S before Q scoring.

Posthoc development experiment after the deterministic mean and S calibration
results. No retraining, new data, query-dependent selection or new readout.
"""
import argparse
from pathlib import Path
import socket

import numpy as np
import torch

from inference import observation_log_prob
from residual_probe import localization as loc
from residual_probe import change_mode as mode
from run_multi_history import digest

ROOT, SOURCE = loc.ROOT, loc.SOURCE
OUT = ROOT/'results/source_change_mode_v1_20261027'
ARMS = ('history_mode', 'learned_mode')


def grid_arrays(event, ap):
    with np.load(loc.GRID/event/f'ap{ap}.npz', allow_pickle=False) as a:
        return {k:torch.from_numpy(a[k]).double() for k in ('qm','qv','cross','phi_q')}


def support_modes(event, theta):
    cases, modes = {}, {arm:{} for arm in ARMS}
    for ap in event['APs']:
        path = loc.PREPARED/event['id']/f'ap{ap}.npz'
        with np.load(path, allow_pickle=False) as a:
            case = {k:torch.from_numpy(a[k]).double() for k in ('sm','sk','phi_s','n','k','s1','s2')}
        with np.load(SOURCE/'paired_features'/event['id']/path.name, allow_pickle=False) as a:
            case['phi_s'] = torch.cat((case['phi_s'],torch.from_numpy(a['cross_s']).double()),1)
        cases[ap] = case
        built = mode.build_modes(case, grid_arrays(event['id'],ap), theta)
        for arm in ARMS:
            modes[arm][ap] = built[arm]
    return cases, modes


def freeze():
    loc.verify()
    check = OUT/'checks.json'
    if not check.exists():
        raise FileNotFoundError('Complete change-mode logical and real-AP checks first')
    assert loc.read(check)['code_sha256']==digest(ROOT/'residual_probe/change_mode.py')
    files = [Path(__file__), ROOT/'residual_probe/change_mode.py',
             ROOT/'residual_probe/adaptive_proposal.py', ROOT/'residual_probe/learning.py',
             ROOT/'inference.py', ROOT/'support_integral.py', ROOT/'reference_pmc.py']
    models = {e['building']:SOURCE/'learning_v2'/e['building']/'paired/model.json'
              for e in loc.manifest()['events']}
    for b,p in models.items():
        m = loc.read(p)
        assert b not in m['training_buildings'] and m['feature_indices']==list(range(10))
    if (OUT/'freeze.json').exists():
        raise FileExistsError(OUT/'freeze.json')
    loc.write(OUT/'freeze.json', dict(created=loc.now(), arms=ARMS,
        chronology=__doc__,
        hypothesis='Source experience may transfer a spatial change shape more reliably than its sign/amplitude. S infers per-AP amplitude via the original field posterior. Learned shape must beat the unlearned history shape to support a learning increment.',
        model='Original mean and full history/local/intercept covariance plus tau^2 q(x)q(xprime); independent latent amplitudes per AP, one S-selected tau per event/arm',
        normalization='Each AP direction centered and unit RMS on the frozen H-union-S grid; S uses identical center/scale. Constant/empty direction gives zero kernel.',
        selection='tau in [0,5,10,15,20] dB by all-AP S-only one-position censored composite likelihood, both arms fixed before Q. Original w/sigma/base/obs held fixed.',
        reverse_audit='Source-heldout paired coefficients and H-only fields define q; S labels select tau and update the field. Then Q RSS scores all APs on the unchanged grid. Q coordinates enter only final metrics.',
        failure_conditions='Wrong q shape, qS=0, confounding with intercept/local field, or variance fit without correct spatial correlation can prevent gains. This is a mature rank-one GP/random-slope construction, not a new kernel claim. No promise of no negative transfer.',
        interpretation='All 24 events are development data; J repeats queries; g-only leave-building-out, B0 base/obs not held out. No cross-unit mean or independent significance claim.',
        unchanged='K5/R1/Q12, raw scan full AP likelihood sum, uniform frozen H/S grid, posterior mean coordinates, original proposal with one S-only ESS refresh if needed',
        zero_reuse='When S selects tau=0, reuse the already completed exact zero-g event and point to its saved scores; do not recompute or duplicate arrays.',
        code_sha256={str(p):digest(p) for p in files}, checks_sha256=digest(check),
        prepared_manifest_sha256=digest(loc.PREPARED/'manifest.json'),
        source_models={b:dict(path=str(p),sha256=digest(p)) for b,p in models.items()}))
    loc.log(stage='change_mode_frozen', events=24, arms=ARMS)


@torch.no_grad()
def event_score(event, frozen):
    dest = OUT/event['id']; dest.mkdir(exist_ok=True)
    theta = torch.tensor(loc.read(frozen['source_models'][event['building']]['path'])['theta']).double()
    cases,modes = support_modes(event,theta)
    obs = loc.read(loc.PREPARED/'parameters.json')['obs']
    selected = {arm:mode.select_tau(cases,modes[arm],obs) for arm in ARMS}
    # This record is written before any Q observation or label is loaded.
    cosine = []
    for ap in event['APs']:
        a,b = (modes[arm][ap]['q_g'] for arm in ARMS)
        denom = a.norm()*b.norm()
        if denom>0: cosine.append(float((a@b)/denom))
    loc.write(dest/'adaptation.json',dict(created=loc.now(),selection=selected,
        normalization={arm:{str(ap):modes[arm][ap]['normalization'] for ap in event['APs']} for arm in ARMS},
        direction_comparison=dict(nonconstant_pairs=len(cosine),
            absolute_cosine_median=float(np.median(np.abs(cosine))) if cosine else None),query_used=False))
    del cases
    active = [arm for arm in ARMS if selected[arm]['tau']!=0]
    for arm in ARMS:
        if arm in active:continue
        previous=loc.OUT/'zero'/event['id']
        result=dict(loc.read(previous/'result.json'),kind=arm,tau=0.,
                    score_source=str(previous/'predictions.npz'),reuse_reason='tau0 exact original model',completed=loc.now())
        loc.write(dest/arm/'result.json',result)
        loc.log(stage='change_mode_tau0_reused',event=event['id'],kind=arm,mde=result['mde'])
    if not active:return
    points = torch.from_numpy(np.load(loc.GRID/event['id']/'grid.npy',allow_pickle=False)).double()
    scores = {arm:torch.zeros((12,len(points))) for arm in active}
    ess = {arm:[] for arm in active}
    predictors = {arm:mode.make_predictor(arm,event['id'],selected[arm]['tau'],dest/arm/'proposal_refreshes.jsonl') for arm in active}
    for index,ap in enumerate(event['APs']):
        path = loc.PREPARED/event['id']/f'ap{ap}.npz'
        case = loc.learning().load_case(path,SOURCE/'paired_features')
        case.update(grid_arrays(event['id'],ap))
        for arm in active:
            changed = mode.add_covariance(case,modes[arm][ap],selected[arm]['tau'])
            value,effective = mode.zero_prediction(changed,obs,predictors[arm])
            scores[arm] += observation_log_prob(case['rss'][:,None],value['mean'][None],value['variance'][None],obs)
            ess[arm].append(float(effective))
        if index%100==0:loc.log(stage='change_mode_APs',event=event['id'],complete=index+1,total=len(event['APs']))
    # All model choices and all scan scores are fixed before scoring Q labels.
    with np.load(loc.PREPARED/event['id']/f"ap{event['APs'][0]}.npz",allow_pickle=False) as a:
        truth = torch.from_numpy(a['qx'][a['group']]).double()
        weights = torch.from_numpy(a['query_weights']).double()
    distance,nearest = (truth[:,None]-points[None]).square().sum(-1).min(1)
    rows = torch.arange(len(truth))
    for arm in active:
        score=scores[arm]
        if not torch.isfinite(score).all():raise FloatingPointError('Nonfinite full-scan scores')
        logp=score.log_softmax(1);prediction=logp.exp()@points;error=(prediction-truth).norm(dim=1)
        result=dict(event=event['id'],building=event['building'],kind=arm,J=event['J'],
            coordinate_unit=event['coordinate_unit'],mde=float(weights@error),
            position_nll=float(-weights@logp[rows,nearest]),
            nearest_rank=float(weights@(1+(score>score[rows,nearest,None]).sum(1)).double()),
            min_ESS=min(ess[arm]),tau=selected[arm]['tau'],
            mean_nearest_grid_distance=float(weights@distance.sqrt()),
            per_query_errors=error.tolist(),query_weights=weights.tolist(),completed=loc.now())
        target=dest/arm;target.mkdir(exist_ok=True)
        np.savez_compressed(target/'predictions.npz',scores=score.numpy(),predictions=prediction.numpy(),
                            query_xy=truth.numpy(),query_weights=weights.numpy(),errors=error.numpy())
        loc.write(target/'result.json',result);loc.log(stage='change_mode_complete',**result)


def run():
    frozen=loc.read(OUT/'freeze.json');loc.verify()
    for path,sha in frozen['code_sha256'].items():assert digest(Path(path))==sha,path
    assert digest(OUT/'checks.json')==frozen['checks_sha256']
    assert digest(loc.PREPARED/'manifest.json')==frozen['prepared_manifest_sha256']
    for item in frozen['source_models'].values():assert digest(Path(item['path']))==item['sha256']
    for event in loc.manifest()['events']:
        if all((OUT/event['id']/arm/'result.json').exists() for arm in ARMS):continue
        event_score(event,frozen);summarize()


def summarize():
    rows=[loc.read(p) for p in sorted(OUT.glob('*/*/result.json'))]
    groups=[]
    for b in sorted({r['building'] for r in rows}):
        for arm in ARMS:
            selected=[r for r in rows if r['building']==b and r['kind']==arm]
            if selected:groups.append(dict(building=b,kind=arm,events=len(selected),
                coordinate_unit=selected[0]['coordinate_unit'],mde=float(np.mean([r['mde'] for r in selected])),
                position_nll=float(np.mean([r['position_nll'] for r in selected])),
                min_ESS=min(r['min_ESS'] for r in selected),taus=[r['tau'] for r in selected]))
    loc.write(OUT/'summary.json',dict(complete=len(rows)==48,groups=groups,events=rows))


if __name__=='__main__':
    if socket.gethostname()!='ray-web':raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','summary']);args=parser.parse_args()
    {'freeze':freeze,'run':run,'summary':summarize}[args.stage]()
