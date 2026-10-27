"""Versioned broad-task source entry; old frozen learners stay unchanged.

The model and observation objective are the existing paired10 probe. This
entry changes the task distribution and uses the already checked bounded
proposal policy at every K. All data preparation is explicit and offline.
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from residual_probe.prepare import ROOT, data_module, InterceptBankPrior, choose_intercept
from residual_probe.learning import observation_loss, prediction, load_case
from conditional_operator_learning import fit_episode
from bounded_pmc import setup, sample_reference, log_proposal
from support_integral import SupportIntegral
from inference import _chol
from run import tensor
import run_k_curve as curve

OUT = ROOT / 'results/history_learning_stage2_20261027'
INDICES = list(range(10))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'little')


@lru_cache(maxsize=1)
def row_plan():
    return read(OUT / 'source_row_plan.json')


@lru_cache(maxsize=2)
def event_inputs(event_id):
    plan = row_plan()
    meta = dict(next(e for e in plan['events'] if e['id'] == event_id))
    old = data_module()
    building = old.building_data(meta['building'])
    histories = meta['histories']
    sources = {str(h): old.h_selection(meta['building'], h) for h in histories}
    rows = {h: np.asarray(sources[str(h)]['rows'], dtype=int) for h in histories}
    origin, hxy, hgrid = curve.episodes.history_frame(building, histories, rows)
    s_rows = np.asarray(meta['support_rows'], dtype=int)
    aps = curve.episodes.selected_aps(building, meta['target'], histories, rows, s_rows)
    grid = curve.episodes.grid_definition(
        np.concatenate((hxy, building['floors'][meta['target']]['xy'][s_rows] - origin)),
        'retained H positions union current K Support only; outward 2-unit rounding')
    meta.update(history_metadata=sources, origin=origin.tolist(),
                history_grid=hgrid, grid=grid, AP_indices=aps.tolist())
    task = curve.episodes.build_task(meta)
    params = read(old.PARAMETERS)
    return meta, task, params


@lru_cache(maxsize=640)
def ap_catalog(event_id):
    return event_inputs(event_id)[0]['AP_indices']


@lru_cache(maxsize=2)
def event_data(event_id):
    meta, task, params = event_inputs(event_id)
    return meta, task, curve.support_cases(meta, task, params['base']), params


def adapt(event_id, destination):
    meta, task, cases, params = event_data(event_id)
    start = time.perf_counter()
    weights, trials = fit_episode(cases, params['base'], params['obs'])
    sigma = choose_intercept(cases, weights, params['base'], params['obs'])
    result = dict(event=event_id, weights=weights, trials=trials, sigma=sigma,
                  AP_indices=meta['AP_indices'], seconds=time.perf_counter()-start,
                  source_row_plan_sha256=digest(OUT/'source_row_plan.json'),
                  query_used=False)
    write(destination, result)
    return result


@torch.no_grad()
def matrices(event_id, ap, adaptation):
    meta, task, cases, params = event_data(event_id)
    case = cases[ap]
    prior = InterceptBankPrior(case['histories'], adaptation['weights'], params['base'],
                              adaptation['sigma']['intercept_std'])
    sx, qx = case['sx'], tensor(task['Q']['positions'])
    sm, sf = prior.features(sx)
    qm, qf = prior.features(qx)
    column = meta['AP_indices'].index(ap)
    q = task['Q']
    result = dict(sm=sm, sk=prior.covariance(sx, sx, sf, sf), qm=qm,
        qv=prior.diagonal(qx, qf), cross=prior.covariance(qx, sx, qf, sf),
        phi_s=curve.phi(case, sx, params['base']),
        phi_q=curve.phi(case, qx, params['base']), **case['stats'],
        rss=tensor(q['rss'][:, column]), group=torch.tensor(q['group']),
        query_weights=tensor(q['scan_weights']), path=f'{event_id}/ap{ap}')
    return result, params['obs']


@torch.no_grad()
def proposal(case, obs, theta, seed):
    sm = case['sm'] + 20*(case['phi_s'] @ theta)
    stats = {k: case[k] for k in ('n', 'k', 's1', 's2')}
    target, initial, center, root, _ = setup(sm, case['sk'], stats, obs)
    samples, _, info = sample_reference(target, initial, center, root, seed, defensive_fraction=.1)
    if info['effective_samples'] >= 1000:
        return dict(samples=samples, log_proposal=log_proposal(samples, center, root, info)), dict(
            sampler='bounded_pmc', ESS=info['effective_samples'], samples=len(samples), seed=seed)
    # Same fixed two-scramble policy as the completed K-curve continuation.
    batches, means, covs, evidence, ess = [], [], [], [], []
    for mask in (0x13579BDF, 0x2468ACE0):
        p = SupportIntegral(sm, case['sk'], stats, obs, power=18,
                            seed=seed ^ mask, proposal='rss_gaussian')
        w = p.log_weights.exp()
        m = w @ p.samples
        centered = p.samples-m
        batches.append(p.samples)
        means.append(m)
        covs.append((centered.T*w)@centered)
        evidence.append(p.log_detection_evidence)
        ess.append(float(w.square().sum().reciprocal()))
    scale = ((covs[0].diag()+covs[1].diag())/2).sqrt()
    gaps = dict(mean=float((means[0]-means[1]).abs().max()),
                covariance=float(((covs[0]-covs[1]).abs()/(scale[:,None]*scale[None])).max()),
                log_evidence=float((evidence[0]-evidence[1]).abs()))
    record = dict(sampler='rss_gaussian_two_scrambles', seed=seed,
        original_ESS=info['effective_samples'], ESS=ess, gaps=gaps, samples=2**19,
        accepted=min(ess)>=1000 and gaps['mean']<=.1*obs['noise_std']
                 and gaps['covariance']<=.1 and gaps['log_evidence']<=.05)
    if not record['accepted']:
        raise RuntimeError(f'Fixed-budget source QMC failed: {record}')
    samples = torch.cat(batches)
    gaussian = torch.distributions.MultivariateNormal(
        p.rss['mean'], scale_tril=_chol(p.rss['cov']), validate_args=False)
    return dict(samples=samples, log_proposal=gaussian.log_prob(samples)), record


def check_numerics():
    """New K support: real H/S, artificial Q signals, no training updates."""
    directory = OUT/'numerical_check'
    directory.mkdir(exist_ok=False)
    checks = []
    events = [f'UJI_B2_F0_draw0_K{k}' for k in (1,3,5,10,20)]
    events += ['Pisa2019-b60_F0_draw0_K20']
    for event_id in events:
        start = time.perf_counter()
        adaptation = adapt(event_id, directory/f'{event_id}_adaptation.json')
        _, _, cases, _ = event_data(event_id)
        # A source-only difficult AP: at least one detection and historical H,
        # with the fewest S detections; no Q signal or score selects it.
        eligible = [a for a,c in cases.items()
                    if any(h is not None for h in c['histories']) and c['stats']['k'].sum()>0]
        ap = min(eligible, key=lambda a:(float(cases[a]['stats']['k'].sum()),a))
        case, obs = matrices(event_id, ap, adaptation)
        case['rss'] = torch.where(torch.arange(len(case['rss']))%2==0,-70.,100.).double()
        theta = torch.linspace(-.02,.03,10,requires_grad=True)
        numeric, info = proposal(case, obs, theta.detach(), seed_for(f'check/{event_id}/{ap}'))
        case.update(numeric)
        loss, ess = observation_loss(theta, case, obs, INDICES)
        grad, = torch.autograd.grad(loss, theta)
        direction = torch.linspace(.3,1.2,10)
        eps = 1e-5
        plus,_ = observation_loss(theta.detach()+eps*direction,case,obs,INDICES)
        minus,_ = observation_loss(theta.detach()-eps*direction,case,obs,INDICES)
        fd = (plus-minus)/(2*eps)
        torch.testing.assert_close(grad@direction,fd,atol=2e-6,rtol=2e-5)
        record = dict(event=event_id,AP=ap,ESS=float(ess.detach()),proposal=info,
            directional_gradient_error=float((grad@direction-fd).abs()),
            seconds=time.perf_counter()-start,
            proposal_tensor_bytes=sum(v.numel()*v.element_size() for v in numeric.values()),
            actual_query_scored=False,training_updates=0)
        checks.append(record)
        write(directory/f'{event_id}_check.json',record)
        print(json.dumps(record),flush=True)
    write(directory/'summary.json',dict(complete=True,checks=checks,
        code_sha256=digest(Path(__file__)),scope='Finite numerical and gradient checks, not localization or proof for all source cases'))


def schedule():
    """Fix all 4 x 512 updates and AP draws without model/query scores."""
    plan = read(OUT/'source_row_plan.json')
    buildings = sorted({e['building'] for e in plan['events']})
    schedules, wanted = {}, {}
    for heldout in buildings:
        train = [b for b in buildings if b != heldout]
        rng = np.random.default_rng(261026)
        steps = []
        for step in range(512):
            b = train[int(rng.integers(len(train)))]
            floors = sorted({e['target'] for e in plan['events'] if e['building']==b})
            f = floors[int(rng.integers(len(floors)))]
            events = [e for e in plan['events'] if e['building']==b and e['target']==f]
            event = events[int(rng.integers(len(events)))]['id']
            aps = rng.choice(ap_catalog(event),size=4,replace=True).tolist()
            steps.append(dict(step=step,event=event,APs=aps))
            wanted.setdefault(event,set()).update(aps)
        schedules[heldout] = steps
    wanted = {event:sorted(aps) for event,aps in wanted.items()}
    k_by_event = {e['id']:e['K'] for e in plan['events']}
    size = sum((k_by_event[event]+1)*32768*8*len(aps)
               for event,aps in wanted.items())
    result = dict(schedules=schedules,requested_cases=wanted,
        events=len(wanted),unique_AP_cases=sum(map(len,wanted.values())),
        ordinary_proposal_tensor_bytes=size,source_row_plan_sha256=digest(OUT/'source_row_plan.json'),
        training_updates=0,query_scored=False)
    write(OUT/'schedule.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('schedules','requested_cases')}),flush=True)


def freeze():
    import sys
    import importlib.metadata
    original = curve.verify()
    schedule = read(OUT/'schedule.json')
    numerical = read(OUT/'numerical_check/summary.json')
    assert numerical['complete'] and len(numerical['checks'])==6
    check_code = OUT/'numerical_check/broad_source_checked.py'
    assert digest(check_code)==numerical['code_sha256']
    # The completed check covered these exact numerical functions; later
    # additions supply orchestration, not a changed posterior or gradient.
    import ast
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False)
                for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    before, after = functions(check_code), functions(__file__)
    for name in ('matrices','proposal','check_numerics'):
        assert before[name]==after[name],name
    files = {Path(__file__),Path(__file__).with_name('episode_rows.py'),
             Path(__file__).with_name('prepare_episode_rows.py')}
    for module in list(sys.modules.values()):
        path = getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py'):
            files.add(Path(path))
    inputs = {OUT/'decision.json',OUT/'source_row_plan.json',OUT/'row_plan_check.json',
              OUT/'schedule.json',OUT/'validation_reservation.json',
              OUT/'numerical_check/summary.json',check_code}
    old = data_module()
    inputs.update(Path(p) for p in read(curve.OUT/'data_check.json')['source_sha256'])
    for b in old.CONFIGS:
        for f in old.building_data(b)['floors']:
            info = old.h_selection(b,f)
            inputs.update(Path(info['cache']).glob('*.npz'))
    record = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(p):digest(p) for p in sorted(files)},
        input_sha256={str(p):digest(p) for p in sorted(inputs)},
        model='Existing paired10; same post-S per-AP observation NLL and ridge .001',
        training='4 held-building folds; 512 scheduled updates/fold; Adam .01; 4 AP draws/update; final checkpoint only',
        numerical='Bounded PMC 6x4096+32768, defensive .1; only ESS<1000 permits one two-scramble 2^18 RSS-Gaussian attempt with prior continuation consistency gates. At changed theta, one H/S-only proposal refresh; otherwise stop.',
        cache_budget='Only scheduled AP cases, at most 16 GiB compressed cache per building; no dataset/environment downloads; CPU2/process, at most4 preparation or training processes, no GPU.',
        gradient='Original ImportanceSupport fixed-proposal derivatives, includes S posterior weights and Q propagation; proposal auxiliaries detached.',
        permissions='H/current S define AP and grid and fit w/sigma. Q coordinates supervise source predictions only; no reserved official corpus loaded. g fold excludes entire target building.',
        reverse_audit='matrices uses the same prior/covariance/phi and posterior readout as frozen curve. Weights and intercept selected before g. S-only control retains exact feature set, zero start, same .001 penalty and converged local fitter. base/obs remain B0-trained, so development folds are g-only holdouts.',
        stopping='decision.json; no numerical failed-AP dropping, selective split averaging or post-test adjustment.',
        package_versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','pypmc')},
        unique_AP_cases=schedule['unique_AP_cases'],ordinary_proposal_tensor_bytes=schedule['ordinary_proposal_tensor_bytes'],
        independent_validation='Reserved; scoring interface audit pending, not needed to execute the already isolated development experiment.')
    write(OUT/'freeze.json',record)
    print(json.dumps(dict(stage='frozen',sources=len(files),inputs=len(inputs),cases=record['unique_AP_cases'])),flush=True)


def verify():
    frozen = read(OUT/'freeze.json')
    for key in ('code_sha256','input_sha256'):
        for path,sha in frozen[key].items():
            if digest(path)!=sha:
                raise ValueError(f'Frozen source/input changed: {path}')
    return frozen


def prepare(building):
    verify()
    schedule = read(OUT/'schedule.json')
    records, total_bytes = [], 0
    for event,aps in schedule['requested_cases'].items():
        if event_inputs(event)[0]['building'] != building:
            continue
        adaptation = adapt(event,OUT/'adaptation'/f'{event}.json')
        for ap in aps:
            start = time.perf_counter()
            print(json.dumps(dict(stage='preparing',event=event,AP=ap)),flush=True)
            case,obs = matrices(event,ap,adaptation)
            numeric,info = proposal(case,obs,torch.zeros(10),seed_for(f'prepare/{event}/{ap}'))
            path = OUT/'cache'/event/f'ap{ap}.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            arrays = {k:v.numpy() for k,v in dict(case,**numeric).items() if isinstance(v,torch.Tensor)}
            with path.open('xb') as stream:
                np.savez_compressed(stream,**arrays)
            total_bytes += path.stat().st_size
            if total_bytes > 16*1024**3:
                raise RuntimeError(f'Source cache exceeds frozen building budget: {building}')
            records.append(dict(event=event,AP=ap,path=str(path),sha256=digest(path),
                bytes=path.stat().st_size,proposal=info,seconds=time.perf_counter()-start))
            print(json.dumps(dict(stage='prepared',building=building,event=event,AP=ap,
                cases=len(records),bytes=total_bytes)),flush=True)
    write(OUT/'prepared'/f'{building}.json',dict(complete=True,building=building,cases=records,bytes=total_bytes))


def train(heldout):
    verify()
    schedule = read(OUT/'schedule.json')['schedules'][heldout]
    for building in sorted(data_module().CONFIGS):
        if building==heldout:continue
        records=read(OUT/'prepared'/f'{building}.json')
        assert records['complete']
        for case in records['cases']:
            assert digest(case['path'])==case['sha256'],case['path']
    out=OUT/'models'/heldout
    out.mkdir(parents=True,exist_ok=False)
    obs=read(data_module().PARAMETERS)['obs']
    theta=torch.nn.Parameter(torch.zeros(10))
    optimizer=torch.optim.Adam([theta],lr=.01)
    started=time.perf_counter()
    with (out/'trace.jsonl').open('x') as trace:
        for step in schedule:
            optimizer.zero_grad()
            losses,ess_values,refreshes=[],[],[]
            for ap in step['APs']:
                path=OUT/'cache'/step['event']/f'ap{ap}.npz'
                case=load_case(path)
                try:
                    loss,ess=observation_loss(theta,case,obs,INDICES)
                except RuntimeError as error:
                    if 'proposal has insufficient ESS:' not in str(error):raise
                    print(json.dumps(dict(stage='source_proposal_refresh',heldout=heldout,
                        step=step['step'],event=step['event'],AP=ap,trigger=str(error))),flush=True)
                    numeric,info=proposal(case,obs,theta.detach(),seed_for(f'{heldout}/{step["step"]}/{ap}'))
                    case.update(numeric)
                    loss,ess=observation_loss(theta,case,obs,INDICES)
                    refreshes.append(dict(AP=ap,proposal=info))
                losses.append(loss);ess_values.append(float(ess.detach()))
            data_loss=torch.stack(losses).mean()
            objective=data_loss+.001*theta.square().sum()
            objective.backward()
            if not torch.isfinite(objective) or not torch.isfinite(theta.grad).all():
                raise FloatingPointError('Nonfinite broad-source objective/gradient')
            optimizer.step()
            record=dict(**step,observation_loss=float(data_loss.detach()),
                min_ESS=min(ess_values),theta=theta.detach().tolist(),refreshes=refreshes)
            trace.write(json.dumps(record)+'\n');trace.flush()
            if step['step']%32==0:print(json.dumps(dict(stage='training',heldout=heldout,step=step['step'],min_ESS=min(ess_values))),flush=True)
    write(out/'model.json',dict(kind='paired',theta=theta.detach().tolist(),feature_indices=INDICES,
        heldout=heldout,training_buildings=sorted(b for b in data_module().CONFIGS if b!=heldout),
        steps=512,seed=261026,seconds=time.perf_counter()-started,freeze_sha256=digest(OUT/'freeze.json')))
    print(json.dumps(dict(stage='training_complete',heldout=heldout,seconds=time.perf_counter()-started)),flush=True)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['check','schedule','freeze','prepare','train'])
    parser.add_argument('--building')
    args = parser.parse_args()
    if args.stage=='check':check_numerics()
    elif args.stage=='schedule':schedule()
    elif args.stage=='freeze':freeze()
    elif args.stage=='prepare':prepare(args.building)
    else:train(args.building)
