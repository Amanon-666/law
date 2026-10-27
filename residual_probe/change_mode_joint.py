"""Joint-S evidence selection for the frozen rank-one change modes.

For each AP, C_tau = sk + tau^2 qS qS^T. ImportanceSupport first absorbs
the detected RSS with this FULL covariance and then estimates detection
evidence under the RSS-updated JOINT Gaussian. Thus log p(S|tau) is
-posterior.rss['loss'] + posterior.log_detection_evidence. APs have independent
latent amplitudes; their log evidences sum to select one shared tau. This is
a numerical joint-S evidence estimate, conditional on frozen q/w/sigma/means,
not exact integration or a calibrated posterior over tau.

Minimal driver API:
  selection = select_tau_joint(cases, modes, obs, namespace=kind+'/'+event_id,
                               proposal_dir=NEW_output/'selected_proposals')
  changed = add_covariance(grid_case, mode, selection['tau'])  # frozen module
  changed.update(load_selected_proposal(selection, ap))
  value, ess = zero_prediction(changed, obs)  # no second adaptive refresh

cases/modes are small AP-keyed mappings. Each S case needs only
path/sm/sk/n/k/s1/s2; mode needs q_s. Original proposals are read from path
one AP at a time across the fixed tau grid, keeping no proposal arrays.
Only a selected refreshed proposal is replayed with the same deterministic
seed and saved. Unrefreshed selections reuse the original immutable cache.
Neither pass reads Q fields/signals/coordinates or predicts any grid point.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from reference_pmc import setup, sample_reference, log_proposal
from residual_probe.change_mode import TAU_GRID
from support_integral import ImportanceSupport

MIN_ESS = 1000


def _original_proposal(path):
    with np.load(path, allow_pickle=False) as stored:
        return {k:torch.from_numpy(stored[k]).double() for k in ('samples','log_proposal')}


@torch.no_grad()
def _evaluate(case, mode, obs, tau, namespace):
    # All off-diagonals, including the original intercept/H/local covariance,
    # remain in both the RSS update and the normalized detection integration.
    sk = case['sk'] if tau == 0 else case['sk'] + tau**2 * mode['q_s'][:, None] * mode['q_s'][None]
    stats = {k: case[k] for k in ('n', 'k', 's1', 's2')}
    proposal = {k: case[k] for k in ('samples', 'log_proposal')}

    def posterior():
        post = ImportanceSupport(case['sm'], sk, stats, obs, **proposal)
        ess = float(post.log_weights.exp().square().sum().reciprocal())
        return post, ess

    post, original_ess = posterior()
    ess, seed = original_ess, None
    if ess < MIN_ESS:
        key = f'change-mode-joint-v1|{namespace}|{case["path"]}|tau={tau:.17g}|1'
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
        target, initial, center, root, _ = setup(case['sm'], sk, stats, obs)
        samples, _, info = sample_reference(target, initial, center, root, seed, defensive_fraction=.1)
        proposal = dict(samples=samples.detach(), log_proposal=log_proposal(samples, center, root, info).detach())
        post, ess = posterior()
    if ess < MIN_ESS:
        raise RuntimeError(f'Joint evidence ESS below {MIN_ESS} after one S-only refresh: '
                           f'{namespace}, {case["path"]}, tau={tau}, seed={seed}, ESS={ess}')
    score = float(-post.rss['loss'] + post.log_detection_evidence)
    if not math.isfinite(score) or not math.isfinite(ess):
        raise FloatingPointError(f'Nonfinite joint evidence/ESS: {case["path"]}, tau={tau}')
    return score, dict(original_ESS=original_ess, ESS=ess, refreshed=seed is not None, seed=seed), proposal


@torch.no_grad()
def select_tau_joint(cases, modes, obs, *, namespace, proposal_dir):
    """Sum complete S log evidence over ALL APs; retain only selected proposals.

    The caller supplies every eligible AP once, in a fixed mapping order.
    q/w/sigma/mean/obs and the original proposals must remain frozen between
    passes. No optimizer, new direction, Q access or tau-grid change occurs.
    """
    scores = np.zeros(len(TAU_GRID))
    curves = {}
    for ap, support in cases.items():
        case = dict(support, **_original_proposal(support['path']))
        mode = modes[ap]
        values, diagnostics = [], []
        for i, tau in enumerate(TAU_GRID):
            value, diagnostic, _ = _evaluate(case, mode, obs, tau, namespace)
            scores[i] += value
            values.append(value); diagnostics.append(diagnostic)
        curves[str(ap)] = dict(original_path=case['path'], log_evidence=values, diagnostics=diagnostics)
    best = int(scores.argmax()); tau = TAU_GRID[best]
    selected = {ap: dict(original_path=row['original_path'], refreshed_path=None,
                         log_evidence=row['log_evidence'][best], **row['diagnostics'][best])
                for ap, row in curves.items()}
    pending = {ap for ap, row in selected.items() if row['refreshed']}
    if pending:
        proposal_dir = Path(proposal_dir)
        proposal_dir.mkdir(parents=True, exist_ok=True)
        for key, support in cases.items():
            ap = str(key)
            if ap not in pending:
                continue
            case = dict(support, **_original_proposal(support['path']))
            mode = modes[key]
            value, diagnostic, proposal = _evaluate(case, mode, obs, tau, namespace)
            if value != selected[ap]['log_evidence'] or diagnostic != curves[ap]['diagnostics'][best]:
                raise RuntimeError(f'Selected proposal replay changed evidence/ESS: {ap}')
            path = proposal_dir / f'ap{ap}.npz'
            with path.open('xb') as stream:
                np.savez(stream, **{k: v.numpy() for k, v in proposal.items()})
            selected[ap]['refreshed_path'] = str(path)
    return dict(tau=tau, grid=list(TAU_GRID), joint_log_evidence=scores.tolist(), APs=len(curves),
                boundary_hit=best in (0, len(TAU_GRID)-1), namespace=namespace,
                per_AP=curves, selected_proposals=selected, min_ESS=MIN_ESS,
                scope='S-only full joint evidence estimate, independently integrated per AP; frozen nuisance parameters',
                query_used=False, w_sigma_reselected=False, means_unchanged=True,
                proposal_policy='Original proposal first; one current-covariance S-only refresh below ESS1000; selected refresh replay only')


def load_selected_proposal(selection, ap):
    """Overlay on the original case before frozen zero_prediction, without refresh."""
    path = selection['selected_proposals'][str(ap)]['refreshed_path']
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as stored:
        return {k: torch.from_numpy(stored[k]).double() for k in ('samples', 'log_proposal')}


def _contrast_check(output):
    from inference import JITTER, gaussian_update
    from residual_probe.change_mode import select_tau

    obs = dict(alpha=1-1e-12, beta=-1000., detect_scale=1., noise_std=.1)
    y = torch.tensor([3., -3.]); q = torch.tensor([1., -1.])
    # Total observation covariance is I+100*11^T before the existing JITTER.
    path = output/'gaussian_contrast_original_proposal.npz'
    case = dict(path=str(path), sm=torch.zeros(2), sk=.99*torch.eye(2)+100*torch.ones((2, 2)),
                n=torch.ones(2), k=torch.ones(2), s1=y, s2=y.square())
    stats = {k: case[k] for k in ('n', 'k', 's1', 's2')}
    target, initial, center, root, _ = setup(case['sm'], case['sk'], stats, obs)
    samples, _, info = sample_reference(target, initial, center, root, 261027, defensive_fraction=.1)
    np.savez(path,samples=samples.numpy(),log_proposal=log_proposal(samples,center,root,info).numpy())
    mode = dict(q_s=q)
    composite = select_tau({0:case}, {0:mode}, obs)
    joint = select_tau_joint({0:case}, {0:mode}, obs, namespace='check/gaussian_contrast',
                             proposal_dir=output/'contrast_selected_proposals')
    exact = [-.5*(2*math.log(2*math.pi)+math.log(201+JITTER)
                   + math.log(1+2*t*t+JITTER)+18/(1+2*t*t+JITTER)) + 2*math.log(obs['alpha'])
             for t in TAU_GRID]
    assert composite['tau'] == 0 and joint['tau'] == 5
    # Detection is saturated. The remaining discrepancy is importance sampling
    # of normalized Gaussians under the same finite original proposal.
    error = float(np.max(np.abs(np.array(joint['joint_log_evidence'])-exact)))
    assert error < .01
    baseline = gaussian_update(case['sm'], case['sk'], stats, obs)
    return dict(composite=composite, joint=joint, gaussian_exact_curve=exact,
                contrast_eigenvalue='1+2*tau^2', common_eigenvalue=201,
                continuous_optimum_tau=math.sqrt(8.5), MC_max_error=error,
                original_RSS_log_evidence=-float(baseline['loss']), query_used=False)


def _real_check(output):
    from residual_probe.change_mode import build_modes
    from run_multi_history import digest

    source = ROOT/'results/source_residual_probe_v1_20261026'
    event = next(e for e in json.loads((source/'prepared/manifest.json').read_text())['events'] if e['J']==2)
    ap = event['APs'][0]
    path = source/'prepared'/event['id']/f'ap{ap}.npz'
    before = digest(path)
    # Deliberately do not load the cached Q arrays, Q coordinates or Q RSS.
    with np.load(path, allow_pickle=False) as stored:
        case = {k:torch.from_numpy(stored[k]).double() for k in
                ('sm','sk','phi_s','n','k','s1','s2')}
        prepared_evidence = float(stored['log_evidence'])
    case['path'] = str(path)
    with np.load(source/'paired_features'/event['id']/path.name, allow_pickle=False) as stored:
        case['phi_s'] = torch.cat((case['phi_s'],torch.from_numpy(stored['cross_s']).double()),1)
    grid_path = source/'grid_fields_v1'/event['id']/path.name
    with np.load(grid_path, allow_pickle=False) as stored:
        grid = {k:torch.from_numpy(stored[k]).double() for k in ('qm','phi_q')}
    model_path = source/'learning_v2'/event['building']/'paired/model.json'
    model = json.loads(model_path.read_text())
    assert event['building'] not in model['training_buildings']
    mode = build_modes(case,grid,torch.tensor(model['theta']))['learned_mode']
    obs = json.loads((source/'prepared/parameters.json').read_text())['obs']
    selection = select_tau_joint({ap:case}, {ap:mode}, obs,
        namespace=f'check/learned_mode/{event["id"]}', proposal_dir=output/'real_selected_proposals')
    stats = {k:case[k] for k in ('n','k','s1','s2')}
    original = ImportanceSupport(case['sm'],case['sk'],stats,obs,**_original_proposal(path))
    original_evidence = float(-original.rss['loss']+original.log_detection_evidence)
    assert selection['joint_log_evidence'][0] == original_evidence
    assert abs(original_evidence-prepared_evidence) < 1e-8
    selected_case = dict(case, **_original_proposal(path), **load_selected_proposal(selection,ap))
    repeated, diagnostic, _ = _evaluate(selected_case,mode,obs,selection['tau'],'unused-if-no-refresh')
    assert not diagnostic['refreshed']
    assert repeated == selection['selected_proposals'][str(ap)]['log_evidence']
    assert digest(path) == before
    return dict(event=event['id'], AP=ap, kind='learned_mode', selection=selection,
                tau0_exact_original=True, tau0_stored_evidence_error=abs(original_evidence-prepared_evidence),
                selected_proposal_reused_without_refresh=True, original_cache_unchanged=True,
                full_S_covariance=True, Q_arrays_loaded=False, grid_prediction=False,
                normalization=mode['normalization'],
                inputs_sha256={str(p):digest(p) for p in (path,grid_path,model_path,source/'prepared/parameters.json')})


def check():
    from run_multi_history import digest, write
    output = ROOT/'results/source_change_mode_joint_v1_20261027'
    output.mkdir(parents=True,exist_ok=False)
    frozen = [ROOT/'residual_probe/change_mode.py',ROOT/'results/source_change_mode_v1_20261027/checks.json']
    before = {str(p):digest(p) for p in frozen}
    write(output/'check_context.json',dict(created=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        mechanism=__doc__,code_sha256=digest(Path(__file__)),frozen_v1_sha256=before,
        hypothesis='Composite S evidence can discard a useful contrast direction; joint S evidence reads its covariance orientation',
        reverse_audit='Only S sm, FULL sk+tau²qSqS.T, n/k/s1/s2 and detached proposals enter evidence/refresh. No Q or grid prediction. Existing means/w/sigma/q are frozen.',
        scope='Prescribed no-Q Gaussian contrast and ONE real source AP only; no training or 24-event localization'))
    result = dict(contrast=_contrast_check(output),real=_real_check(output),
        completed=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),code_sha256=digest(Path(__file__)),
        cpu_threads=torch.get_num_threads(),training=False,performance_scored=False)
    assert {str(p):digest(p) for p in frozen} == before
    result['frozen_v1_unchanged'] = True
    write(output/'checks.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical checks stay on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['check'])
    args=parser.parse_args(); globals()[args.stage]()
