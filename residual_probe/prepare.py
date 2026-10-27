"""Frozen Bank + S-adapted random-intercept matrices and M-PMC proposals.

No g fitting, Q scoring, localization, data adapter or solver rewrite.
See protocol.md for array semantics, computation budget and data rights.
"""
import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import socket
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
PREPARATION = ROOT / 'results/source_residual_probe_v1_20261026/preparation'
OUT = ROOT / 'results/source_residual_probe_v1_20261026/prepared'
sys.path.insert(0, str(PREPARATION / 'code'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'operator_prior_v1'))

import numpy as np
import torch

import history_bank
import inference
import operator_prior
import reference_pmc
import source_weight_mixture
import support_integral
import transfer
from history_bank import BankPrior
from inference import gaussian_update, predict_from_posterior
from operator_prior import marginal_loglik, source_tokens
from reference_pmc import setup, sample_reference, log_proposal
from run_multi_history import digest, write
from source_weight_mixture import field_from_prior
from support_integral import ImportanceSupport

SIGMA = [0., 5., 10., 15., 20.]
FEATURES = ['mean_H_signal', 'mean_H_sd', 'mean_dz', 'mean_H_peak',
            'mean_peak_distance', 'mean_present', 'log1p_J', 'constant']


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def log(**row):
    print(json.dumps(dict(time=now(), **row), allow_nan=False), flush=True)


class InterceptBankPrior(BankPrior):
    def __init__(self, histories, weights, base, intercept_std):
        present = any(h is not None for h in histories)
        super().__init__(histories if present else [], weights if present else [], base)
        self.intercept_std = intercept_std

    def covariance(self, x, y, fx, fy):
        return super().covariance(x, y, fx, fy) + self.intercept_std ** 2

    def diagonal(self, x, fx):
        return super().diagonal(x, fx) + self.intercept_std ** 2


def history_features(case, x, base):
    if not any(h is not None for h in case['histories']):
        return x.new_zeros((len(x), 8))
    tokens = torch.stack([source_tokens(h, dz, x, base)
                          for h, dz in zip(case['histories'], case['dz'])]).mean(0)
    count = x.new_full((len(x), 1), np.log1p(len(case['histories'])))
    return torch.cat((tokens, count, x.new_ones((len(x), 1))), -1)


@torch.no_grad()
def choose_intercept(support_cases, weights, base, obs):
    scores = torch.zeros(len(SIGMA), dtype=torch.float64)
    for case in support_cases.values():
        prior = InterceptBankPrior(case['histories'], weights, base, 0.)
        mean, features = prior.features(case['sx'])
        variance = prior.diagonal(case['sx'], features)
        for i, sigma in enumerate(SIGMA):
            scores[i] += marginal_loglik(mean, variance + sigma ** 2,
                                        **case['stats'], obs=obs).sum()
    best = int(scores.argmax())
    return dict(intercept_std=SIGMA[best], grid=SIGMA,
                composite_log_likelihood=scores.tolist(), boundary_hit=best in [0, len(SIGMA) - 1],
                criterion='All AP/S-position composite marginal likelihood; not a calibrated posterior or exact joint EB',
                query_used=False)


def seed_for(episode_id, ap):
    key = f'residual-probe-v1|{episode_id}|AP{ap}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')


@torch.no_grad()
def prepare_case(case, weights, base, obs, intercept_std, seed):
    prior = InterceptBankPrior(case['histories'], weights, base, intercept_std)
    sx, qx, stats = case['sx'], case['qx'], case['stats']
    sm, sf = prior.features(sx); sk = prior.covariance(sx, sx, sf, sf)
    qm, qf = prior.features(qx); qv = prior.diagonal(qx, qf)
    cross = prior.covariance(qx, sx, qf, sf)
    target, proposal, center, root, _ = setup(sm, sk, stats, obs)
    samples, logw, info = sample_reference(target, proposal, center, root, seed, defensive_fraction=.1)
    logq = log_proposal(samples, center, root, info)
    if info['effective_samples'] < 1000:
        raise RuntimeError(f'Insufficient ESS for frozen proposal: {info["effective_samples"]}')
    log_rss = -float(gaussian_update(sm, sk, stats, obs)['loss'])
    arrays = {k: v.numpy() for k, v in dict(sx=sx, qx=qx, sm=sm, sk=sk, qm=qm, qv=qv,
        cross=cross, phi_s=history_features(case, sx, base), phi_q=history_features(case, qx, base),
        samples=samples, log_proposal=logq, logw=logw, rss=case['rss'], group=case['group'],
        query_weights=case['query_weights'], **stats).items()}
    arrays.update(log_evidence=np.array(log_rss + info['log_detection_evidence']),
        log_detection_evidence=np.array(info['log_detection_evidence']), log_rss_evidence=np.array(log_rss),
        ess=np.array(info['effective_samples']), seed=np.array(seed, dtype=np.uint32),
        proposal_center=center, proposal_root=root, proposal_info=np.array(json.dumps(info, allow_nan=False)))
    return arrays, prior


def save_case(path, arrays):
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


@lru_cache(maxsize=1)
def data_module():
    return importlib.import_module('public_corpus.probe_episodes')


def available_manifest():
    data = data_module()
    manifest = data.verify_freeze(PREPARATION)
    summary = json.loads((PREPARATION / 'summary.json').read_text())
    if not summary['complete'] or summary['events'] != 24:
        raise RuntimeError('Public 24-episode S-only preparation is not complete')
    return manifest


def context(manifest):
    files = [Path(__file__), Path(__file__).with_name('protocol.md'), Path(data_module().__file__),
             *[Path(m.__file__) for m in [history_bank, inference, operator_prior, reference_pmc,
                                         support_integral, source_weight_mixture, transfer]]]
    inputs = [PREPARATION / 'manifest.json', PREPARATION / 'freeze.json',
              PREPARATION / 'summary.json', Path(manifest['parameters'])]
    inputs += [PREPARATION / e['id'] / 'adaptation.json' for e in manifest['events']]
    return dict(created=now(), code_sha256={str(p): digest(p) for p in files},
        inputs_sha256={str(p): digest(p) for p in inputs}, source_preparation=str(PREPARATION),
        model='BankPrior plus AP-specific Gaussian random intercept, S-chosen shared prior scale',
        sigma_grid=SIGMA, features=FEATURES, g='20 * Phi @ theta; theta not fitted',
        sampler=dict(adaptation_steps=6, adaptation_samples=4096, final_samples=32768,
                     defensive_fraction=.1, seed_rule='SHA256(residual-probe-v1|episode_id|APindex) first 4 bytes, little endian'),
        query_scored=False, localization_run=False, shared_model_updates=0,
        reverse_trace='w is existing full-AP S-only adaptation; sigma reads only histories/sx/stats. Phi reads H functions at each S/Q coordinate; Q RSS is copied only. Proposal conditions on S only. No grid field or Q loss.',
        parameter_scope=manifest['parameter_scope'])


def load_episode(episode_id):
    cases, meta = data_module().load_probe_episode(episode_id, PREPARATION)
    adaptation = meta['adaptation']
    assert adaptation['query_used_for_weights'] is False
    assert list(cases) == meta['AP_indices']
    return cases, meta, adaptation['weights']


def event_record(meta, weights, selection):
    paths = [str(OUT / meta['id'] / f'ap{a}.npz') for a in meta['AP_indices']]
    return dict(id=meta['id'], building=meta['building'], target=meta['target'],
        histories=meta['histories'], J=len(meta['histories']), weights=weights,
        intercept_std=selection['intercept_std'], sigma_selection=selection,
        APs=meta['AP_indices'], case_paths=paths, coordinate_unit=meta['coordinate_unit'],
        position_kind=meta['position_kind'], origin=meta['origin'],
        episode_training_weight=meta['episode_training_weight'], AP_training_weight=meta['AP_training_weight'],
        position_training_weight=meta['position_training_weight'], metadata_source=str(PREPARATION / 'manifest.json'),
        weight_source=str(PREPARATION / meta['id'] / 'adaptation.json'), parameters=str(OUT / 'parameters.json'),
        freeze=str(OUT / 'freeze.json'), query_scored=False, source_development_only=True)


def select_for(cases, meta, weights):
    directory = OUT / meta['id']; directory.mkdir(exist_ok=True)
    path = directory / 'event.json'
    if path.exists():
        return json.loads(path.read_text())
    support = {ap: {k: c[k] for k in ['histories', 'sx', 'stats', 'dz']} for ap, c in cases.items()}
    selection = choose_intercept(support, weights, meta['base'], meta['obs'])
    event = event_record(meta, weights, selection)
    write(path, event)
    return event


@torch.no_grad()
def pilot():
    manifest = available_manifest(); OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'pilot_context.json', context(manifest))
    write(OUT / 'parameters.json', {k: manifest[k] for k in ['base', 'obs', 'parameters', 'parameter_scope']})
    episode_id = manifest['events'][0]['id']; started = time.perf_counter()
    cases, meta, weights = load_episode(episode_id)
    event = select_for(cases, meta, weights)
    ap = next(a for a, c in cases.items() if any(h is not None for h in c['histories']))
    case = cases[ap]; base, obs = meta['base'], meta['obs']; sx, qx = case['sx'], case['qx']
    zero = InterceptBankPrior(case['histories'], weights, base, 0.)
    bank = BankPrior(case['histories'], weights, base)
    sm, sf = zero.features(sx); bm, bf = bank.features(sx)
    torch.testing.assert_close(sm, bm, atol=0, rtol=0)
    torch.testing.assert_close(zero.covariance(sx, sx, sf, sf), bank.covariance(sx, sx, bf, bf), atol=0, rtol=0)
    torch.testing.assert_close(zero.diagonal(sx, sf), bank.diagonal(sx, bf), atol=0, rtol=0)
    reversed_case = dict(case, histories=case['histories'][::-1], dz=case['dz'][::-1])
    torch.testing.assert_close(history_features(case, qx, base), history_features(reversed_case, qx, base), atol=1e-10, rtol=1e-10)
    reverse = InterceptBankPrior(reversed_case['histories'], weights[::-1], base, 0.)
    rm, rf = reverse.features(sx)
    torch.testing.assert_close(rm, sm, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(reverse.covariance(sx, sx, rf, rf), zero.covariance(sx, sx, sf, sf), atol=1e-10, rtol=1e-10)
    # J=1 is a vacuous permutation. Exercise a real multi-source AP as well,
    # without generating another proposal or using any Q RSS.
    multi_id = next(e['id'] for e in manifest['events'] if len(e['histories']) > 1)
    multi_cases, multi_meta, multi_weights = load_episode(multi_id)
    multi_ap = next(a for a, c in multi_cases.items() if sum(h is not None for h in c['histories']) > 1)
    mc = multi_cases[multi_ap]
    rc = dict(mc, histories=mc['histories'][::-1], dz=mc['dz'][::-1])
    for points in [mc['sx'], mc['qx']]:
        torch.testing.assert_close(history_features(mc, points, base), history_features(rc, points, base), atol=1e-10, rtol=1e-10)
    mp = InterceptBankPrior(mc['histories'], multi_weights, base, 10.)
    rp = InterceptBankPrior(rc['histories'], multi_weights[::-1], base, 10.)
    mm, mf = mp.features(mc['sx']); rm, rf = rp.features(mc['sx'])
    torch.testing.assert_close(mm, rm, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(mp.covariance(mc['sx'], mc['sx'], mf, mf), rp.covariance(mc['sx'], mc['sx'], rf, rf), atol=1e-10, rtol=1e-10)
    missing = dict(case, histories=[None] * len(weights))
    assert not bool(history_features(missing, sx, base).any())
    local = InterceptBankPrior(missing['histories'], weights, base, 10.)
    lm, lf = local.features(sx); empty = BankPrior([], [], base); em, ef = empty.features(sx)
    torch.testing.assert_close(lm, em, atol=0, rtol=0)
    torch.testing.assert_close(local.covariance(sx, sx, lf, lf), empty.covariance(sx, sx, ef, ef) + 100.)
    sampled = time.perf_counter()
    arrays, prior = prepare_case(case, weights, base, obs, event['intercept_std'], seed_for(episode_id, ap))
    path = OUT / episode_id / f'ap{ap}.npz'; save_case(path, arrays)
    proposal_seconds = time.perf_counter() - sampled
    with np.load(path, allow_pickle=False) as loaded:
        for key, value in arrays.items():
            np.testing.assert_array_equal(loaded[key], value)
    t = {k: torch.from_numpy(arrays[k]) for k in ['sm', 'sk', 'qm', 'qv', 'cross', 'samples', 'log_proposal', 'n', 'k', 's1', 's2']}
    posterior = ImportanceSupport(t['sm'], t['sk'], {k: t[k] for k in ['n', 'k', 's1', 's2']}, obs, t['samples'], t['log_proposal'])
    w = posterior.log_weights.exp(); center = w @ t['samples']; delta = t['samples'] - center
    prediction = predict_from_posterior(t['qm'], t['qv'], t['cross'], t['sm'], t['sk'],
                                       dict(mean=center, cov=(delta.T * w) @ delta))
    reference = field_from_prior(prior, obs, sx, case['stats'], qx, seed_for(episode_id, ap), defensive_fraction=.1)
    errors = {k: float((prediction[k] - reference[k]).abs().max()) for k in ['mean', 'variance']}
    for key in errors:
        torch.testing.assert_close(prediction[key], reference[key], atol=1e-4, rtol=1e-7)
    weight_error = float((posterior.log_weights - torch.from_numpy(arrays['logw'])).abs().max())
    assert weight_error < 1e-5
    total = sum(len(e['AP_indices']) for e in manifest['events'])
    free = shutil.disk_usage(OUT).free
    record = dict(completed=now(), episode=episode_id, AP=ap, intercept_std=event['intercept_std'],
        sigma0_exact=True, source_permutation=True, permutation_episode=multi_id, permutation_AP=multi_ap,
        permutation_J=len(mc['histories']), no_history_feature_zero=True,
        no_history_keeps_random_intercept=True, lossless_npz_roundtrip=True,
        matrix_prediction_max_error=errors, normalized_logw_max_error=weight_error,
        ESS=float(arrays['ess']), samples=len(arrays['samples']), file_bytes=path.stat().st_size,
        proposal_and_save_seconds=proposal_seconds, full_pilot_seconds=time.perf_counter() - started,
        total_episodes=len(manifest['events']), total_AP_proposals=total,
        estimated_total_bytes=path.stat().st_size * total,
        estimated_proposal_seconds=proposal_seconds * total, available_bytes=free,
        query_scored=False, localization_run=False)
    write(OUT / 'pilot.json', record); log(stage='pilot', **record)


def freeze():
    manifest = available_manifest(); current = context(manifest)
    old = json.loads((OUT / 'pilot_context.json').read_text())
    for key in ['code_sha256', 'inputs_sha256']:
        assert current[key] == old[key], key
    audit = json.loads((OUT / 'pilot.json').read_text())
    if audit['estimated_total_bytes'] > shutil.disk_usage(OUT).free:
        raise RuntimeError('Pilot disk estimate exceeds available space; ask main thread to decide')
    write(OUT / 'freeze.json', dict(**current, pilot_sha256=digest(OUT / 'pilot.json'),
                                  pilot_context_sha256=digest(OUT / 'pilot_context.json')))
    (OUT / 'code').mkdir(exist_ok=True)
    for p in current['code_sha256']:
        shutil.copy2(p, OUT / 'code' / Path(p).name)
    log(stage='frozen', proposals=audit['total_AP_proposals'], estimated_bytes=audit['estimated_total_bytes'])


def verify():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    for key in ['code_sha256', 'inputs_sha256']:
        for path, expected in frozen[key].items():
            assert digest(Path(path)) == expected, path
    return frozen


@torch.no_grad()
def run():
    manifest = available_manifest(); verify(); events = []
    for meta0 in manifest['events']:
        verify(); started = time.perf_counter()
        cases, meta, weights = load_episode(meta0['id']); event = select_for(cases, meta, weights)
        directory = OUT / meta['id']; size = 0; minimum = float('inf')
        for i, (ap, case) in enumerate(cases.items()):
            path = directory / f'ap{ap}.npz'
            if not path.exists():
                arrays, _ = prepare_case(case, weights, meta['base'], meta['obs'], event['intercept_std'], seed_for(meta['id'], ap))
                save_case(path, arrays)
            with np.load(path, allow_pickle=False) as a:
                minimum = min(minimum, float(a['ess']))
                assert a['samples'].shape == (32768, 5)
            size += path.stat().st_size
            if i % 50 == 0:
                log(stage='APs', event=meta['id'], complete=i + 1, APs=len(cases), sigma=event['intercept_std'])
        event.update(prepared=True, completed=now(), seconds=time.perf_counter() - started,
                     proposal_bytes=size, min_ESS=minimum)
        write(directory / 'completed.json', event); events.append(event)
        write(OUT / 'manifest.json', dict(events=events, complete=len(events) == len(manifest['events']),
            parameters=str(OUT / 'parameters.json'), features=FEATURES, query_scored=False,
            shared_updates=0, scope='Frozen source matrices and full proposals only'))
        log(stage='completed', event=meta['id'], APs=len(cases), sigma=event['intercept_std'], bytes=size, min_ESS=minimum)
    write(OUT / 'summary.json', dict(completed=now(), complete=True, episodes=len(events),
        AP_proposals=sum(len(e['APs']) for e in events), proposal_bytes=sum(e['proposal_bytes'] for e in events),
        min_ESS=min(e['min_ESS'] for e in events), seconds=sum(e['seconds'] for e in events),
        intercept_std_by_event={e['id']: e['intercept_std'] for e in events},
        query_scored=False, localization_run=False, g_fitted=False, samples_per_AP=32768))
    log(stage='all_prepared', episodes=len(events), proposals=sum(len(e['APs']) for e in events))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical preparation stays on ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['pilot', 'freeze', 'run'])
    args = parser.parse_args(); globals()[args.stage]()
