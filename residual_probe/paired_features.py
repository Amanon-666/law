"""H-only RSS/elevation pairing features for frozen source probe episodes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from public_corpus.probe_episodes import load_probe_episode, verify_freeze


ROOT = Path(os.environ['LAW_ROOT'])
SOURCE = ROOT / 'results/source_residual_probe_v1_20261026'
PREPARATION = SOURCE / 'preparation'
PREPARED = SOURCE / 'prepared'
OUT = SOURCE / 'paired_features'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def paired_mean(means, present, dz):
    """means: [J,N]; missing H rows are ignored, denominator remains J."""
    scale = present[:, None] * (means + 75.) / 20.
    z = dz[:, None] / 3.
    return torch.stack(((scale * z).mean(0), (scale * z.abs()).mean(0)), -1)


@torch.no_grad()
def features(case, x, base):
    histories = case['histories']
    if not any(h is not None for h in histories):
        return x.new_zeros((len(x), 2))
    means = torch.stack([x.new_zeros(len(x)) if h is None else
                         base['mean'] + h.projection(x) @ (h.m - base['mean'])
                         for h in histories])
    present = x.new_tensor([h is not None for h in histories])
    dz = x.new_tensor(case['dz'])
    return paired_mean(means, present, dz)


def checks():
    means = torch.tensor([[-55., -65.], [-95., -85.]], dtype=torch.float64)
    present = torch.tensor([1., 1.], dtype=torch.float64)
    dz = torch.tensor([-1., 2.], dtype=torch.float64)
    torch.testing.assert_close(paired_mean(means, present, dz),
                               paired_mean(means.flip(0), present.flip(0), dz.flip(0)),
                               atol=0., rtol=0.)
    assert not bool(paired_mean(means, present * 0., dz).any())
    swapped = means.flip(0)
    torch.testing.assert_close(means.mean(0), swapped.mean(0), atol=0., rtol=0.)
    assert not torch.equal(paired_mean(means, present, dz),
                           paired_mean(swapped, present, dz))


def freeze():
    manifest = verify_freeze(PREPARATION)
    old = json.loads((PREPARATION / 'freeze.json').read_text())
    history_paths = {str(Path(e['history_metadata'][str(f)]['cache']) / f'ap{ap}.npz')
                     for e in manifest['events'] for f in e['histories'] for ap in e['AP_indices']}
    histories = {path: old['inputs_sha256'][path] for path in sorted(history_paths)}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'freeze.json'
    if target.exists():
        raise FileExistsError(target)
    record = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
                  source='public_corpus.probe_episodes.load_probe_episode; frozen 24 source episodes',
                  script=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)),
                  loader_script=str(Path(load_probe_episode.__code__.co_filename)),
                  loader_script_sha256=digest(Path(load_probe_episode.__code__.co_filename)),
                  preparation_manifest=str(PREPARATION / 'manifest.json'),
                  preparation_manifest_sha256=digest(PREPARATION / 'manifest.json'),
                  preparation_freeze=str(PREPARATION / 'freeze.json'),
                  preparation_freeze_sha256=digest(PREPARATION / 'freeze.json'),
                  prepared_manifest=str(PREPARED / 'manifest.json'),
                  prepared_manifest_sha256=digest(PREPARED / 'manifest.json'),
                  history_cache_sha256=histories,
                  episodes=len(manifest['events']), APs=sum(len(e['AP_indices']) for e in manifest['events']),
                  features=['cross_signed', 'cross_abs'], denominator='all J including missing H',
                  inputs=['histories', 'dz', 'sx', 'qx'], query_RSS_used=False,
                  training=False, scoring=False, source_development_only=True)
    write_json(target, record)
    print(json.dumps(dict(stage='freeze', episodes=record['episodes'], APs=record['APs'],
                          history_cache_files=len(histories))), flush=True)


def frozen():
    record = json.loads((OUT / 'freeze.json').read_text())
    for path_key, hash_key in [('script', 'script_sha256'), ('loader_script', 'loader_script_sha256'),
                               ('preparation_manifest', 'preparation_manifest_sha256'),
                               ('preparation_freeze', 'preparation_freeze_sha256'),
                               ('prepared_manifest', 'prepared_manifest_sha256')]:
        assert digest(Path(record[path_key])) == record[hash_key], path_key
    return record


def save_case(path, case, base):
    cross_s = features(case, case['sx'], base).numpy()
    cross_q = features(case, case['qx'], base).numpy()
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, cross_s=cross_s, cross_q=cross_q)
    temporary.replace(path)


def pilot():
    record = frozen()
    checks()
    manifest = json.loads((PREPARATION / 'manifest.json').read_text())
    chosen = [manifest['events'][0], next(e for e in manifest['events'] if len(e['histories']) > 1),
              manifest['events'][-1]]
    examples = []
    for event in chosen:
        cases, meta = load_probe_episode(event['id'], PREPARATION)
        ap = next(a for a, c in cases.items() if any(h is not None for h in c['histories']))
        case = cases[ap]
        permuted = dict(histories=case['histories'][::-1], dz=case['dz'][::-1])
        for points in [case['sx'], case['qx']]:
            torch.testing.assert_close(features(case, points, meta['base']),
                                       features(permuted, points, meta['base']), atol=1e-12, rtol=1e-12)
        empty = dict(histories=[None] * len(case['histories']), dz=case['dz'])
        assert not bool(features(empty, case['qx'], meta['base']).any())
        directory = OUT / event['id']
        directory.mkdir(exist_ok=True)
        path = directory / f'ap{ap}.npz'
        save_case(path, case, meta['base'])
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {'cross_s', 'cross_q'}
            assert arrays['cross_s'].shape == (5, 2) and arrays['cross_q'].shape == (12, 2)
        examples.append(dict(event=event['id'], AP=ap, bytes=path.stat().st_size))
    write_json(OUT / 'pilot.json', dict(examples=examples, permutation=True,
                                       all_H_empty=True, same_mean_different_pairing=True,
                                       freeze_sha256=digest(OUT / 'freeze.json')))
    print(json.dumps(dict(stage='pilot', examples=examples)), flush=True)


def run():
    record = frozen()
    assert (OUT / 'pilot.json').exists()
    manifest = json.loads((PREPARATION / 'manifest.json').read_text())
    start = time.perf_counter()
    events = []
    for event in manifest['events']:
        cases, meta = load_probe_episode(event['id'], PREPARATION)
        directory = OUT / event['id']
        directory.mkdir(exist_ok=True)
        for ap, case in cases.items():
            path = directory / f'ap{ap}.npz'
            if not path.exists():
                save_case(path, case, meta['base'])
        events.append(dict(id=event['id'], APs=len(cases)))
        print(json.dumps(dict(stage='event', **events[-1])), flush=True)
    count = sum(e['APs'] for e in events)
    assert len(events) == record['episodes'] and count == record['APs']
    size = sum(p.stat().st_size for p in OUT.glob('*/ap*.npz'))
    write_json(OUT / 'summary.json', dict(complete=True, events=events, episodes=len(events),
                                         APs=count, npz_bytes=size, seconds=time.perf_counter() - start,
                                         freeze_sha256=digest(OUT / 'freeze.json'),
                                         query_RSS_used=False, training=False, scoring=False,
                                         prepared_proposals_copied=False))
    print(json.dumps(dict(stage='complete', episodes=len(events), APs=count, bytes=size,
                          seconds=time.perf_counter() - start)), flush=True)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical feature generation stays on ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'pilot', 'run'])
    args = parser.parse_args()
    globals()[args.stage]()
