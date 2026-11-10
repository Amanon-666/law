"""Fixed Hurdle + retained GP deployment on reserved IPIN2025 streams.

prepare uses only the shared frozen Training H/S assets. predict requires all
15 contexts, seals all 30 floor/K/log streams, and never opens scoring labels.
The separate ipin2025_score entry point is the only GT consumer. Source model
weights, GP parameters, adaptation and half/half pooling are unchanged.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

import predict as gp
from building_information_tab_trial import digest, now, write
from fields import History
from frozen_hurdle_translator import FrozenHurdleTranslator, PreparedFrozenHurdle
from hurdle_gp_pool import fuse_hurdle_gp
from ipin2025_observation_data import read_query_stream, rss_matrix
from position_fusion_gp import prepare_from_scans, move_task, _retained_field, score_retained_fields
from uji import tensor

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'results/ipin2025_history_assets_20261110'
SOURCE = ROOT/'results/claude_hurdle_b0_ens5_20261110'
PARAMETERS = ROOT/'results/paired_assets_20261106/parameters.json'
OLD = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
RESERVATION = OLD/'results/history_learning_stage2_20261027/validation_reservation.json'
METHODS = ['hurdle', 'GP', 'GP_candidates', 'coordinate_half', 'probability_half']
CODE = ('ipin2025_confirmation.py', 'ipin2025_observation_data.py', 'ipin2025_score.py',
        'frozen_hurdle_translator.py', 'vertical_translator/translator.py', 'hurdle_gp_pool.py',
        'position_fusion_gp.py', 'building_information_gp_trial.py', 'predict.py',
        'adaptation.py', 'fields.py', 'observations.py', 'posterior.py', 'uji.py',
        'episodic_tasks.py', 'building_information_tab_trial.py')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return dict(path=str(path), sha256=digest(path))


def freeze(output):
    assets = read(ASSETS/'complete.json')
    assert assets['complete'] and not assets['official_query_opened'] and not assets['GT_opened']
    assert digest(ASSETS/'freeze.json') == assets['freeze_sha256']
    assert digest(ASSETS/'manifest.json') == assets['manifest_sha256']
    assert digest(ASSETS/'histories.json') == assets['histories_sha256']
    manifest = read(ASSETS/'manifest.json')
    assert manifest['eligible_floors'] == [2, 3, 4, 5, 6]
    reserved = [r for r in read(RESERVATION)['files'] if r['year'] == 2025]
    logs = []
    for trial in ('03', '04'):
        query, = [r for r in reserved if r['role'] == 'sealed_query_scans'
                  and f'ScoringTrial{trial}' in r['path']]
        truth, = [r for r in reserved if r['role'] == 'sealed_query_labels'
                  and f'ScoringTrial{trial}' in r['path']]
        # Copy prior reservation identities; neither file is opened here.
        logs.append(dict(id=trial, query_path=query['path'], query_sha256=query['sha256'],
                         gt_path=truth['path'], gt_sha256=truth['sha256']))
    output.mkdir(exist_ok=False)
    paths = [ASSETS/name for name in ('complete.json', 'freeze.json', 'manifest.json', 'histories.json')]
    paths += [PARAMETERS, RESERVATION, SOURCE/'complete.json', SOURCE/'freeze.json', SOURCE/'deployment_check.json']
    write(output/'freeze.json', dict(created=now(), eligible_floors=manifest['eligible_floors'],
        K=[1, 3, 5], R=1, methods=METHODS, logs=logs,
        inputs=[binding(p) for p in paths], code={name:digest(ROOT/name) for name in CODE},
        contract=dict(method='Original B0 seeds1..5 Hurdle; retained GP; fixed half log pool',
            source_updates=0, parameter_search=False, history='Shared frozen FPS128 per other floor',
            support='Nested hash-ranked actual Training receipts K1/3/5, R1',
            axes='BSSID union of selected H and current K S only; ignore Query-only APs',
            geometry='9011 common metric frame, H/currentS bounds rounded on absolute 2m lattice',
            timing='Absolute 2Hz; last already received WiFi snapshot; sensor AppTimestamp horizon',
            coverage='All eligible floors independently predicted; Level1 stays in site denominator',
            scoring='After all30 streams sealed; known true floor 2D, last time <= GT+.01',
            primary='probability_half vs both constituent paths and fixed coordinate_half',
            metrics='Covered mean/p90 and full-site coverage, by K/log/floor; no outcome-based exclusions',
            source_selection='B0-only reader selection corroborates fixed constants; UJI remains development',
            resources='Existing offline assets; CPU2 and at most one naturally idle GPU; explicit bounded resume')))


def verify_freeze(output):
    frozen = read(output/'freeze.json')
    assert all(digest(ROOT/name) == sha for name, sha in frozen['code'].items())
    assert all(digest(r['path']) == r['sha256'] for r in frozen['inputs'])
    return frozen


def context(manifest, floor, k):
    """Reindex each source's own BSSIDs; S5 must not enlarge a smaller K axis."""
    raw = {}
    for source, record in manifest['histories'].items():
        if int(source) == floor:
            continue
        assert digest(record['path']) == record['sha256']
        with np.load(record['path']) as a:
            raw[f'IPIN2025_B2025F{source}'] = {key:a[key] for key in ('xy', 'rss', 'ap_ids')}
    srecord = manifest['support'][str(floor)]
    assert digest(srecord['path']) == srecord['sha256']
    with np.load(srecord['path']) as a:
        support = dict(xy=a['xy'][:k], rss=a['rss'][:k], ap_ids=a['ap_ids'])
    ap_ids = sorted(set().union(*(set(h['ap_ids']) for h in raw.values()),
        set(support['ap_ids'][(support['rss'] != 100).any(0)])))
    columns = {ap:i for i, ap in enumerate(ap_ids)}

    def align(h):
        rss = np.full((len(h['rss']), len(ap_ids)), 100.)
        for j, ap in enumerate(h['ap_ids']):
            if ap in columns:
                rss[:, columns[ap]] = h['rss'][:, j]
        return dict(xy=h['xy'], rss=rss)

    return {name:align(h) for name,h in raw.items()}, align(support), ap_ids


def prepare(args):
    output = args.output
    if not args.resume:
        freeze(output)
    frozen = verify_freeze(output)
    assert not (output/'contexts_complete.json').exists()
    manifest, index = read(ASSETS/'manifest.json'), read(ASSETS/'histories.json')
    assert index['parameters_sha256'] == digest(PARAMETERS)
    parameters = read(PARAMETERS)
    records = {(r['floor'], r['ap_id']):r for r in index['fields']}
    translator = FrozenHurdleTranslator(SOURCE)
    start, contexts = time.monotonic(), []
    completed_this_segment = 0
    for floor in frozen['eligible_floors']:
        for k in frozen['K']:
            eid = f'IPIN2025_B2025F{floor}_K{k}'
            directory = output/eid
            complete_path = directory/'complete.json'
            if complete_path.exists():
                saved = read(complete_path)
                assert saved['freeze_sha256'] == digest(output/'freeze.json')
                contexts.append(dict(event=eid, floor=floor, K=k, **binding(complete_path)))
                continue
            if time.monotonic()-start >= args.seconds or completed_this_segment >= args.max_contexts:
                return 2
            directory.mkdir(exist_ok=True)
            tick = time.monotonic()
            histories, support, ap_ids = context(manifest, floor, k)
            hpath = directory/'hurdle.npz'
            if not hpath.exists():
                h = translator.prepare(histories, support, f'IPIN2025_B2025F{floor}')
                np.savez_compressed(hpath, cand_xy=h.cand_xy, aps=h.aps, p=h.p, m=h.m,
                                    s2=h.s2, bounds=h.bounds, ap_count=h.ap_count)
            with np.load(hpath) as a:
                bounds = a['bounds']

            def load(source, ap):
                source_floor = source.rsplit('F', 1)[1]
                record = records[(source_floor, ap_ids[ap])]
                assert digest(record['path']) == record['sha256']
                with np.load(record['path']) as a:
                    return History(tensor(a['z']), dict(mean=tensor(a['mean']), cov=tensor(a['cov'])),
                                   parameters['base'])

            task = prepare_from_scans(histories, support, parameters['base'], parameters['obs'],
                                      load, np.zeros(2), bounds)
            assert np.array_equal(task['aps'], np.arange(len(ap_ids)))
            state_path = directory/'state.json'
            if not state_path.exists():
                state_start = time.monotonic()
                state = gp.adapt(task['cases'], task['base'], task['obs'])
                write(state_path, dict(state=state, ap_ids=ap_ids, sources=list(histories),
                    floor=floor, K=k, seconds=time.monotonic()-state_start, query_used=False))
            state_record = read(state_path)
            assert state_record['ap_ids'] == ap_ids
            state = state_record['state']
            move_task(task, torch.device(args.device))
            progress = directory/'fields.jsonl'
            fields = [json.loads(s) for s in progress.read_text().splitlines()] if progress.exists() else []
            done = {r['ap'] for r in fields}
            assert len(done) == len(fields)
            assert all(digest(r['path']) == r['sha256'] for r in fields)
            for ap, case in task['cases'].items():
                if ap in done:
                    continue
                if time.monotonic()-start >= args.seconds:
                    print(json.dumps(dict(event=eid, fields=len(fields), total=len(ap_ids),
                        stop='segment_budget', explicit_resume_required=True)), flush=True)
                    return 2
                field_start = time.monotonic()
                value, diagnostic = _retained_field(task, state, ap, case, eid, k, None)
                path = directory/f'gp_ap{ap}.npz'
                np.savez_compressed(path, mean=value['mean'].cpu().numpy(),
                                    variance=value['variance'].cpu().numpy())
                record = dict(ap=ap, ap_id=ap_ids[ap], **binding(path),
                    seconds=time.monotonic()-field_start, diagnostic=diagnostic)
                with progress.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                fields.append(record)
                if len(fields)%50 == 0:
                    print(json.dumps(dict(event=eid, fields=len(fields), total=len(ap_ids))), flush=True)
            grid_path = directory/'grid.npz'
            np.savez_compressed(grid_path, grid=task['absolute_grid'], aps=task['aps'], bounds=task['bounds'])
            write(complete_path, dict(complete=True, event=eid, floor=floor, K=k, ap_ids=ap_ids,
                freeze_sha256=digest(output/'freeze.json'), state=binding(state_path),
                hurdle=binding(hpath), grid=binding(grid_path), fields=sorted(fields, key=lambda r:r['ap']),
                obs=task['obs'], sources=list(histories), query_used=False,
                seconds_this_segment=time.monotonic()-tick))
            contexts.append(dict(event=eid, floor=floor, K=k, **binding(complete_path)))
            completed_this_segment += 1
            print(json.dumps(dict(event=eid, complete=True, contexts=len(contexts))), flush=True)
    write(output/'contexts_complete.json', dict(complete=True, created=now(), contexts=contexts,
        freeze_sha256=digest(output/'freeze.json'), official_query_opened=False, GT_opened=False))
    return 0


def load_prepared(record, device):
    assert digest(record['path']) == record['sha256']
    complete = read(record['path'])
    for value in [complete['hurdle'], complete['grid'], complete['state'], *complete['fields']]:
        assert digest(value['path']) == value['sha256']
    with np.load(complete['hurdle']['path']) as a:
        h = PreparedFrozenHurdle(*(a[key] for key in ('cand_xy', 'aps', 'p', 'm', 's2', 'bounds')),
                                  int(a['ap_count']))
    means, variances = [], []
    for field in complete['fields']:
        with np.load(field['path']) as a:
            means.append(a['mean'])
            variances.append(a['variance'])
    with np.load(complete['grid']['path']) as a:
        prepared = dict(mean=tensor(np.stack(means)).to(device), variance=tensor(np.stack(variances)).to(device),
            grid=tensor(a['grid']).to(device), aps=a['aps'], origin=np.zeros(2), bounds=a['bounds'],
            obs=complete['obs'], diagnostics=[r['diagnostic'] for r in complete['fields']])
    return h, prepared, complete['ap_ids']


def predict_streams(args):
    output = args.output
    frozen = verify_freeze(output)
    contexts = read(output/'contexts_complete.json')
    assert contexts['complete'] and contexts['freeze_sha256'] == digest(output/'freeze.json')
    expected = {(f,k) for f in frozen['eligible_floors'] for k in frozen['K']}
    assert {(r['floor'],r['K']) for r in contexts['contexts']} == expected
    assert len(contexts['contexts']) == len(expected)
    assert not (output/'predictions.json').exists()
    progress = output/'predictions_progress.json'
    saved = read(progress) if args.resume and progress.exists() else []
    assert args.resume or not progress.exists()
    assert all(digest(r['path']) == r['sha256'] for r in saved)
    done = {(r['log_id'],r['floor'],r['K']) for r in saved}
    start = time.monotonic()
    streams = {}
    for log in frozen['logs']:
        assert digest(log['query_path']) == log['query_sha256']
        streams[log['id']] = read_query_stream(log['query_path'])
    for record in contexts['contexts']:
        h, prepared, ap_ids = load_prepared(record, torch.device(args.device))
        for log in frozen['logs']:
            key = log['id'], record['floor'], record['K']
            if key in done:
                continue
            if time.monotonic()-start >= args.seconds:
                return 2
            stream = streams[log['id']]
            used = stream['used_snapshot_indices']
            query = rss_matrix([stream['snapshots'][i] for i in used], ap_ids)
            predictions = {method:[] for method in METHODS}
            for begin in range(0, len(query), 16):
                rss = query[begin:begin+16]
                hp = h.predict(rss)
                g = score_retained_fields(prepared, rss)
                fused = fuse_hurdle_gp(h.cand_xy, hp['log_position_mass'],
                    prepared['grid'].cpu().numpy(), g['scores'], g['projected'], h.bounds)
                for method in METHODS:
                    predictions[method].append(fused['predictions'][method])
            expand = np.searchsorted(used, stream['snapshot_indices'])
            predictions = {m:(np.concatenate(p)[expand] if p else np.empty((0,2))) for m,p in predictions.items()}
            path = output/f"{record['event']}_log{log['id']}.npz"
            np.savez_compressed(path, times=stream['prediction_times'],
                snapshot_times=np.asarray([stream['snapshots'][i]['app_timestamp'] for i in stream['snapshot_indices']]),
                **predictions)
            saved.append(dict(log_id=log['id'],floor=record['floor'],K=record['K'], **binding(path),
                receipt_predictions=len(used), ticks=len(stream['prediction_times']),
                duplicate_bssid_rows=sum(r['duplicate_bssid_rows'] for r in stream['snapshots']),
                conflicting_rss_rows=sum(r['conflicting_rss_rows'] for r in stream['snapshots'])))
            write(progress, saved)
            print(json.dumps(dict(log=log['id'],event=record['event'],sealed=len(saved))), flush=True)
    assert len(saved) == 2*len(expected)
    write(output/'predictions.json', dict(complete=True,phase='predictions',created=now(),
        freeze_sha256=digest(output/'freeze.json'),
        contexts_complete_sha256=digest(output/'contexts_complete.json'),
        eligible_floors=frozen['eligible_floors'], K=frozen['K'], methods=METHODS,
        logs=frozen['logs'], predictions=saved, GT_opened=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'predict'))
    parser.add_argument('--output',type=Path,default=ROOT/'results/ipin2025_confirmation_20261110')
    parser.add_argument('--seconds',type=float,default=780.)
    parser.add_argument('--max-contexts',type=int,default=15)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert socket.gethostname() == 'ray-web' and 0 < args.seconds <= 780
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    raise SystemExit(prepare(args) if args.phase == 'prepare' else predict_streams(args))
