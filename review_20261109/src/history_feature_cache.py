"""Build frozen, full-H source features once per floor on ray-web CUDA0.

This is a helper, not a runnable training entry point. The caller owns the wall
deadline and failure record. A partially written manifest is never complete.
Every required S/Q row is encoded unlabeled; validation coordinates are stored
for later evaluation and never enter H preprocessing or model adaptation.
"""
import csv
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from source_episode_catalog import make_episode
from tabicl_history import PACKAGE, state_hash
from tabicl_history_features import HistoryFeatures
from recomposition_trial import ROOT, CSV, digest, now, write


SOURCE = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
CATALOG = ROOT/'results/source_episode_catalog_20261105/catalog.json'
CODE = ('history_feature_cache.py', 'tabicl_history_features.py',
        'tabicl_history.py', 'tabicl_history_learning.py', 'source_episode_catalog.py',
        'acquisition_tasks.py', 'uji.py', 'recomposition_trial.py')
INFERENCE = {
    name: dict(device='cuda:0', use_amp=False, use_fa3=False,
               min_batch_size=1, safety_factor=.5, use_async=False,
               offload='cpu' if name == 'COL_CONFIG' else 'gpu')
    for name in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG')
}


def _event_rows(catalog, fit_specs, validation_specs):
    protected = set(catalog['protected_rows'])
    floors = catalog['source_floors']
    assert floors == ['B0F0', 'B0F1', 'B0F2']
    result = {floor: dict(history_rows=catalog['history_rows'][floor],
        fit_support=set(), fit_query=set(), validation_support=set(),
        validation_query=set(), events=[]) for floor in floors}
    for floor, record in result.items():
        assert floor not in record['history_rows']
        for name, rows in record['history_rows'].items():
            assert name in floors and not protected.intersection(rows)
    for split, specs in (('fit', fit_specs), ('validation', validation_specs)):
        for spec in specs:
            assert spec['phase'] == split
            event = make_episode(catalog, **spec)
            record = result[event['floor']]
            assert event['history_rows'] == record['history_rows']
            s, q = set(event['support_rows']), set(event['query_rows'])
            assert not protected.intersection(s)
            assert not s.intersection(q)
            if split == 'fit':
                assert not protected.intersection(q)
            else:
                assert q <= protected
            record[split+'_support'].update(s)
            record[split+'_query'].update(q)
            record['events'].append(dict(id=event['id'], split=split,
                support_rows=event['support_rows'], query_rows=event['query_rows']))
    for record in result.values():
        keys = ('fit_support', 'fit_query', 'validation_support', 'validation_query')
        record['row_ids'] = sorted(set().union(*(record[k] for k in keys)))
        for key in keys:
            record[key] = sorted(record[key])
    return result


def _read_rows(provenance):
    allowed = {}
    for floor, record in provenance.items():
        for name, ids in record['history_rows'].items():
            for row_id in ids:
                assert allowed.setdefault(row_id, name) == name
        for row_id in record['row_ids']:
            assert allowed.setdefault(row_id, floor) == floor
    rows = {}
    with open(CSV, newline='') as handle:
        for row_id, row in enumerate(csv.DictReader(handle), 1):
            if row_id not in allowed:
                continue
            name = f"B{row['BUILDINGID']}F{row['FLOOR']}"
            assert name == allowed[row_id] and row['BUILDINGID'] == '0'
            rows[row_id] = dict(
                rss=np.array([int(row[f'WAP{i:03}']) for i in range(1, 521)], dtype=np.int16),
                xy=np.array([float(row['LONGITUDE']), float(row['LATITUDE'])], dtype=np.float64),
            )
    assert rows.keys() == allowed.keys()
    return rows


def build_cache(output_dir, catalog, fit_specs, validation_specs, checkpoint, seed):
    """Return a manifest indexing floor.npz(row_ids, mean, phi, xy).

    output_dir must be a new dedicated cache directory. Specs are the frozen
    source schedule flattened to 72 fit events and the same 27 validation events.
    No target-building data, optimizer, model forward retry, or CPU fallback.
    The official inference manager may split independent columns/rows internally;
    the entire ICL H sequence and all APs are retained on every call.
    """
    assert socket.gethostname() == 'ray-web'
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    source = json.loads((SOURCE/'complete.json').read_text())
    source_freeze = json.loads((SOURCE/'freeze.json').read_text())
    assert source == json.loads((SOURCE/'final.json').read_text())
    assert source['complete'] and source['formal_step'] == source['source_updates'] == 24
    assert source['source_episodes'] == 72 and source['source']['source_passed']
    assert digest(SOURCE/'freeze.json') == source['freeze_sha256']
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == source['checkpoint_sha256']['24']
    assert catalog == json.loads(CATALOG.read_text())
    assert fit_specs == [s for batch in source_freeze['schedule'] for s in batch]
    assert validation_specs == source_freeze['validation']
    assert len(fit_specs) == 72 and len(validation_specs) == 27
    for path in (CSV, CATALOG):
        assert digest(path) == source_freeze['inputs'][str(path)]
    for name in CODE:
        if name in source_freeze['code']:
            assert digest(ROOT/name) == source_freeze['code'][name], name
    package = {str(p.relative_to(PACKAGE)):digest(p)
               for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package and package == source_freeze['package']
    provenance = _event_rows(catalog, fit_specs, validation_specs)

    # The present resource decision assumes an available card, not the earlier
    # 1.7–2 GiB shared-card condition. Refuse a changed condition before loading.
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info(0)
    if free < 8*1024**3:
        raise RuntimeError('Full-H cache requires at least 8 GiB currently free on CUDA0')
    torch.cuda.set_per_process_memory_fraction(.75, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    freeze = dict(created=now(), seed=int(seed), code={n:digest(ROOT/n) for n in CODE},
        package=package,
        inputs={str(p):digest(p) for p in (CSV, CATALOG, checkpoint,
            SOURCE/'complete.json', SOURCE/'final.json', SOURCE/'freeze.json')},
        checkpoint=dict(path=str(checkpoint), sha256=checkpoint_hash,
                        source_step=24, state_sha256=source['trained_state_sha256']),
        provenance=provenance, fit_specs=fit_specs, validation_specs=validation_specs,
        AP_schema=list(range(520)), views_per_axis=8, axes=2,
        resources=dict(device='cuda:0', CPU=2, max_gpu_fraction=.75,
            minimum_free_bytes=8*1024**3, initial_free_bytes=free, total_device_bytes=total,
            inference_config=INFERENCE, tf32=False, query_chunking=False,
            wall_deadline='Owned by calling runner', downloads=0),
        scope='Full source H, frozen final24 model; validation rows encoded without labels; no fitting or scoring',
        new_training_updates=0, localization_scores=0)
    write(output_dir/'freeze.json', freeze)
    for name in CODE:
        (output_dir/name).write_bytes((ROOT/name).read_bytes())
    rows = _read_rows(provenance)
    model = HistoryFeatures(checkpoint, device='cuda:0')
    before = state_hash(model.loaded.model_)
    assert before == source['trained_state_sha256']
    manifest = dict(complete=False, created=now(), freeze_sha256=digest(output_dir/'freeze.json'),
        checkpoint_sha256=checkpoint_hash, model_state_sha256=before, floors={},
        new_training_updates=0, localization_scores=0)
    write(output_dir/'manifest.json', manifest)
    for floor, record in provenance.items():
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        floor_started = time.monotonic()
        histories = {name:dict(
            rss=np.stack([rows[i]['rss'] for i in ids]),
            xy=np.stack([rows[i]['xy'] for i in ids]),
        ) for name, ids in record['history_rows'].items()}
        hxy = np.concatenate([h['xy'] for h in histories.values()])
        bounds = np.stack((hxy.min(0), hxy.max(0)))
        model.fit_history(histories, floor, np.arange(520), bounds, seed)
        for estimator in model.estimators:
            estimator.inference_config_.update_from_dict(INFERENCE)
        assert all(not p.requires_grad for p in model.loaded.model_.parameters())
        row_ids = np.asarray(record['row_ids'], dtype=np.int64)
        encoded = model.encode(np.stack([rows[i]['rss'] for i in row_ids]))
        torch.cuda.synchronize(0)
        seconds = time.monotonic()-floor_started
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        assert encoded['mean'].shape == (len(row_ids), 2)
        assert encoded['phi'].shape == (len(row_ids), 2, 8, model.feature_dim)
        assert state_hash(model.loaded.model_) == before
        path = output_dir/f'{floor}.npz'
        np.savez(path, row_ids=row_ids, mean=encoded['mean'], phi=encoded['phi'],
                 xy=np.stack([rows[i]['xy'] for i in row_ids]))
        manifest['floors'][floor] = dict(path=path.name, sha256=digest(path),
            history_bounds=bounds.tolist(),
            H_rows=sum(len(v) for v in record['history_rows'].values()),
            encoded_rows=len(row_ids), feature_shape=list(encoded['phi'].shape),
            history_rows=record['history_rows'], row_ids=record['row_ids'],
            sources={k:record[k] for k in ('fit_support', 'fit_query',
                'validation_support', 'validation_query')},
            seconds=seconds, peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved, model_state_sha256=before)
        manifest['seconds'] = time.monotonic()-started
        write(output_dir/'manifest.json', manifest)
        print(json.dumps(dict(stage='history_feature_cache', floor=floor,
            **{k:manifest['floors'][floor][k] for k in
               ('H_rows', 'encoded_rows', 'seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')})),
            flush=True)
        del encoded
    assert len(manifest['floors']) == 3
    assert state_hash(model.loaded.model_) == before
    manifest.update(complete=True, completed=now(), seconds=time.monotonic()-started,
                    model_state_unchanged=True)
    write(output_dir/'manifest.json', manifest)
    return manifest
