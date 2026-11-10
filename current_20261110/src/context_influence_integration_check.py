"""One original source task, no learning or query-coordinate access.

Checks the actual H/S preparation + GP/Tab/half calling path with a zero
relation projection, against disabled original Tab and an existing GP output.
This is an integration check, not a localization experiment or permission to
start the paused source-training comparison. No optimizer or backward call.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import gc
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/home/panyushuo/projects/panyushuo/ReFi-UJI-Core-20261031')
OVERLAY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OVERLAY))

import numpy as np
import torch

from building_information_tab_trial import digest, now, read_scans, write
from building_information_gp_trial import move_task
from context_influence_consumer import InfluenceHistoryRegressor
from context_influence_system import InfluencePositionSystem
from position_fusion_trial import prepare, CHECKPOINT, PARAMETERS, CATALOG, CSV, SOURCE_ASSETS, SOURCE_REUSE
from tabicl_history import state_hash

EVENT = 'B0F0_group0_to_B0F0_group2_draw2_K1_HB0F1-B0F2_validation'
OUT = ROOT/'results/context_influence_integration_20261110'
CODE = ('context_influence_adapter.py', 'context_influence_features.py',
        'context_influence_consumer.py', 'context_influence_system.py',
        'position_fusion_gp.py', 'context_influence_integration_check.py')
EXPECTED_STATE = 'fc89f3fbe84b0d1a46b480d415b97ed0f9aeb09b0c4f1139a0acd1abaa34f0cd'
EXPECTED_GP = '22a86b56fbcda4e54783387330f3c64ddfe66b92d9533df0deda9abec8f42d83'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert socket.gethostname() == 'ray-web' and torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    started = time.monotonic()
    source_freeze = read(SOURCE_REUSE/'freeze.json')['contract']
    event = next(e for e in source_freeze['source_events'] if e['id'] == EVENT)
    assert event['K'] == 1 and len(event['query_rows']) == 18
    protected = set(read(CATALOG)['protected_rows'])
    assert set(event['query_rows']) <= protected
    assert not protected.intersection(event['support_rows'])
    assert all(not protected.intersection(rows) for rows in event['history_rows'].values())
    index = read(SOURCE_ASSETS/'histories.json')
    assert event['history_rows'] == {f: index['history_rows'][f] for f in event['histories']}
    assert index['parameters_sha256'] == digest(PARAMETERS)
    state_path = SOURCE_REUSE/'source'/f'{EVENT}_state.json'
    gp_path = SOURCE_REUSE/'source'/f'{EVENT}_gp.npz'
    assert digest(state_path) == EXPECTED_STATE and digest(gp_path) == EXPECTED_GP
    assert source_freeze['inputs'][str(CHECKPOINT)] == digest(CHECKPOINT)
    state = read(state_path)['state']
    OUT.mkdir(exist_ok=False)
    freeze = dict(created=now(), event=event, scope='One source integration check, no query coordinates or scoring',
        code={name:digest(OVERLAY/name) for name in CODE},
        inputs={str(p):digest(p) for p in (SOURCE_ASSETS/'histories.json', PARAMETERS,
            CATALOG, CHECKPOINT, state_path, gp_path)},
        protocol='Existing source H/S and GP state; one original S posterior pass; full24 frozen; zero relation projection',
        comparisons='Enabled-zero vs disabled original consumer; prepared GP vs its existing same-event output',
        numerical_gate_max_absolute_coordinate_difference=1e-4,
        resources=dict(gpu=1, cpu_threads=2, hard_seconds=600, downloads=0, optimizer_steps=0),
        forbidden=['Query XY', 'source-protected gradient', 'target data', 'new H/GP fitting', 'optimizer', 'retry'])
    write(OUT/'freeze.json', freeze)
    for name in CODE:
        (OUT/name).write_bytes((OVERLAY/name).read_bytes())

    histories, support, tab_bounds, aps, task = prepare(event, 'source', None, index, read(PARAMETERS))
    assert len(aps) == 170 and sum(len(h['rss']) for h in histories.values()) == 2085
    assert len(support['rss']) == 3
    move_task(task, torch.device('cuda:0'))
    consumer = InfluenceHistoryRegressor(CHECKPOINT, device='cuda:0')
    before = state_hash(consumer.loaded.model_)
    encoder_before = state_hash(consumer.relation_encoder)
    assert not any(p.requires_grad for p in consumer.loaded.model_.parameters())
    assert torch.count_nonzero(consumer.relation_encoder.output.weight) == 0
    system = InfluencePositionSystem(consumer)
    tick = time.monotonic()
    system.fit(histories, support, event['floor'], aps, tab_bounds, 1150546,
               task=task, state=state, event_id=EVENT, k=1)
    preparation_seconds = time.monotonic()-tick
    del task
    gc.collect()
    torch.cuda.empty_cache()
    write(OUT/'progress.json', dict(stage='H_S_prepared', created=now(),
        seconds=preparation_seconds, APs=len(aps), relation_shape=list(consumer.relation_features.shape)))
    print(json.dumps(dict(stage='H_S_prepared', seconds=preparation_seconds)), flush=True)

    # Coordinates are explicitly excluded by the existing read_scans API.
    qrss = read_scans(CSV, event['query_rows'], False)['rss']
    tick = time.monotonic()
    enabled = system.predict(qrss, enabled=True)
    disabled = system.predict(qrss, enabled=False)
    prediction_seconds = time.monotonic()-tick
    arrays = {f'{mode}_{name}': value for mode, values in (('enabled', enabled), ('disabled', disabled))
              for name, value in values.items()}
    np.savez_compressed(OUT/'unscored.npz', **arrays, query_rows=event['query_rows'], aps=aps)
    with np.load(gp_path) as cached:
        assert np.array_equal(cached['query_rows'], event['query_rows'])
        assert np.array_equal(cached['aps'], aps)
        cached_gp = np.clip(cached['raw'], system.gp_fields['bounds'][0], system.gp_fields['bounds'][1])
    differences = {name:float(np.max(np.abs(enabled[name]-disabled[name]))) for name in enabled}
    differences['GP_cached'] = float(np.max(np.abs(enabled['GP']-cached_gp)))
    weights_unchanged = before == state_hash(consumer.loaded.model_)
    encoder_unchanged = encoder_before == state_hash(consumer.relation_encoder)
    assert weights_unchanged and encoder_unchanged
    finite = all(np.isfinite(value).all() for value in arrays.values())
    completed = dict(complete=True, created=now(), integration_pass=finite and max(differences.values()) <= 1e-4,
        event=EVENT, preparation_seconds=preparation_seconds, prediction_seconds=prediction_seconds,
        total_seconds=time.monotonic()-started, max_abs_coordinate_differences=differences,
        original_weights_unchanged=weights_unchanged, relation_weights_unchanged=encoder_unchanged,
        query_coordinates_read=0, optimizer_steps=0, backward_calls=0, new_GP_fits=0, downloads=0,
        localization_performance=None, relation_shape=list(consumer.relation_features.shape),
        peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
        freeze_sha256=digest(OUT/'freeze.json'), prediction_sha256=digest(OUT/'unscored.npz'),
        limits=['Zero initialized relation route only; trained relation and full backward not verified',
                'One existing source validation task, not new localization or transfer evidence'])
    write(OUT/'complete.json', completed)
    print(json.dumps(completed), flush=True)
    if not completed['integration_pass']:
        raise RuntimeError('Integration difference exceeds declared gate; stop without retry or tolerance change')


if __name__ == '__main__':
    main()
