"""One source B2F0 full-H C backward and optimizer-step budget measurement."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECTS = Path('/home/panyushuo/projects/panyushuo')
CORE = PROJECTS/'ReFi-UJI-Core-20261031'
BENCH = PROJECTS/'paper_bench_wt/current_20261110'
WORK = Path(__file__).resolve().parent
OUTPUT = WORK/'result'
sys.path.insert(0, str(BENCH/'src'))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from law_bench import run
from law_bench.data import load
from law_bench.protocol import make_tasks
sys.path.insert(0, str(CORE))
from direct_coordinate import DirectCoordinateRegressor, PUBLIC_REGRESSOR_SHA256
from tabicl_history import CHECKPOINT


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    (OUTPUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    started = time.monotonic()
    OUTPUT.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1)
    buildings, axes = load('uji')
    tasks = make_tasks('uji', 'B2', buildings['B2'], axes['B2'], 0,
        budgets=(1, 3, 5), repeat=0, seed=20261111, R=3, query_positions=60)
    task, truth = tasks[-1]
    assert task.K == 5 and sorted(task.H) == [1, 2, 3, 4]
    reference = BENCH/'results/paper_uji_B2_20261111/events/refi'/(task.task_id+'.json')
    old = json.loads(reference.read_text())
    assert old['query_row_indices'] == truth['row_indices'].tolist()
    assert old['query_groups'] == truth['groups'].tolist() and old['meta'] == task.meta
    checkpoint_sha = digest(CHECKPOINT)
    assert checkpoint_sha == PUBLIC_REGRESSOR_SHA256
    freeze = dict(status='frozen_before_C_backpropagation', source_only=True,
        task_id=task.task_id, H_rows={f: len(h['rss']) for f, h in task.H.items()},
        S_rows=len(task.S['rss']), Q_rows=len(task.Q_rss), query_positions=task.meta['query_positions'],
        original_AP_columns=len(task.ap_ids), H_floors=sorted(task.H), meta=task.meta,
        source_data=run._data_identity({'B2': buildings['B2']}, {'B2': axes['B2']}),
        query_rows=truth['row_indices'].tolist(), query_groups=truth['groups'].tolist(),
        reference_path=str(reference), reference_sha256=digest(reference),
        checkpoint=dict(path=str(CHECKPOINT), sha256=checkpoint_sha),
        source_seed=1, protocol_seed=20261111, budgets_generated_together=[1, 3, 5],
        views=1, norm_methods=['none'], precision='original fp16 AMP, GradScaler initial1024',
        optimizer=dict(name='AdamW', lr=1e-5, weight_decay=.01, grad_clip=1., steps=1),
        recompute=True, context_policy='Every original H/S scan; original H/S observed AP mask',
        query_permissions='Only this source B2 task; Query coordinates used for the original pinball loss',
        scope='Single engineering update; no checkpoint retained or localization score computed',
        review=dict(
            information='H/S fit context transforms, AP mask, origin and scales before source Query loss',
            gradient='Original EpisodicHistoryRegressor.backward position-balanced pinball and both axes',
            inference='Same official estimator transforms, single view, shared full-parameter regressor',
            counterexamples='Identical Query positions share unit weight; H/S translation centers labels and translates clipping bounds; unseen APs follow the original H/S mask',
            failure='OOM or nonfinite gradient ends this attempt; optimizer capacity and H/AP scope remain frozen'),
        code={str(p):digest(p) for p in [WORK/'direct_coordinate.py', Path(__file__),
            CORE/'tabicl_history.py', CORE/'tabicl_history_learning.py',
            BENCH/'src/law_bench/data.py', BENCH/'src/law_bench/protocol.py']})
    write('freeze.json', freeze)
    torch.cuda.reset_peak_memory_stats()
    model = DirectCoordinateRegressor(seed=1, device='cuda:0', amp=True)
    optimizer = torch.optim.AdamW(model.parameters, lr=1e-5, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    first_parameter = model.parameters[0]
    initial_parameter = first_parameter.detach().clone()
    torch.cuda.synchronize()
    tick = time.monotonic()
    loss = model.source_task_loss(task, truth)
    torch.cuda.synchronize()
    backward_seconds = time.monotonic()-tick
    assert math.isfinite(loss)
    assert all(estimator.model_ is model.model for estimator in model.estimators)
    assert all(estimator.n_samples_in_ == model.diagnostics['context_rows'] for estimator in model.estimators)
    x, y = model.views(task.Q_rss[:1], 0)
    assert len(x) == 1 and y.shape[1] == model.diagnostics['context_rows']
    del x, y
    model.scaler.unscale_(optimizer)
    parameters_with_gradient = sum(p.numel() for p in model.parameters if p.grad is not None)
    nonzero_parameters = sum(int(torch.count_nonzero(p.grad)) for p in model.parameters
        if p.grad is not None)
    assert parameters_with_gradient == sum(p.numel() for p in model.parameters)
    assert nonzero_parameters > 0
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters, 1., error_if_nonfinite=True))
    model.scaler.step(optimizer)
    model.scaler.update()
    torch.cuda.synchronize()
    assert int(optimizer.state[first_parameter]['step']) == 1
    changed = bool(torch.any(first_parameter.detach() != initial_parameter))
    assert changed
    optimizer.zero_grad(set_to_none=True)
    peak_after_step = torch.cuda.max_memory_allocated()/1024**2
    tick = time.monotonic()
    prediction = model.predict_task(task)
    torch.cuda.synchronize()
    predict_seconds = time.monotonic()-tick
    assert prediction.shape == truth['xy'].shape and np.isfinite(prediction).all()
    assert np.all(prediction >= model.bounds[0]) and np.all(prediction <= model.bounds[1])
    result = dict(complete=True, source_only=True, task_id=task.task_id,
        wall_seconds=time.monotonic()-started, device=torch.cuda.get_device_name(0),
        actual_context=model.diagnostics, actual_Q_scans=len(task.Q_rss),
        loss=loss, backward_including_context_fit_seconds=backward_seconds,
        predict_including_context_fit_seconds=predict_seconds,
        initial_grad_scaler=1024., final_grad_scaler=model.scaler.get_scale(),
        trainable_parameters=sum(p.numel() for p in model.parameters),
        parameters_with_gradient=parameters_with_gradient,
        parameters_with_nonzero_gradient=nonzero_parameters,
        finite_gradients=True, gradient_norm=norm, optimizer_steps=1,
        first_parameter_changed=changed, peak_after_step_GPU_MiB=peak_after_step,
        peak_GPU_MiB=torch.cuda.max_memory_allocated()/1024**2,
        predicted_coordinate_shape=list(prediction.shape), localization_score_computed=False,
        freeze_sha256=digest(OUTPUT/'freeze.json'))
    write('complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        try:
            main()
        except Exception as error:
            if OUTPUT.exists():
                write('failed.json', dict(error_type=type(error).__name__, error=repr(error),
                    peak_GPU_MiB=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_initialized() else None))
            raise
