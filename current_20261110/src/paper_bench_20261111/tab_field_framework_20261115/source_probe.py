"""Source-only interface and throughput probe. No target task is evaluated."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import tabicl
from tabicl.__about__ import __version__ as tabicl_version
from threadpoolctl import threadpool_limits

from native_tab import NativeFieldRole, context_labels
from source_cells import context_and_query, matched_hgb_and_scale, prepare_source


PROJECTS = Path('/home/panyushuo/projects/panyushuo')
ADOPTED = PROJECTS / 'ReFi-UJI-Core-20261031'
WEIGHTS = PROJECTS / 'RNP-UJI/tabicl_weights'
CHECKPOINTS = {
    'det': (WEIGHTS / 'tabicl-classifier-v2-20260212.ckpt', 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'),
    'reg': (WEIGHTS / 'tabicl-regressor-v2-20260212.ckpt', '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1)
    checkpoints = {}
    for kind, (path, expected) in CHECKPOINTS.items():
        assert digest(path) == expected
        checkpoints[kind] = dict(path=str(path), sha256=expected, bytes=path.stat().st_size)
    csv_path = Path(os.environ['LAW_UJI_CSV'])
    package = Path(tabicl.__file__).parent
    freeze = dict(status='frozen_before_source_probe', seed=1, source_buildings=[0, 2],
        target_building=1, target_evaluation=False, context_row_budgets=args.context_rows,
        query_rows=args.query_rows, steps_per_case=args.steps, views=1,
        normalization=['none'], classifier_temperature=1., precision='float32',
        pilot_split='Random 10% whole label units are reserved only for this interface probe. Formal source development uses predeclared whole floors/buildings before retraining on permitted B0+B2.',
        checkpoints=checkpoints, parameter_training='All native learnable weights; preserve architecture-fixed RoPE frequencies.',
        source_CSV=dict(path=str(csv_path), sha256=digest(csv_path)), tabicl_version=tabicl_version,
        package_code_sha256={str(p.relative_to(package)):digest(p) for p in package.rglob('*.py')},
        classifier_source_revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
        classifier_source_url='https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt',
        code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        adopted_code={str(p.relative_to(ADOPTED)):digest(p) for p in [ADOPTED/'claude_hurdle_20261110/hurdle.py', ADOPTED/'claude_hurdle_20261110/translator.py', ADOPTED/'claude_hurdle_20261110/vt_lib.py']},
        labels='Original n/nd/mu/variance sufficient statistics, with whole-unit source-context/query separation.',
        scale_probe_label_generator='HGB original two-half floor crossfit on matched source training units. These labels validate numerical/compute interfaces only. Formal Tab scale training needs Tab-generated OOF means.',
        optimizer=dict(name='AdamW', lr=1e-5, weight_decay=.01, grad_clip=1.),
        resources=dict(gpu=1, cpu_threads=2, maximum_wall_seconds=3600),
        scope='Source-budget verification only. One-view/single-model field probe differs from the adopted five-HGB-seed system and old eight-view Tab assets.')
    write(args.output/'freeze.json', freeze)
    sys.path.insert(0, str(ADOPTED/'claude_hurdle_20261110'))
    import hurdle as H
    with threadpool_limits(limits=2):
        tick = time.monotonic()
        data = prepare_source(H, seed=1)
        preparation_seconds = time.monotonic() - tick
        write(args.output/'source_manifest.json', data['manifest'])
        np.savez_compressed(args.output/'source_units.npz', unique_units=data['unique_units'],
                            row_unit_index=data['unit_index'], row_training=data['train'], row_half=data['half'])
        print(json.dumps(dict(stage='source_arrays_verified', seconds=preparation_seconds, **data['manifest'])), flush=True)
        tick = time.monotonic()
        matched, scale_X, scale_target = matched_hgb_and_scale(H, data, seed=1)
        hgb_seconds = time.monotonic() - tick
        write(args.output/'source_HGB_probe.json', dict(seconds=hgb_seconds, matched_training_rows=int(data['train'].sum()),
            source_buildings=[0, 2], validation_units_excluded=True, original_hyperparameters=True,
            scale_labels='HGB OOF for this numerical/throughput probe only'))
        del matched
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = dict(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0))
    records = []
    for role in ('det', 'mean', 'scale'):
        checkpoint = CHECKPOINTS['det' if role == 'det' else 'reg'][0]
        model = NativeFieldRole(role, checkpoint, device='cuda:0', views=1)
        optimizer = torch.optim.AdamW(model.parameters, lr=1e-5, weight_decay=.01)
        named = dict(model.model.named_parameters())
        fixed = [name for name, value in named.items() if not value.requires_grad]
        assert fixed == ['row_interactor.tf_row.rope.freqs']
        parameter_metadata = dict(total=sum(p.numel() for p in named.values()), trainable=sum(p.numel() for p in model.parameters), fixed=fixed, dropout=model.model_config['dropout'])
        for case, budget in enumerate(args.context_rows):
            context, query = context_and_query(data, role, budget, args.query_rows, seed=1+case)
            context_X, context_y = context_labels(data, role, context, scale_X, scale_target)
            features = data['X'] if role != 'scale' else scale_X
            targets = data['nd'] if role == 'det' else (data['mu'] if role == 'mean' else scale_target)
            weights = data['n'] if role == 'det' else data['nd']
            eligible = data['train'] & ((data['n'] > 0) if role == 'det' else (data['nd'] > 0))
            training_mean_weight = float(weights[eligible].mean())
            tick = time.monotonic()
            model.fit_context(context_X, context_y, seed=1)
            transformed = model.transformed_views(features[query])
            preparation = time.monotonic() - tick
            assert len(transformed[0]) == 1
            record = dict(role=role, context_row_budget=budget, actual_context_rows=len(context_y),
                context_units=len(context), query_rows=len(query), query_units=len(np.unique(data['unit_index'][query])),
                context_query_unit_overlap=0, raw_features=features.shape[1], transformed_features=transformed[0].shape[-1],
                source_training_examples=int((data['train'] & ((data['n']>0) if role=='det' else (data['nd']>0))).sum()),
                loss_count_normalizer=training_mean_weight,
                context_preprocessing_seconds=preparation, parameters=parameter_metadata,
                class_shuffles=[list(map(int, s)) for s in transformed[2]], steps=[])
            for step in range(args.steps):
                if time.monotonic() - started > 3500:
                    raise RuntimeError('Source probe exhausted its fixed wall budget')
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.monotonic()
                predictions = model.forward_views(transformed)
                loss = model.loss(predictions, targets[query], weights[query], training_mean_weight)
                assert torch.isfinite(loss)
                loss.backward()
                gradients = [p.grad for p in model.parameters if p.grad is not None]
                assert all(torch.isfinite(g).all() for g in gradients)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters, 1., error_if_nonfinite=True))
                assert norm > 0
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                record['steps'].append(dict(step=step+1, loss=float(loss.detach()), gradient_norm=norm,
                    parameters_with_gradient=sum(p.numel() for p in model.parameters if p.grad is not None),
                    parameters_with_nonzero_gradient=sum(p.numel() for p in model.parameters if p.grad is not None and bool((p.grad!=0).any())),
                    seconds=elapsed, peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20))
            optimizer.zero_grad(set_to_none=True)
            model.model.eval()
            torch.cuda.synchronize()
            tick = time.monotonic()
            if role == 'det':
                prediction = model.estimator.predict_proba(features[query])[:, 1]
            else:
                prediction = model.estimator.predict(features[query])
            torch.cuda.synchronize()
            record['native_predict_seconds'] = time.monotonic() - tick
            assert np.isfinite(prediction).all()
            if role == 'det':
                assert ((prediction>=0)&(prediction<=1)).all()
            with torch.no_grad():
                raw = model.forward_views(transformed)[0]
                comparable = raw.softmax(-1)[:, 1] if role == 'det' else raw
            difference = float(np.max(np.abs(prediction-comparable.cpu().numpy())))
            record['train_path_vs_native_prediction_max_abs_difference'] = difference
            assert difference < (1e-4 if role == 'det' else 1e-3)
            times = [s['seconds'] for s in record['steps']]
            record['one_full_training_example_pass_extrapolated_seconds'] = record['source_training_examples']/len(query)*(float(np.median(times))+preparation)
            full_examples = len(data['X']) if role == 'det' else int((data['nd'] > 0).sum())
            record['full_source_table_examples'] = full_examples
            record['full_source_table_pass_extrapolated_seconds'] = full_examples/len(query)*(float(np.median(times))+preparation)
            records.append(record)
            write(args.output/'throughput.json', records)
            print(json.dumps(dict(stage='native_role_case_verified', **record)), flush=True)
            del predictions, loss, gradients, raw, comparable
        del optimizer, model, named
        torch.cuda.empty_cache()
    write(args.output/'complete.json', dict(complete=True, device=device, wall_seconds=time.monotonic()-started,
        source_preparation_seconds=preparation_seconds, matched_HGB_seconds=hgb_seconds,
        source_manifest_sha256=digest(args.output/'source_manifest.json'),
        source_units_sha256=digest(args.output/'source_units.npz'), freeze_sha256=digest(args.output/'freeze.json'),
        throughput_sha256=digest(args.output/'throughput.json'), source_only=True, target_evaluation=False,
        scope='Numerical interfaces and throughput, with HGB-OOF scale labels. No final Tab OOF or target performance result.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-rows', type=int, nargs='+', default=[256, 1024, 2048])
    parser.add_argument('--query-rows', type=int, default=256)
    parser.add_argument('--steps', type=int, default=2)
    run(parser.parse_args())
