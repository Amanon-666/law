"""Fixed source-only training and building-held-out epoch selection."""
import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch

from native_tab import NativeFieldRole, context_labels
from field_predictor import NativeObservationField
from source_cells import array_digest, prepare_source
from source_probe import ADOPTED, CHECKPOINTS, digest, write


ROLE_IDS = {'det': 0, 'mean': 1, 'mean_other_0': 2, 'mean_other_1': 3, 'scale': 4}
CONTEXT_ROWS = 1024
QUERY_ROWS = 256


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def save_weights(model, path, provenance):
    atomic_save(dict(config=model.model_config,
        state_dict={k:v.detach().cpu() for k,v in model.model.state_dict().items()}, provenance=provenance), path)


def source_data(seed):
    sys.path.insert(0, str(ADOPTED/'claude_hurdle_20261110'))
    import hurdle as H
    data = prepare_source(H, seed=seed)
    count = np.bincount(data['unit_index'], minlength=len(data['unique_units']))
    data['unit_count'] = count
    data['unit_start'] = np.r_[0, np.cumsum(count)[:-1]]
    data['unit_row_order'] = np.argsort(data['unit_index'], kind='stable')
    return data


def sample_context(data, kind, train_mask, query, rng):
    canonical = data['first']
    eligible = train_mask[canonical] & ((data['n'][canonical] > 0) if kind == 'det' else (data['nd'][canonical] > 0))
    eligible &= ~np.isin(np.arange(len(canonical)), np.unique(data['unit_index'][query]))
    units = rng.permutation(np.flatnonzero(eligible))
    if kind == 'det':
        # Native classification contexts must represent both known labels.
        positive = units[data['nd'][canonical[units]] > 0]
        negative = units[data['nd'][canonical[units]] < data['n'][canonical[units]]]
        anchors = np.unique([rng.choice(positive), rng.choice(negative)])
        units = np.r_[anchors, units[~np.isin(units, anchors)]]
    count = data['n'][canonical[units]] if kind == 'det' else data['nd'][canonical[units]]
    units = units[np.cumsum(count) <= CONTEXT_ROWS]
    offset = data['unit_start'][units] + rng.integers(data['unit_count'][units])
    rows = data['unit_row_order'][offset]
    assert len(rows) and not np.intersect1d(data['unit_index'][rows], data['unit_index'][query]).size
    return rows


def train_role(data, kind, role, train_mask, X, target, weight, epochs, seed, directory, freeze_sha):
    directory.mkdir(parents=True, exist_ok=True)
    eligible = train_mask & ((data['n'] > 0) if kind == 'det' else (data['nd'] > 0))
    rows = np.flatnonzero(eligible)
    mean_weight = float(weight[eligible].mean())
    checkpoint = CHECKPOINTS['det' if kind == 'det' else 'reg'][0]
    role_seed = 10*seed + ROLE_IDS[role]
    random.seed(role_seed)
    np.random.seed(role_seed)
    torch.manual_seed(role_seed)
    torch.cuda.manual_seed_all(role_seed)
    model = NativeFieldRole(kind, checkpoint, views=1)
    optimizer = torch.optim.AdamW(model.parameters, lr=1e-5, weight_decay=.01)
    latest = directory/'latest.pt'
    completed_epoch, next_batch = 0, 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['freeze_sha256'] == freeze_sha
        model.model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        completed_epoch, next_batch = saved['completed_epoch'], saved['next_batch']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved

    def persist(epoch, batch):
        atomic_save(dict(config=model.model_config,
            state_dict={k:v.detach().cpu() for k,v in model.model.state_dict().items()},
            optimizer=optimizer.state_dict(), completed_epoch=epoch, next_batch=batch,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            freeze_sha256=freeze_sha), latest)

    history_path = directory/'training.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    predictions = loss = None
    for epoch in range(completed_epoch + 1, epochs + 1):
        permutation = np.random.default_rng([seed, ROLE_IDS[role], epoch]).permutation(rows)
        batches = math.ceil(len(rows)/QUERY_ROWS)
        start_batch = next_batch if epoch == completed_epoch + 1 else 0
        tick = time.monotonic()
        weighted_loss, seen_rows = 0., 0
        for batch in range(start_batch, batches):
            query = permutation[batch*QUERY_ROWS:(batch+1)*QUERY_ROWS]
            rng = np.random.default_rng([seed, ROLE_IDS[role], epoch, batch, 871])
            context = sample_context(data, kind, train_mask, query, rng)
            context_X, context_y = context_labels(data, kind, context, X, target)
            model.fit_context(context_X, context_y, seed=seed)
            transformed = model.transformed_views(X[query])
            assert len(transformed[0]) == 1
            optimizer.zero_grad(set_to_none=True)
            predictions = model.forward_views(transformed)
            loss = model.loss(predictions, target[query], weight[query], mean_weight)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters, 1., error_if_nonfinite=True)
            assert norm > 0
            optimizer.step()
            weighted_loss += float(loss.detach())*len(query)
            seen_rows += len(query)
            if (batch+1) % 256 == 0:
                persist(epoch-1, batch+1)
                print(json.dumps(dict(stage='training_progress', role=role, epoch=epoch,
                    completed_batches=batch+1, batches=batches)), flush=True)
        torch.cuda.synchronize()
        path = directory/f'epoch_{epoch}.ckpt'
        save_weights(model, path, dict(role=role, seed=seed, source_epoch=epoch,
                     freeze_sha256=freeze_sha, source_only=True))
        persist(epoch, 0)
        record = dict(role=role, epoch=epoch, source_rows=len(rows), newly_processed_rows=seen_rows,
            mean_loss=weighted_loss/seen_rows if seen_rows else None, seconds=time.monotonic()-tick,
            checkpoint_sha256=digest(path), resumed_at_batch=start_batch)
        history = [h for h in history if h['epoch'] != epoch] + [record]
        write(history_path, history)
        print(json.dumps(dict(stage='epoch_complete', **record)), flush=True)
        next_batch = 0
    del predictions, loss, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


def load_predictor(data, kind, checkpoint, train_mask, X, target, seed):
    model = NativeFieldRole(kind, checkpoint, views=1)
    context = sample_context(data, kind, train_mask, np.array([], dtype=int),
                             np.random.default_rng([seed, 921, 0 if kind=='det' else 1]))
    context_X, context_y = context_labels(data, kind, context, X, target)
    model.fit_context(context_X, context_y, seed=seed)
    model.model.eval()
    model.estimator.kv_cache = True
    model.estimator.cache_mode_ = 'kv'
    model.estimator._build_kv_cache()
    return model, context


def predict(model, X, chunk=512):
    output = []
    for start in range(0, len(X), chunk):
        values = (model.estimator.predict_proba(X[start:start+chunk])[:, 1] if model.role == 'det'
                  else model.estimator.predict(X[start:start+chunk]))
        assert np.isfinite(values).all()
        output.append(values)
    return np.concatenate(output)


def own_oof(data, train_mask, checkpoints, seed, destination):
    if destination.exists():
        with np.load(destination) as saved:
            assert saved['checkpoint_sha256'].tolist() == [digest(p) for p in checkpoints]
            return saved['oof'], saved['target']
    oof = np.full(len(data['X']), np.nan)
    for half in (0, 1):
        other = train_mask & (data['half'] != half)
        model, context = load_predictor(data, 'mean', checkpoints[half], other, data['X'], data['mu'], seed)
        own = train_mask & (data['half'] == half) & (data['nd'] > 0)
        oof[own] = predict(model, data['X'][own])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    eligible = train_mask & (data['nd'] > 0)
    target = np.full(len(oof), np.nan)
    target[eligible] = np.log(np.nan_to_num(data['var'][eligible], nan=0.) +
                             (data['mu'][eligible]-oof[eligible])**2 + 9.)
    assert np.isfinite(target[eligible]).all()
    np.savez_compressed(destination, oof=oof, target=target,
                        checkpoint_sha256=np.array([digest(p) for p in checkpoints]))
    return oof, target


def field_engine(data, train_mask, checkpoints, oof, scale_target, seed):
    models = {}
    for kind in ('det', 'mean', 'scale'):
        source_X = data['X'] if kind != 'scale' else np.column_stack((data['X'], oof.astype(np.float32)))
        source_y = data['nd'] if kind == 'det' else (data['mu'] if kind == 'mean' else scale_target)
        model, context = load_predictor(data, kind, checkpoints[kind], train_mask, source_X, source_y, seed)
        models[kind] = model
    return NativeObservationField(models)


def development_localization(engine):
    from law_bench.data import load
    from law_bench.protocol import make_tasks
    from law_bench.score import score
    buildings, axes = load('uji')
    errors = []
    for floor in sorted(buildings['B2']):
        tasks = make_tasks('uji','B2',buildings['B2'],axes['B2'],floor,
                           budgets=(1,3,5),repeat=0,seed=20261111,R=3,query_positions=60)
        for task, truth in tasks:
            prediction = engine.prepare(task).predict(task.Q_rss)['projected']
            errors.extend(score(prediction,truth)['position_errors'])
    return dict(tasks=15,position_budget_observations=len(errors),
                mean=float(np.mean(errors)),p90=float(np.quantile(errors,.9)))


def validation_nll(data, rows, prediction):
    n, nd = data['n'][rows], data['nd'][rows]
    # The probability bounds are the unchanged FrozenHurdle readout bounds.
    p = np.clip(prediction['det'].astype(np.float64), 1e-4, 1-1e-4)
    loss = -(nd*np.log(p)+(n-nd)*np.log1p(-p))
    detected = nd > 0
    log_var = prediction['scale'][detected].astype(np.float64)
    residual = (data['mu'][rows][detected]-prediction['mean'][detected])**2
    residual += np.nan_to_num(data['var'][rows][detected], nan=0.)
    loss[detected] += .5*nd[detected]*(np.log(2*np.pi)+log_var+residual*np.exp(-log_var))
    assert np.isfinite(loss).all()
    return float(loss.sum()/n.sum())


def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    for path, expected in CHECKPOINTS.values():
        assert digest(path) == expected
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output/'freeze.json'
    freeze = dict(phase=args.phase, seed=args.seed, maximum_source_epochs=args.epochs,
        training_buildings=[0] if args.phase=='development' else [0, 2],
        development_building=2 if args.phase=='development' else None,
        source_dev_query_permissions='B2 has no training-label or training-context role in development. Final models restart from public weights and use all permitted B0+B2.',
        source_selection='Minimum B2 source-only field localization pooled MDE among epochs 0/1/2/3; exact tie takes earlier. Fixed B2 floors, K1/3/5, r0, static-resample-v2 base seed20261111. Observation NLL is auxiliary.',
        context_count_rows=CONTEXT_ROWS, query_rows=QUERY_ROWS, views=1, normalization=['none'],
        precision='float32', optimizer=dict(name='AdamW', learning_rate=1e-5, weight_decay=.01, clip=1.),
        context_sampling='Whole source label units, with one random original K/draw alias per unit and exact n/nd count expansion. Exclude all aliases of every query unit.',
        detection_context='Include real source observations of both known classes, as required by the native classifier label encoder.',
        source_data='Original HGB 55-feature table and scan sufficient statistics; scale appends the opposite-floor-half Tab mean as feature 56.',
        scale_training='Independent native Tab means trained only on each opposite original floor half produce OOF labels. Scale target is log(var+(mu-OOF_mean)^2+9), with nd weights.',
        losses=dict(det='Count-weighted native two-class CE after inverse class shuffle, temperature 1.',
                    mean='nd-weighted RSS-unit MSE of the inverse-scaled native quantile mean.',
                    scale='nd-weighted log-variance-unit MSE of the inverse-scaled native quantile mean.'),
        code_sha256={name:digest(Path(__file__).parent/name) for name in ('formal_source.py','field_predictor.py','native_tab.py','source_cells.py','source_probe.py')},
        reader_code_sha256={name:digest(ADOPTED/name) for name in ('frozen_hurdle_translator.py','vertical_translator/translator.py')},
        raw_source_CSV_sha256=digest(Path(os.environ['LAW_UJI_CSV'])),
        public_checkpoints={key:dict(path=str(path),sha256=expected) for key,(path,expected) in CHECKPOINTS.items()})
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == freeze
    else:
        write(freeze_path, freeze)
    freeze_sha = digest(freeze_path)
    data = source_data(args.seed)
    train_mask = np.isin(data['units'][:, 0], freeze['training_buildings'])
    data['train'] = train_mask
    write(args.output/'source_manifest.json', dict(original=data['manifest'],
        formal_training_rows=int(train_mask.sum()),
        formal_training_units=len(np.unique(data['unit_index'][train_mask])),
        training_mask_sha256=array_digest(train_mask), pilot_random_unit_split_used=False,
        source_rows_by_building={str(b):dict(rows=int((data['units'][:,0]==b).sum()), detected_rows=int(((data['units'][:,0]==b)&(data['nd']>0)).sum())) for b in (0,2)}))
    role_paths = {}
    for role in ('det', 'mean', 'mean_other_0', 'mean_other_1'):
        kind = 'det' if role == 'det' else 'mean'
        mask = train_mask if not role.startswith('mean_other_') else train_mask & (data['half'] != int(role[-1]))
        target = data['nd'] if kind == 'det' else data['mu']
        weight = data['n'] if kind == 'det' else data['nd']
        directory = args.output/role
        train_role(data, kind, role, mask, data['X'], target, weight, args.epochs, args.seed, directory, freeze_sha)
        role_paths[role] = directory
    candidates = range(args.epochs+1) if args.phase=='development' else [args.epochs]
    curves = []
    validation_rows = np.flatnonzero(data['units'][:, 0] == 2)
    if args.phase == 'development':
        validation_rows = np.random.default_rng(1541).choice(validation_rows, size=min(10000,len(validation_rows)), replace=False)
        write(args.output/'development_queries.json', dict(building=2, rows=validation_rows.tolist(),
            unique_label_units=len(np.unique(data['unit_index'][validation_rows])),
            labels_never_used_in_training=True, auxiliary_metric='Scan-count-weighted Bernoulli plus conditional-Gaussian observation NLL on fixed B2 table rows.'))
    for epoch in candidates:
        public_mean = CHECKPOINTS['reg'][0]
        oof_paths = [role_paths[f'mean_other_{half}']/f'epoch_{epoch}.ckpt' if epoch else public_mean for half in (0,1)]
        oof, target = own_oof(data, train_mask, oof_paths, args.seed, args.output/f'oof_epoch_{epoch}.npz')
        scale_X = np.column_stack((data['X'], oof.astype(np.float32)))
        scale_directory = args.output/f'scale_for_epoch_{epoch}'
        if epoch:
            train_role(data, 'scale', 'scale', train_mask, scale_X, target, data['nd'], epoch, args.seed, scale_directory, freeze_sha)
        checkpoints = {kind:role_paths[kind]/f'epoch_{epoch}.ckpt' if epoch else CHECKPOINTS['det' if kind=='det' else 'reg'][0] for kind in ('det','mean')}
        checkpoints['scale'] = scale_directory/f'epoch_{epoch}.ckpt' if epoch else public_mean
        if args.phase == 'development':
            engine = field_engine(data, train_mask, checkpoints, oof, target, args.seed)
            prediction = engine.parameters_at(data['X'][validation_rows])
            localization = development_localization(engine)
            curves.append(dict(epoch=epoch, source_development_nll=validation_nll(data,validation_rows,prediction),
                               source_development_localization=localization))
            write(args.output/'learning_curve.json', curves)
            engine.close()
            del engine
        else:
            contexts = {}
            for kind in ('det','mean','scale'):
                X = data['X'] if kind!='scale' else scale_X
                y = data['nd'] if kind=='det' else (data['mu'] if kind=='mean' else target)
                model, indices = load_predictor(data,kind,checkpoints[kind],train_mask,X,y,args.seed)
                context_X, context_y = context_labels(data,kind,indices,X,y)
                path = args.output/f'{kind}_context.npz'
                np.savez_compressed(path,X=context_X,y=context_y,source_row_indices=indices,source_units=data['units'][indices])
                contexts[kind] = dict(path=str(path),sha256=digest(path),count_rows=len(context_y),units=len(indices),checkpoint=str(checkpoints[kind]),checkpoint_sha256=digest(checkpoints[kind]))
                del model
                gc.collect()
                torch.cuda.empty_cache()
            write(args.output/'model_assets.json', dict(seed=args.seed,source_epoch=epoch,contexts=contexts,source_only=True))
    selected = min(curves,key=lambda r:(r['source_development_localization']['mean'],r['epoch']))['epoch'] if curves else args.epochs
    write(args.output/'complete.json', dict(complete=True,phase=args.phase,seed=args.seed,
        selected_source_epoch=selected,freeze_sha256=freeze_sha,source_only=True,target_evaluation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development','final'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
