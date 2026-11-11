"""Candidate asset inheritance entry: detection_ipw20 or scale_gamma.

Run from CORE with python -m candidates.scale_gamma_20261111.claude_hurdle_detection_ipw
and --recipe scale_gamma. This candidate copy leaves all frozen core code intact.
scale_gamma accepts only the adopted original B0 five-model generation. It
inherits det/mean and changes only the scale loss/target: Gamma on positive
Z=within_var+(mean-crossfit_mean)^2+9, with original count weights and splits.
Its prediction is variance directly (scale_output='variance'); legacy assets
remain log_variance. Cross-fit source mean predictions are required to rebuild
Z; no held-out target prediction or localization scoring occurs here.
This is consistency for the existing proxy second moment, not a resolution
of Support overlap, nearest-position labels or cross-acquisition mismatch.

Implementation only at handoff. Training is restricted to ray-web, CPU2,
no GPU/downloads, with the original 840-second internal budget. Inputs are
an already completed original five-model generation and its existing CSV.

Reconstruct the selected table through unchanged H.cell_table, using original
ordered source pairs, even/odd groups, default_rng([seed,i]), draws4 and its
unchanged K choices/6000-row cap. On raw 55-column X, the original information
predicate uses H columns 0:40:5 and S2[:,1], which is raw column 44 because
history_features has 43 columns and support_features has 12 columns.
Selected cells with neither H nor S hint and nd==0 receive cell weight20.
Multiply BOTH nd and n-nd scan counts by the cell weight, then keep positive
count rows exactly as the original classifier does. Other cells receive1.

Clone each original det estimator to retain all HGB constructor parameters,
random state and early-stopping configuration. Its saved feature columns,
the global columns and the fitted mean/scale states are inherited directly.
The IPW output kind remains claude_hurdle_v1. Gamma uses a distinct kind so an
old reader cannot silently interpret its variance prediction as log variance.
freeze/complete/payload explicitly mark the changed detection_ipw20 recipe.
This addresses one label-dependent selection probability only. The retained
6000-row cap and task mixture prevent a claim of full-population unbiased
calibration. The IPW branch performs no source/target predictions or scoring;
the Gamma branch performs only the original source cross-fit mean predictions.

The SHA-pinned original B0 generation predates the current metadata schema.
Its floor list and complete-file floor counts are normalized explicitly.
Reader constants for that generation follow frozen_hurdle_translator.py and
run_nine_floor_hurdle.py's B0-development recipe. A02 uses its current schema.
"""
import claude_hurdle_assets as A  # Establish the original CPU2/offline settings first.

import argparse
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from .frozen_hurdle_translator import FrozenHurdleTranslator

RECIPE = 'detection_ipw20'
HISTORY_WIDTH = 43
FEATURE_WIDTH = 55
SUPPORT_HINT = HISTORY_WIDTH + 1
LEGACY_B0_FREEZE_SHA = '371f5815a44f7c4ddac67cbb1ad7932ba90e3189958d5723f4984f86cf1d08a7'


def normalized_source_metadata(freeze, complete, freeze_sha):
    if freeze_sha == LEGACY_B0_FREEZE_SHA:
        floors = {'0': freeze['source_floors']}
        rows = {f'B0F{floor}': count for floor, count in complete['source_rows_by_floor'].items()}
        reader = dict(beta=128., eps=.02, offsets=[-8., -4., 0., 4., 8.],
            selection_source='UJI B0 development; not independent of B0',
            evaluation_status='exposed_development',
            provenance='frozen_hurdle_translator.py EPS/BETA/OFFSETS and original run_nine_floor_hurdle.py B0-development constants')
    else:
        floors = freeze['source_floors']
        rows = freeze['source_rows_by_floor']
        reader = freeze['reader']
    return floors, rows, reader


def refit_scale_gamma(H, model, parts, budget):
    """Same hurdle.py:107-114 construction; only log-Z MSE -> positive-Z Gamma."""
    Xs, zs, ws = [], [], []
    for i in (0, 1):
        budget()
        Xi, ni, ndi, mi, vi = parts[i]
        Xo, no, ndo, mo, vo = parts[1-i]
        do, di = ndo > 0, ndi > 0
        other = H._Fit(H._hgb_reg(model['seed']), Xo[do][:, model['cols']], mo[do], ndo[do])
        mh = other.predict(Xi[di][:, model['cols']])
        Xs.append(np.hstack([Xi[di][:, model['cols']], mh[:, None].astype(np.float32)]))
        zs.append(np.nan_to_num(vi[di], nan=0.0) + (mi[di]-mh)**2 + H.S0**2)
        ws.append(ndi[di])
    X, Z, weights = np.vstack(Xs), np.concatenate(zs), np.concatenate(ws)
    if not np.array_equal(np.flatnonzero(~np.isnan(X).all(0)), model['scale']['cols']):
        raise ValueError('Reconstructed original scale feature columns differ')
    estimator = clone(model['scale']['estimator']).set_params(loss='gamma')
    budget()
    estimator.fit(X[:, model['scale']['cols']], Z, sample_weight=weights)
    return estimator, dict(selected_cells=sum(len(p[0]) for p in parts),
        scale_cells=len(Z), detected_scan_weight=float(weights.sum()),
        Z_min=float(Z.min()), Z_max=float(Z.max()), source_crossfit_mean_fits=2,
        source_crossfit_predictions=len(Z), inherited_state_hashes={
            key: joblib.hash(model[key]) for key in ('det', 'mean')})


def run(args):
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Training requires ray-web')
    if not 0 < args.seconds <= 840:
        raise ValueError('Internal budget must be in (0,840] seconds')
    started = time.monotonic()
    generation = args.generation_dir.resolve()
    output = args.output.resolve()
    csv_path = args.csv.resolve()
    if output == generation or generation in output.parents:
        raise ValueError('Output must be separate from the original generation directory')
    # The deployment reader checks complete/freeze/model hashes and schema.
    original = FrozenHurdleTranslator(generation)
    base_freeze = original.freeze
    base_complete = json.loads((generation/'complete.json').read_text())
    if base_complete['original_recipe_changed']:
        raise ValueError('Input generation must use the completed original A recipe')
    sources = base_freeze['source_buildings']
    base_freeze_sha = A.digest(generation/'freeze.json')
    gamma = args.recipe == 'scale_gamma'
    if gamma and base_freeze_sha != LEGACY_B0_FREEZE_SHA:
        raise ValueError('scale_gamma requires the adopted original B0 five models, not IPW20 or A02')
    source_floors, expected_rows, reader = normalized_source_metadata(
        base_freeze, base_complete, base_freeze_sha)
    source_permission = (
        f'Original full corpus of ordered source buildings {sources}; only these sources enter fitting. '
        'Original task sampling, selected cells and scan-count structure retained. '
        f'The {args.recipe} recipe changes only the recorded head objective.')
    csv_sha = A.digest(csv_path)
    if csv_sha not in base_freeze['inputs'].values():
        raise ValueError('CSV differs from the original generation input')
    versions = {name: A.version(name) for name in base_freeze['versions']}
    if versions != base_freeze['versions']:
        raise ValueError('Use the original generation dependency versions')
    for name, expected in base_freeze['upstream_git_blobs'].items():
        data = (A.PACKAGE/name).read_bytes()
        blob = A.hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob != expected:
            raise ValueError(f'Original module differs: {name}')
    source_rows = {}
    with csv_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            building, floor = int(row['BUILDINGID']), int(row['FLOOR'])
            if building in sources:
                key = f'B{building}F{floor}'
                source_rows[key] = source_rows.get(key, 0) + 1
    if source_rows != expected_rows:
        raise ValueError('Source floor row counts differ from the base generation')
    os.environ['LAW_UJI_CSV'] = str(csv_path)
    sys.path.insert(0, str(A.PACKAGE))
    import hurdle as H

    pairs = [(building, floor) for building in sources for floor in H.V.FLOORS[building]]
    if {str(b): H.V.FLOORS[b] for b in sources} != source_floors:
        raise ValueError('Original source floor order differs')
    output.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), Path(A.__file__),
            Path(sys.modules[FrozenHurdleTranslator.__module__].__file__),
            *[A.PACKAGE/name for name in base_freeze['upstream_git_blobs']]]
    recipe = dict(recipe=RECIPE, original_recipe_changed=True, refit_heads=['det'],
        inherited_heads=['mean', 'scale'], target_predictions=0, localization_scores=0,
        weight_rule='20 iff raw H hints absent, raw S hint absent and nd==0; otherwise1; multiply both nd and n-nd',
        feature_rule=dict(width=FEATURE_WIDTH, history_width=HISTORY_WIDTH,
                          history_hint_slice=[0,40,5], support_hint_column=SUPPORT_HINT),
        scope='Only the 5% label-dependent retention correction; row cap/task-mixture correction is outside this recipe')
    if gamma:
        recipe = dict(recipe='scale_gamma', kind='claude_hurdle_scale_gamma_v1',
            original_recipe_changed=True, refit_heads=['scale'],
            inherited_heads=['det', 'mean'], scale_output='variance',
            target_predictions=0, localization_scores=0,
            target='Same Z=within_var+(mean-m_crossfit)^2+9; original positive nd weights',
            objective='HGB half Gamma deviance on Z, log-link; Z/v+log(v)+target-only constant',
            source_crossfit='Original even/odd source pairs; refit original mean estimator on opposite half',
            scope='Correct second moment of the same proxy target; S overlap, near-position labels and acquisition mismatch remain')
    A.write(output/'freeze.json', dict(
        created=datetime.now(timezone.utc).isoformat(), **recipe,
        upstream_commit=base_freeze['upstream_commit'], upstream_git_blobs=base_freeze['upstream_git_blobs'],
        seeds=A.SEEDS, draws=4, source_buildings=sources,
        source_floors=source_floors, source_rows_by_floor=source_rows,
        source_permission=source_permission,
        source_pairs=pairs, pair_groups=[pairs[0::2], pairs[1::2]],
        table_recipe='Unmodified H.cell_table per parity group, default_rng([seed,i]), draws4, default Ks/max_rows',
        reader=reader, versions=versions,
        base_generation=dict(path=str(generation), freeze_sha256=base_freeze_sha,
            complete_sha256=A.digest(generation/'complete.json'), models=base_complete['models']),
        code={str(path.resolve()): A.digest(path) for path in code}, inputs={str(csv_path): csv_sha},
        fit=('Clone original scale estimator, change only loss to gamma; original scale/global columns, original Z and nd weights'
             if gamma else 'Clone original det estimator and refit with count weights times cell weight; reuse original global/det columns'),
        resources=dict(CPU_threads=2, GPU=0, seconds=args.seconds, external_timeout=900, downloads=0),
        access='Original eager CSV/floor-stat loading; only frozen ordered source pairs enter head/cross-fit fitting',
        implementation_status='Implemented; this asset run must finish five models before complete=true'))
    freeze_sha = A.digest(output/'freeze.json')
    models = []
    current_seed = None

    def budget():
        if time.monotonic()-started >= args.seconds:
            raise TimeoutError('840-second-bounded asset budget exhausted')

    try:
        with threadpool_limits(limits=2):
            for model in original.models:
                current_seed = model['seed']
                budget()
                tick = time.monotonic()
                A.write(output/'progress.json', dict(complete=False, phase='rebuild_selected_table',
                    active_seed=current_seed, models=models, freeze_sha256=freeze_sha, **recipe))
                parts = [H.cell_table(group, np.random.default_rng([current_seed, i]), draws=4)
                         for i, group in enumerate((pairs[0::2], pairs[1::2]))]
                if any(part[0].shape[1] != FEATURE_WIDTH for part in parts):
                    raise ValueError('Expected original 43 H + 12 S feature layout')
                if gamma:
                    A.write(output/'progress.json', dict(complete=False, phase='refit_scale_gamma',
                        active_seed=current_seed, models=models, freeze_sha256=freeze_sha, **recipe))
                    estimator, stats = refit_scale_gamma(H, model, parts, budget)
                    head = 'scale'
                else:
                    X = np.vstack([part[0] for part in parts])
                    n = np.concatenate([part[1] for part in parts])
                    nd = np.concatenate([part[2] for part in parts])
                    h_hint = np.nan_to_num(X[:,0:40:5], nan=H.T.ND).max(1) > H.T.ND
                    s_hint = np.nan_to_num(X[:,SUPPORT_HINT], nan=H.T.ND) > H.T.ND
                    upweighted = (~h_hint) & (~s_hint) & (nd == 0)
                    cell_weights = np.where(upweighted, 20., 1.)
                    Xd = np.vstack([X, X])[:,model['cols']]
                    labels = np.r_[np.ones(len(X)), np.zeros(len(X))]
                    weights = np.r_[nd*cell_weights, (n-nd)*cell_weights]
                    keep = weights > 0
                    stats = dict(selected_cells=len(X), upweighted_cells=int(upweighted.sum()),
                        unit_weight_cells=int((~upweighted).sum()),
                        original_detected_scan_weight=float(nd.sum()), original_missing_scan_weight=float((n-nd).sum()),
                        weighted_detected_scan_weight=float(weights[:len(X)].sum()),
                        weighted_missing_scan_weight=float(weights[len(X):].sum()))
                    budget()
                    A.write(output/'progress.json', dict(complete=False, phase='fit_det',active_seed=current_seed,
                        active_seed_counts=stats, models=models, freeze_sha256=freeze_sha, **recipe))
                    estimator = clone(model['det']['estimator'])
                    estimator.fit(Xd[keep][:,model['det']['cols']], labels[keep], sample_weight=weights[keep])
                    del X, Xd, n, nd, weights
                    head = 'det'
                budget()
                payload = dict(model, freeze_sha256=freeze_sha, **recipe)
                payload.update(source_floors=source_floors, source_rows_by_floor=source_rows,
                               source_permission=source_permission)
                payload[head] = dict(model[head], estimator=estimator)
                payload[args.recipe+'_counts'] = stats
                path = output/f'seed_{current_seed}.joblib'
                temporary = output/f'.seed_{current_seed}.tmp.joblib'
                joblib.dump(payload, temporary)
                temporary.replace(path)
                record = dict(seed=current_seed, path=path.name, sha256=A.digest(path),
                    bytes=path.stat().st_size, seconds=time.monotonic()-tick, **stats)
                models.append(record)
                A.write(output/'progress.json', dict(complete=False, phase='seed_saved', models=models,
                    freeze_sha256=freeze_sha, **recipe))
                print(json.dumps(record), flush=True)
                del parts, payload, estimator
            budget()
        A.write(output/'complete.json', dict(complete=True, created=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic()-started, freeze_sha256=freeze_sha, models=models,
            source_buildings=sources, source_floors=source_floors,
            source_rows_by_floor=source_rows, source_permission=source_permission, **recipe))
    except Exception as error:
        A.write(output/'progress.json', dict(complete=False, phase='failed', active_seed=current_seed,
            error_type=type(error).__name__, error=str(error), models=models, freeze_sha256=freeze_sha, **recipe))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-dir', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=840.)
    parser.add_argument('--recipe', choices=('detection_ipw20', 'scale_gamma'), required=True)
    run(parser.parse_args())
