"""One-time conversion of the existing B1-parameter, full-B0 history fields.

Run on the server: python -m law_bench.import_b1_b0_fields --cache-dir PATH
No GP/model inference, fitting or original-file modification. Existing output
history directories are rejected. Returns gp_options for TrainedFieldGP;
its current fold must explicitly permit uji/B1 as a GP training source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import load
from .method_retained_gp import history_key


RESULTS = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/'
               'Law-Probabilistic-Transfer-20261023/results')
ASSETS = RESULTS/'fusion_history_B1_20261028'
PARAMETERS = RESULTS/'source_base_B1_20261028/parameters.json'
PARAMETERS_SHA = 'ef59c9a4335fd4398c945cd2fc138f847f8a7a6599e77b5e742033bd039588db'
FREEZE_SHA = '6054463cca7944e0e4d4c93593c1651c0db0213eca5efad952135e12c2c61105'
MANIFEST_SHA = '2f33c189bd3b6bdb8e39adead4bfc6e438e3b977b861ad914635a45db21fc994'
CSV_SHA = '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
EXPECTED = {0: (1059, 119), 1: (1356, 133), 2: (1443, 168), 3: (1391, 162)}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def import_fields(cache_dir):
    if os.uname().sysname != 'Linux':
        raise RuntimeError('Convert these existing assets on the lab server')
    for path, digest in ((PARAMETERS, PARAMETERS_SHA), (ASSETS/'freeze.json', FREEZE_SHA),
                         (ASSETS/'manifest.json', MANIFEST_SHA)):
        if _sha(path) != digest:
            raise ValueError(f'Original asset checksum mismatch: {path}')
    freeze, manifest = _read(ASSETS/'freeze.json'), _read(ASSETS/'manifest.json')
    params, source = _read(PARAMETERS), _read(PARAMETERS.parent/'freeze.json')
    done = _read(ASSETS/'UJI_B0_finished.json')
    if not (params['accepted'] and params['source_building'] == 'UJI_B1'
            and _sha(PARAMETERS.parent/'freeze.json') == params['freeze_sha256']
            and source['source_floors'] == ['B1F0', 'B1F1', 'B1F2', 'B1F3']
            and freeze['parameters_sha256'] == PARAMETERS_SHA
            and freeze['base'] == params['base'] and freeze['obs'] == params['obs']
            and done['complete'] and done['freeze_sha256'] == FREEZE_SHA
            and freeze['input_sha256'][str(ASSETS/'manifest.json')] == MANIFEST_SHA):
        raise ValueError('B1 parameter provenance or B0 completion binding mismatch')
    csv = next(Path(p) for p in freeze['input_sha256'] if p.endswith('/trainingData.csv'))
    if _sha(csv) != CSV_SHA or freeze['input_sha256'][str(csv)] != CSV_SHA:
        raise ValueError('Original full-UJI CSV identity mismatch')
    buildings, axes = load('uji', root=csv)
    task = SimpleNamespace(dataset='uji', building='B0', H=buildings['B0'], ap_ids=axes['B0'])
    selections = {s['floor']: s for s in manifest['selections'] if s['building'] == 'UJI_B0'}
    if set(selections) != set(EXPECTED) or task.ap_ids != [f'WAP{i:03}' for i in range(1, 521)]:
        raise ValueError('Expected the four full B0 floors and original 520-AP axis')
    cache_dir = Path(cache_dir).resolve()
    total, skipped, keys = 0, 0, {}
    for floor, (nrows, naps) in EXPECTED.items():
        h, sel = task.H[floor], selections[floor]
        if not (len(h['rss']) == nrows == sel['scans']
                and sel['rows'] == list(range(nrows)) and sel['AP_ids'] == task.ap_ids
                and sel['scan_ids'] == [f'trainingData.csv@row{i}' for i in h['row_ids']]
                and sel['observed_APs'] == naps):
            raise ValueError(f'B0F{floor} full raw row/AP binding mismatch')
        detected = (h['rss'] != 100).sum(0)
        if int((detected > 0).sum()) != naps:
            raise ValueError(f'B0F{floor} observed AP set mismatch')
        origin = np.asarray(h['xy'], dtype=np.float64).mean(0)
        old_origin = np.asarray(sel['origin'], dtype=np.float64)
        expected_z = np.unique(h['xy']-old_origin, axis=0)
        key = history_key(task, floor, PARAMETERS_SHA)
        dest = cache_dir/key
        dest.mkdir(parents=True, exist_ok=False)
        binding = dict(source_assets=str(ASSETS), freeze_sha256=FREEZE_SHA, manifest_sha256=MANIFEST_SHA,
            parameters_path=str(PARAMETERS), parameters_sha256=PARAMETERS_SHA,
            source_parameter_freeze_sha256=params['freeze_sha256'], csv_path=str(csv), csv_sha256=CSV_SHA,
            history_key=key, floor=floor, selection=sel, source_origin=origin.tolist(),
            empty_APs=np.flatnonzero(detected == 0).tolist(),
            legacy_max_iterations=freeze['VI_maxiter'], conversion='z + old_origin - source_origin; mean/cov unchanged')
        binding_path = dest/'legacy_binding.json'
        binding_path.write_text(json.dumps(binding, indent=2)+'\n')
        for ap, ap_id in enumerate(task.ap_ids):
            old_path = ASSETS/f'UJI_B0/F{floor}/ap{ap}.npz'
            old_meta_path = old_path.with_suffix('.json')
            record = _read(old_meta_path)
            if not (record['AP'] == ap and record['parameters_sha256'] == PARAMETERS_SHA
                    and record['cache_sha256'] == _sha(old_path)
                    and record['detections'] == int(detected[ap])
                    and record['empty'] == bool(detected[ap] == 0)):
                raise ValueError(f'Original field identity mismatch: {old_path}')
            with np.load(old_path, allow_pickle=False) as old:
                if bool(old['empty']) != record['empty']:
                    raise ValueError(f'Original empty-field flag mismatch: {old_path}')
                if record['empty']:
                    skipped += 1
                    continue
                if not (np.array_equal(old['z'], expected_z)
                        and old['mean'].shape == (len(expected_z),)
                        and old['cov'].shape == (len(expected_z), len(expected_z))):
                    raise ValueError(f'Original field position/covariance binding mismatch: {old_path}')
                path = dest/f'{ap}.npz'
                np.savez_compressed(path, z=old['z']+old_origin-origin,
                                    mean=old['mean'], cov=old['cov'], origin=origin)
            path.with_suffix('.json').write_text(json.dumps(dict(history_key=key, ap_id=ap_id,
                sha256=_sha(path), solver=record['optimizer'],
                imported_from=dict(path=str(old_path), sha256=record['cache_sha256'],
                    metadata_path=str(old_meta_path), metadata_sha256=_sha(old_meta_path),
                    binding_path=str(binding_path), binding_sha256=_sha(binding_path))), indent=2)+'\n')
            total += 1
        keys[str(floor)] = key
    return dict(imported_fields=total, skipped_empty_fields=skipped, history_keys=keys,
        gp_options=dict(parameters_path=str(PARAMETERS), cache_dir=str(cache_dir), fit_missing_history=False,
            provenance=dict(parameters_sha256=PARAMETERS_SHA,
                training_sources=[dict(dataset='uji', building='B1')],
                evidence=dict(freeze_path=str(PARAMETERS.parent/'freeze.json'),
                              freeze_sha256=params['freeze_sha256'], source_floors=source['source_floors']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True, type=Path)
    print(json.dumps(import_fields(parser.parse_args().cache_dir), indent=2))
