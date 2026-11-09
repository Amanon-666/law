"""Existing UAH2016 F1 assets for one frozen external development confirmation.

No model fitting, RSS-based episode selection, downloads or evaluation here.
The metadata-only manifest fixes two file/route directions and nested K=1/3/5.
H is exactly the old cached selection: F0/F2/F3 = 128/128/109 scans, not the
full available history. The original common geodetic metric frame and BSSID
axis are preserved; B20F* names encode physical floors, not new acquisitions.

S/Q are real R1 trajectory scans with POSI-interpolated labels. The already
performed metadata audit found unique XY across the four F1 files; this module
does not reopen Q coordinates to choose episodes. Different files/routes do
not establish temporal independence, a blind test, or static repeated scans.
Old UAH results were exposed. Current GP+ordinary fixed-half is the system to
confirm, not a claim that old cached GP scores equal the current retained GP.

build_manifest reads file identities, selections and cache metadata/hashes,
never RSS or XY arrays. load_context decodes only selected H/S rows. Query
RSS and truth have separate explicit readers, for prediction and later scoring.
The NPY row reader avoids np.load()[whole_array][rows], which would interpret
all target Q RSS/coordinates merely to extract the permitted support subset.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
DATA = OLD/'results/public_corpus_adapter_v1_20261026/data'
CACHE = OLD/'results/public_corpus_history_v1_20261026'
BUILDING = 'UAH2016-b20'
SEED = 11509169
FLOORS = (0, 2, 3)
FILES = ('logfile_UAH_R1_S3.txt', 'logfile_UAH_R4_S4.txt')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _row_array(path, key, rows):
    """Decode selected rows of one existing C-order numeric NPY zip member."""
    rows = list(rows)
    with ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        assert version in ((1, 0), (2, 0)), version
        header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran_order, dtype = header(stream)
        assert len(shape) == 2 and not fortran_order and not dtype.hasobject
        start, row_bytes = stream.tell(), shape[1]*dtype.itemsize
        result = np.empty((len(rows), shape[1]), dtype=dtype)
        for i, row in enumerate(rows):
            assert 0 <= row < shape[0]
            stream.seek(start+row*row_bytes)
            result[i] = np.frombuffer(stream.read(row_bytes), dtype=dtype, count=shape[1])
    return result


def build_manifest(data_root=DATA, cache_root=CACHE):
    """Return JSON-serializable contract and six events using metadata only.

    Each event has id/floor/direction/K/R, support_rows/query_rows (zero-based
    F1 NPZ row indices), original scan IDs and histories=['B20F0',...]. Global
    history_rows holds the exact cached rows once. Paths and SHA256s bind the
    raw logs, converted arrays, old fit parameters, selections and GP fields.
    The caller persists this manifest before reading any H/S measurements.
    """
    data_root, cache_root = Path(data_root), Path(cache_root)
    building = data_root/BUILDING
    metadata_path, corpus_path = building/'metadata.json', data_root/'UAH2016_manifest.json'
    metadata, corpus = read(metadata_path), read(corpus_path)
    frozen_path = cache_root/'freeze.json'
    frozen = read(frozen_path)
    parameters_path = Path(frozen['parameters'])
    assert digest(parameters_path) == frozen['parameters_sha256']
    parameters = read(parameters_path)
    assert parameters['base'] == frozen['base'] and parameters['obs'] == frozen['obs']
    assert read(cache_root/'summary.json')['complete']
    assert metadata['name'] == BUILDING and len(metadata['ap_ids']) == 351
    assert metadata['max_supported_R'] == 1
    assert metadata['transform']['floor_alignment'] == 'shared geodetic frame; no per-floor fitted transform'
    assert corpus['partitions'] == ['01-Training']
    inputs = {str(p): digest(p) for p in (metadata_path, corpus_path, frozen_path,
                                         cache_root/'summary.json', parameters_path)}
    floor_paths = {f'B20F{f}': str(building/f'F{f}.npz') for f in (0, 1, 2, 3)}
    for path in floor_paths.values():
        sha = digest(path)
        assert frozen['input_sha256'][path] == sha
        inputs[path] = sha
    # Metadata identifies acquisitions; no target RSS or coordinates are opened.
    with np.load(floor_paths['B20F1'], allow_pickle=False) as saved:
        target_ids = saved['scan_ids'].astype(str).tolist()
    candidates = [[i for i, sid in enumerate(target_ids)
                   if Path(sid.rsplit('@', 1)[0]).name == filename] for filename in FILES]
    assert list(map(len, candidates)) == [63, 32]
    assert metadata['floors']['1']['max_labels_at_exact_position'] == 1
    histories, history_ids, fields = {}, {}, {}
    for floor, count in zip(FLOORS, (128, 128, 109)):
        name = f'B20F{floor}'
        directory = cache_root/BUILDING/f'F{floor}'
        selection_path = directory/'selection.json'
        selection = read(selection_path)
        rows = selection['row_indices']
        assert len(rows) == selection['used_scans'] == count
        with np.load(floor_paths[name], allow_pickle=False) as saved:
            scan_ids = saved['scan_ids'][rows].astype(str).tolist()
        assert scan_ids == selection['scan_ids']
        histories[name], history_ids[name] = rows, scan_ids
        inputs[str(selection_path)] = digest(selection_path)
        fields[name] = {}
        for ap in range(351):
            path, status = directory/f'ap{ap}.npz', directory/f'ap{ap}.json'
            info = read(status)
            assert info['AP'] == ap and info['finite']
            fields[name][str(ap)] = dict(path=str(path), sha256=digest(path), empty=info['empty'])
            inputs[str(path)] = fields[name][str(ap)]['sha256']
            inputs[str(status)] = digest(status)
    for source in corpus['sources']:
        path = Path(source['path'])
        assert digest(path) == source['sha256']
        inputs[str(path)] = source['sha256']
    events = []
    for direction, (s_index, q_index) in zip(('AB', 'BA'), ((0, 1), (1, 0))):
        seed = int.from_bytes(hashlib.sha256(f'{SEED}/{direction}'.encode()).digest()[:8], 'little')
        selected = np.random.default_rng(seed).permutation(candidates[s_index])[:5].tolist()
        for k in (1, 3, 5):
            support_rows, query_rows = selected[:k], candidates[q_index]
            assert not set(support_rows).intersection(query_rows)
            events.append(dict(id=f'UAH2016_B20F1_{direction}_K{k}', floor='B20F1', building='UAH2016-b20',
                direction=direction, K=k, R=1, seed=seed, histories=list(histories),
                support_rows=support_rows, query_rows=query_rows,
                support_scan_ids=[target_ids[i] for i in support_rows],
                query_scan_ids=[target_ids[i] for i in query_rows],
                support_file=FILES[s_index], query_file=FILES[q_index]))
    return dict(schema='uah-fixed-system-assets-v1', seed=SEED, events=events,
        floor_paths=floor_paths, history_rows=histories, history_scan_ids=history_ids,
        fields=fields, ap_ids=metadata['ap_ids'], ap_namespace=metadata['ap_namespace'],
        parameters_path=str(parameters_path), parameters_sha256=digest(parameters_path),
        parameters=parameters, inputs=inputs, code_sha256=digest(__file__),
        contract=dict(H='Exactly cached original F0/F2/F3 selected rows: 128+128+109',
            split='R1_S3 support to all R4_S4 queries and reverse; one fixed nested K1/3/5 draw per direction',
            selection='Only file identities; seed11509169/direction SHA256 first8 little endian; no RSS/score selection',
            coordinates='Original common public metric frame; context H mean is optional common origin',
            GP_cache='empty or z/mean/cov; no stored origin; z is absolute public-frame XY; subtract event origin once',
            parameters='Original frozen UJI B0F0/F1/F2 trainingData base/obs,32 calibration updates; no UAH target Q',
            APs='Fixed 351 identity directory; event active columns selected by H/current S only',
            domain='H/current S absolute bounds, outward floor/ceil at 2 metre grid spacing',
            query='Only explicit query reader may decode Q RSS; query truth is a separate post-prediction read',
            limits=['External dataset development confirmation, not blind or longitudinal',
                    'Real trajectory R1 scans, POSI-interpolated labels; not static independent repeats',
                    'Old GP cache is partial H and tied to its original parameters, not current cleanB0 refitting',
                    'Different files/routes and disjoint XY do not establish statistical independence']))


def load_context(manifest, event):
    """Read exactly the selected H/S rows; return absolute coordinates and H-only origin."""
    histories = {name: dict(rss=_row_array(manifest['floor_paths'][name], 'rss', manifest['history_rows'][name]),
                           xy=_row_array(manifest['floor_paths'][name], 'xy', manifest['history_rows'][name]))
                 for name in event['histories']}
    path = manifest['floor_paths'][event['floor']]
    support = dict(rss=_row_array(path, 'rss', event['support_rows']),
                   xy=_row_array(path, 'xy', event['support_rows']))
    rss = np.concatenate([h['rss'] for h in histories.values()]+[support['rss']])
    xy = np.concatenate([h['xy'] for h in histories.values()]+[support['xy']])
    aps = np.flatnonzero((rss != 100).any(0))
    bounds = np.stack((np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2))
    origin = np.concatenate([h['xy'] for h in histories.values()]).mean(0)
    return dict(histories=histories, support=support, aps=aps, bounds=bounds, origin=origin)


def load_history(manifest, floor, ap, origin):
    """Restore the existing posterior in the event frame; no fitting or added noise.

    Returns None only for an explicitly empty old cache. The field's mean/cov
    are latent posterior moments. Use manifest['parameters']['base'] for the
    whole GP task as well; replacing the prior invalidates these saved fields.
    """
    import torch
    from fields import History

    record = manifest['fields'][floor][str(int(ap))]
    assert digest(record['path']) == record['sha256']
    with np.load(record['path'], allow_pickle=False) as saved:
        assert bool(saved['empty']) == record['empty']
        if record['empty']:
            return None
        z = torch.as_tensor(saved['z']-np.asarray(origin), dtype=torch.float64)
        posterior = dict(mean=torch.as_tensor(saved['mean'], dtype=torch.float64),
                         cov=torch.as_tensor(saved['cov'], dtype=torch.float64))
    return History(z, posterior, manifest['parameters']['base'])


def read_query_rss(manifest, event):
    """Prediction-only input; does not open query coordinates."""
    return _row_array(manifest['floor_paths'][event['floor']], 'rss', event['query_rows'])


def read_query_truth(manifest, event):
    """Call only after all event predictions are persisted; never for preparation."""
    return _row_array(manifest['floor_paths'][event['floor']], 'xy', event['query_rows'])
