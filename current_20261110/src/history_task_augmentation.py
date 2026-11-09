"""Training-only, replayable coordinate and historical-signal task changes.

Two blocks are implemented. One arbitrary-angle rotation and optional mirror
is shared by ALL H/S/Q coordinates, about the H/S-only mean. This changes
centered per-axis supervision and encourages context-label interpretation;
it does not specifically force S use. Euclidean geometry is preserved in
exact arithmetic, but normalized per-axis pinball and rectangular bounds
need not be rotation invariant. Historical RSS remains unchanged; only the
target floor's support AND query share one additive dB offset.
Ordinary dense-context A+ uses exactly the same semantics:
all its own-floor support/query rows share the target offset; other floors
remain separate, unshifted histories. A matched task key gives the same target offset
across task organisations. Metadata chooses the random transform; Q RSS/XY,
Q labels, scores and row IDs do not choose its seed, range or coefficients.

The existing raw Tab consumer, physical-floor keys, loss and deployment are
unchanged; training labels share the recorded coordinate-frame transform.
Only detected RSS entries are shifted. Masks,
AP identities, row order/count and provenance remain unchanged; grouped RSS
statistics are rebuilt with uji.group_scans and bounds/APs with the original
bounds_and_aps. This is a data relation perturbation, not a fitted target map,
new real acquisition or recovery of censored signal values.

No independent coordinate/floor translation is included: original y centering removes
the former; installed PreprocessingPipeline standardizes BEFORE none/power,
removing a uniform floor-column translation. AP-specific offsets, AP dropout
and query-only jitter are deliberately outside this module.

The caller MUST supply an explicit discrete target_shift_pool_db, plus a
source-only evidence description. There is no presumed 'typical 6 dB' default.
Before use, pool selection must be frozen from
permitted source data or explicitly declared a
hypothesis. Same-XY acquisition contrasts can describe a source range, but
cannot establish that a persistent scalar target-floor shift is shared by
future independent S/Q acquisitions. Existing acquisition differences remain
in the raw data; this augmentation merely adds a declared shared component.

No clipping or rounding is performed, including for identity. Continuous dB
values can extend beyond the empirical observed range; mask preservation does
not model threshold-driven detection changes. Relative RSS and mask patterns
can bypass a scalar shift, so neither S necessity, a unique Bayes solution,
memorization as the observed failure cause, nor transfer improvement follows.
The module performs no fitting, file I/O, model execution or range selection.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from check_tabicl_history_learning import bounds_and_aps
from uji import group_scans

KIND = 'shared_rigid_coordinates_and_target_scalar_rss_v1'
NO_DETECT = 100


def sample_transform(history_names, target_floor, *, seed, task_key,
                     range_evidence, target_shift_pool_db, coordinates=True):
    """Return a JSON-serializable complete transform, without reading scans.

    history_names are original source-floor keys, not acquisition identifiers.
    target_floor is event['floor']. task_key is a prespecified schedule key
    (e.g. fold/update/target-floor); it must not be constructed from Q values
    or evaluation results. Use the SAME key, seed and range for matched C/A+
    tasks, irrespective of context size or support budget. Target RSS has its
    own SHA256-derived stream, independent of history dict order or cardinality.
    Coordinate and signal draws have separate streams.
    coordinates=False plus zero shifts is true identity, not a clipping pass.
    """
    names = sorted(history_names)
    if len(names) != len(set(names)) or target_floor in names:
        raise ValueError('Histories must be distinct other-floor sources')
    pool = np.asarray(target_shift_pool_db, dtype=np.float64)
    if pool.ndim != 1 or not len(pool) or not np.isfinite(pool).all():
        raise ValueError('target_shift_pool_db must be a nonempty finite scalar pool')
    if not isinstance(range_evidence, str) or not range_evidence.strip():
        raise ValueError('Supply the source-only range provenance or explicit hypothesis')

    def stream(block, floor=None):
        payload = json.dumps([KIND, int(seed), task_key, block, floor],
                             sort_keys=True, separators=(',', ':')).encode()
        local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        return np.random.default_rng(local_seed)

    rng = stream('coordinates')
    theta = float(rng.uniform(0., 2*np.pi)) if coordinates else 0.
    reflected = bool(rng.random() < .5) if coordinates else False
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]]) @ np.diag([1., -1. if reflected else 1.])

    return dict(kind=KIND, seed=int(seed), task_key=deepcopy(task_key),
        target_shift_pool_db=pool.tolist(),
        range_evidence=range_evidence, rotation=rotation.tolist(),
        theta_radians=theta, reflected=reflected, center_rule='Current H/S mean only',
        target_floor=target_floor, target_shift_db=float(stream('RSS', target_floor).choice(pool)),
        history_names=names, historical_RSS_unchanged=True,
        detected_only=True, clipping=False, rounding=False,
        seed_rule='SHA256([kind,seed,task_key,block,physical_floor]) first8 bytes little-endian')


def _transform_scan(scans, offset, rotation, center, move_coordinates):
    """Keep additional provenance fields; rebuild statistics after a real change."""
    out = deepcopy(scans)
    if offset == 0 and not move_coordinates:
        return out
    original = np.asarray(scans['rss'])
    detected = original != NO_DETECT
    rss = original.astype(np.float64, copy=True) if offset != 0 else original.copy()
    if offset != 0:
        rss[detected] += offset
    if not np.isfinite(rss[detected]).all() or np.any(rss[detected] == NO_DETECT):
        raise ValueError('Offset produces invalid values or collides with the missing sentinel')
    xy = np.array(scans['xy'], dtype=np.float64, copy=True)
    if move_coordinates:
        xy = (xy-center)@rotation.T+center
    out.update(group_scans(rss, xy))
    return out


def apply_transform(p, transform):
    """Return a fresh pack usable by original fit/backward; never mutate p.

    p must contain event, histories, support, query, bounds and aps as returned
    by the existing source/ordinary packing interfaces. All coordinate labels
    use the same rigid transform. Target group provenance stays as
    recorded; sharing an artificial offset does not merge acquisition groups.
    Replay uses the stored numbers, not a newly seeded draw. The identity
    returns a deep copy of every original value plus augmentation metadata.
    """
    if transform['kind'] != KIND:
        raise ValueError('Unknown augmentation transform')
    if transform['target_floor'] != p['event']['floor']:
        raise ValueError('Transform target floor differs from the packed task')
    if set(transform['history_names']) != set(p['histories']):
        raise ValueError('Transform sources differ from the packed task')
    target_offset = float(transform['target_shift_db'])
    if not np.isfinite(target_offset):
        raise ValueError('Target offset must be finite')
    rotation = np.asarray(transform['rotation'], dtype=np.float64)
    if rotation.shape != (2, 2) or not np.allclose(rotation.T@rotation, np.eye(2), rtol=0, atol=1e-12):
        raise ValueError('The coordinate transform must be orthogonal')
    move_coordinates = not np.array_equal(rotation, np.eye(2))
    # Computed from H/S, never from Q. A replayed complete record stores this
    # center explicitly. C/A+ centers may differ by a translation only; their
    # common rotation and the consumer's centering give the same frame rule.
    center = (np.asarray(transform['coordinate_center'], dtype=np.float64)
              if 'coordinate_center' in transform else
              np.concatenate([h['xy'] for h in p['histories'].values()]+[p['support']['xy']]).mean(0))
    out = deepcopy(p)
    out['augmentation'] = deepcopy(transform)
    out['augmentation']['coordinate_center'] = center.tolist()
    if not move_coordinates and target_offset == 0:
        return out
    out['histories'] = {name: _transform_scan(scans, 0., rotation, center, move_coordinates)
                        for name, scans in p['histories'].items()}
    out['support'] = _transform_scan(p['support'], target_offset, rotation, center, move_coordinates)
    out['query'] = _transform_scan(p['query'], target_offset, rotation, center, move_coordinates)
    out['bounds'], out['aps'] = bounds_and_aps(out['histories'], out['support'])
    return out


def augment_episode(p, *, seed, task_key, range_evidence, target_shift_pool_db,
                    coordinates=True):
    """Sample from task metadata, then apply; callers may store/replay the record."""
    transform = sample_transform(p['histories'], p['event']['floor'], seed=seed,
        task_key=task_key, target_shift_pool_db=target_shift_pool_db,
        range_evidence=range_evidence, coordinates=coordinates)
    return apply_transform(p, transform)
