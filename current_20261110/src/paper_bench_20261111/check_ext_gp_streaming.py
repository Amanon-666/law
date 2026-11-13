"""One server-only equality/lifetime check; external timeout 120 s, CPU2/GPU0.

Synthetic deterministic History objects only: no historical fitting, original
ReFi model loading, real datasets or source training. Run from the source root
with python -m paper_bench_20261111.check_ext_gp_streaming.
"""
import hashlib
import json
from pathlib import Path
import socket
import time
import weakref


def check():
    import numpy as np
    import torch
    from unittest.mock import patch
    from cached_retained_readout import score_cached_fields
    from fields import History
    from predict import adapt
    import position_fusion_gp as fusion
    from paper_bench_20261111.ext_gp_streaming import lazy_task_inputs, _NoFieldCache

    torch.set_num_threads(2)
    base = dict(mean=-75., amplitude=10., length=3., drift=4.)
    obs = dict(alpha=.9, beta=-90., detect_scale=5., noise_std=5.)
    xy = np.array([[0., 0.], [0., 0.], [2., 0.], [2., 0.], [0., 2.], [0., 2.]])
    histories = {
        '0': dict(xy=xy, rss=np.array([[-70., -76., -80., 100.], [-71., -75., -79., 100.],
                    [-74., -78., -82., 100.], [-73., -77., 100., 100.],
                    [-72., -74., -78., 100.], [-71., -75., -77., 100.]])),
        '1': dict(xy=xy, rss=np.array([[-76., -70., 100., 100.], [-75., -71., 100., 100.],
                    [-78., -74., 100., 100.], [-77., -73., 100., 100.],
                    [-74., -72., 100., 100.], [-75., -71., 100., 100.]]))}
    support = dict(xy=np.array([[1., 1.]]*3),
                   rss=np.array([[-74., -74., -80., -76.], [-73., -75., -79., 100.],
                                 [-75., -73., 100., -74.]]))
    qrss = np.array([[-72., -75., -79., 100.], [-76., -73., 100., -75.],
                     [100., -76., -80., -74.]])
    origin = np.concatenate([h['xy'] for h in histories.values()]).mean(0)
    bounds = np.array([[0., 0.], [2., 2.]])
    source_means = {
        ('0', 0): [-70., -74., -72.], ('0', 1): [-76., -78., -74.],
        ('0', 2): [-80., -82., -78.], ('1', 0): [-76., -78., -74.],
        ('1', 1): [-70., -74., -72.]}

    def evaluate(lazy):
        refs, loads = [], []
        peaks = dict(histories=0, distinct_APs=0)

        def load(source, ap):
            value = History(torch.tensor(np.unique(xy, axis=0)-origin, dtype=torch.float64),
                dict(mean=torch.tensor(source_means[source, ap], dtype=torch.float64),
                     cov=torch.eye(3, dtype=torch.float64)*2.), base)
            refs.append((weakref.ref(value), ap))
            loads.append([source, ap])
            active = [a for ref, a in refs if ref() is not None]
            peaks['histories'] = max(peaks['histories'], len(active))
            peaks['distinct_APs'] = max(peaks['distinct_APs'], len(set(active)))
            return value

        def live():
            return sum(ref() is not None for ref, _ in refs)

        if lazy:
            with patch.object(fusion, 'task_inputs', lazy_task_inputs):
                task = fusion.prepare_from_scans(histories, support, base, obs, load, origin, bounds)
        else:
            task = fusion.prepare_from_scans(histories, support, base, obs, load, origin, bounds)
        assert list(task['cases']) == [0, 1, 2, 3]
        assert task['cases'][3]['histories'] == [None, None]
        # This is the original retained-GP preliminary K check. It must not load H.
        assert len(next(iter(task['cases'].values()))['sx']) == 1
        after_prepare = len(loads)
        state = adapt(task['cases'], base, obs)
        after_adapt = live()
        prepared = fusion.prepare_retained_fields(task, state, 'synthetic_stream_equality', 1)
        after_fields = live()
        scores = score_cached_fields(prepared, qrss)
        record = dict(loads=loads, peak_live=peaks, loads_after_prepare=after_prepare,
                      live_after_adapt=after_adapt, live_after_fields=after_fields)
        if lazy:
            assert after_prepare == 0 and after_adapt == after_fields == 0
            assert peaks['histories'] <= 4 and peaks['distinct_APs'] <= 2
            assert loads == [[source, ap] for _ in range(3)
                             for ap in range(4) for source in ('0', '1')
                             if (source, ap) in source_means]
        return state, prepared, scores, record

    eager = evaluate(False)
    lazy = evaluate(True)
    assert eager[0] == lazy[0], 'Original joint weights/sigma state differs'
    assert eager[1]['diagnostics'] == lazy[1]['diagnostics'], 'Sampler/seed/ESS differs'
    errors = {}
    for key in ('mean', 'variance', 'grid'):
        errors[key] = float((eager[1][key]-lazy[1][key]).abs().max())
    for key in ('scores', 'raw', 'projected'):
        errors[key] = float(np.abs(eager[2][key]-lazy[2][key]).max())
    assert max(errors.values()) <= 1e-10, errors
    cache = _NoFieldCache()
    cache['field'] = object()
    assert not cache
    return dict(max_errors=errors, identical_adaptation_state=True,
        identical_sampler_diagnostics=True, no_history_AP=3,
        eager_lifetime=eager[3], lazy_lifetime=lazy[3], persistent_cache_disabled=True)


def main():
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Run this numerical check only on ray-web')
    root = Path(__file__).resolve().parents[1]
    out = root.parent/'results/paper_gp_streaming_check_20261113'
    out.mkdir(parents=True, exist_ok=False)
    names = [Path(__file__), Path(__file__).with_name('ext_gp_streaming.py'),
        root/'episodic_tasks.py', root/'position_fusion_gp.py', root/'adaptation.py',
        root/'fields.py', root/'law_bench/method_retained_gp.py']
    result = dict(complete=False, code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in names}, resources=dict(cpu=2, gpu=0, external_timeout_seconds=120),
        real_data_used=False, historical_fit_performed=False)
    started = time.monotonic()
    try:
        result.update(check(), complete=True)
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', seconds=time.monotonic()-started)
        (out/'failure.json').write_text(json.dumps(result, indent=2)+'\n')
        raise
    result['seconds'] = time.monotonic()-started
    (out/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
