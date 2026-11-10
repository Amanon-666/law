"""Project metric 'position_mean_p90' (history_task_evaluation.aggregate): weighted 90% quantile of per-position
mean errors, weights equal over building, floor, direction, K and position. pred_of(event) -> [n_query, 2]."""
import numpy as np, core as C


def project_metrics(pred_of, Ks=(1, 3, 5)):
    floors_per_b = {}
    for e in C.MAN['events']:
        floors_per_b.setdefault(e['building'], set()).add(e['floor'])
    samples = []
    for e in C.MAN['events']:
        _, _, Q = C.event_arrays(e)
        err = np.linalg.norm(pred_of(e) - C.XY[Q], axis=1)
        _, g = np.unique(C.XY[Q], axis=0, return_inverse=True); g = g.ravel(); n = g.max() + 1
        v = np.array([err[g == i].mean() for i in range(n)])
        w = 1.0 / (len(floors_per_b) * len(floors_per_b[e['building']]) * 2 * n)
        samples.append((e['K'], v, np.full(n, w)))

    def agg(sub):
        v = np.concatenate([s[1] for s in sub]); w = np.concatenate([s[2] for s in sub]); w = w / w.sum()
        o = np.argsort(v)
        return dict(mean=float(w @ v), p90=float(v[o][np.searchsorted(np.cumsum(w[o]), .9)]))
    out = agg(samples)
    out['by_K'] = {k: agg([s for s in samples if s[0] == k]) for k in Ks}
    return out
