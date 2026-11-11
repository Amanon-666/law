"""Position-balanced errors. Failed tasks stay visible in aggregate coverage."""
import numpy as np


def score(prediction, truth):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != truth['xy'].shape or not np.isfinite(prediction).all():
        raise ValueError('Prediction must contain a finite XY for every Query scan')
    errors = np.linalg.norm(prediction - truth['xy'], axis=1)
    _, groups = np.unique(truth['groups'], return_inverse=True)
    groups = groups.ravel()
    position_errors = np.bincount(groups, weights=errors) / np.bincount(groups)
    return dict(mean=float(position_errors.mean()),
        p50=float(np.quantile(position_errors, .5)),
        p90=float(np.quantile(position_errors, .9)),
        query_scans=len(errors), query_positions=len(position_errors),
        scan_errors=errors.tolist(), position_errors=position_errors.tolist())


def aggregate(records):
    """Per-K successful-position statistics plus the full task denominator.

    Do not rank partial-coverage summaries against complete runs. Detailed
    task records remain the source for comparisons on a shared set of tasks.
    """
    output = {}
    for k in sorted({r['K'] for r in records}):
        selected = [r for r in records if r['K'] == k]
        done = [r for r in selected if r['status'] == 'completed']
        errors = [e for r in done for e in r['scores']['position_errors']]
        output[str(k)] = dict(tasks=len(selected), completed_tasks=len(done),
            failed_tasks=len(selected) - len(done), complete=len(done) == len(selected),
            unfinished_optimization_tasks=[r['task_id'] for r in done
                if r.get('diagnostics', {}).get('status') == 'completed_unconverged'],
            mean=float(np.mean(errors)) if errors else None,
            p50=float(np.quantile(errors, .5)) if errors else None,
            p90=float(np.quantile(errors, .9)) if errors else None,
            query_positions=len(errors))
    return output
