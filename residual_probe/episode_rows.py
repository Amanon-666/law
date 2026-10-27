"""Row selection for new source tasks; no RSS, model, or legacy-driver imports.

The caller supplies an already separated training/development row pool.
This function does not turn another random split of a known building into an
independent environment. One real scan is retained per exact label position.
Old frozen episodes are never rewritten through this entry.
"""
import random


def nested_rows(positions, eligible_rows, budgets, query_count, seed):
    """Choose Q once, then a nested S sequence with independent scan draws.

    positions is indexed by original row. eligible_rows must come from the
    caller's explicit data partition, not from model errors or AP visibility.
    The result contains original row indices; APs and domain must subsequently
    be built from H and each *current* S, never from Q or the largest S budget.
    """
    groups = {}
    for row in eligible_rows:
        groups.setdefault(tuple(positions[row]), []).append(row)
    if len(groups) < query_count + max(budgets):
        raise ValueError('The permitted pool has too few distinct positions')
    points = list(groups)
    query_points = random.Random(f'{seed}/query-positions').sample(points, query_count)
    query_set = set(query_points)
    support_points = [point for point in points if point not in query_set]
    random.Random(f'{seed}/support-positions').shuffle(support_points)
    query_rng = random.Random(f'{seed}/query-scans')
    support_rng = random.Random(f'{seed}/support-scans')
    query = [query_rng.choice(groups[point]) for point in query_points]
    support = [support_rng.choice(groups[point])
               for point in support_points[:max(budgets)]]
    return {'query_rows': query,
            'support_rows': {k: support[:k] for k in budgets}}


def check():
    """Catch budget-dependent Q, nonnested S, and repeated-position leakage."""
    positions = [(i, 0) for i in range(40) for _ in range(2)]
    allowed = list(range(6, 80))
    small = nested_rows(positions, allowed, (1, 3, 5), 12, 261027)
    large = nested_rows(positions, allowed, (1, 3, 5, 10, 20), 12, 261027)
    assert small['query_rows'] == large['query_rows']
    for k, rows in small['support_rows'].items():
        assert rows == large['support_rows'][k]
    q = {positions[row] for row in large['query_rows']}
    s = {positions[row] for row in large['support_rows'][20]}
    assert len(q) == 12 and len(s) == 20 and q.isdisjoint(s)
    assert set(large['query_rows'] + large['support_rows'][20]) <= set(allowed)
    assert large == nested_rows(positions, allowed, (1, 3, 5, 10, 20), 12, 261027)
    return {'fixed_query': True, 'nested_support': True,
            'whole_position_exclusion': True, 'explicit_row_pool': True,
            'real_data_scored': False}


if __name__ == '__main__':
    import json
    print(json.dumps(check()))
