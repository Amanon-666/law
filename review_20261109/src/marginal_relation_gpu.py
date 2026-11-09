"""Same finite historical-kernel sums as marginal_relation, on one CUDA GPU."""
import numpy as np
import torch

from marginal_relation import history_tables as cpu_history_tables


@torch.no_grad()
def history_tables(proxy, mean, edges):
    device = torch.device('cuda:0')
    rss = torch.as_tensor(proxy['rss'], dtype=torch.float64, device=device)
    cuts = torch.as_tensor(mean, dtype=torch.float64, device=device)
    spatial = torch.as_tensor(proxy['spatial'], dtype=torch.float64, device=device)
    group = torch.as_tensor(proxy['group'], dtype=torch.long, device=device)
    edge = torch.as_tensor(edges, dtype=torch.long, device=device)
    detected = (rss != 100).to(torch.float64)
    missing = 1-detected
    nodes = np.empty((*mean.shape, 3))
    pairs = np.empty((len(mean), len(edges), 3, 3))
    a, b = edge.T
    for start in range(0, len(mean), 32):
        sl = slice(start, start+32)
        weights = spatial[sl, group]
        z = (cuts[sl, None, :]-rss[None, :, :])/proxy['bandwidth']
        # erfc form retains small negative-tail probabilities in float64.
        low = .5*torch.special.erfc(-z/(2.**.5))*detected
        kernel = (missing.expand_as(low), low, detected-low)
        for i in range(3):
            nodes[sl, :, i] = torch.einsum('gn,gna->ga', weights, kernel[i]).cpu().numpy()
        for i in range(3):
            left = kernel[i][:, :, a]
            for j in range(3):
                pairs[sl, :, i, j] = torch.einsum('gn,gne,gne->ge', weights, left,
                                                 kernel[j][:, :, b]).cpu().numpy()
    aa, bb = edges.T
    assert np.max(np.abs(nodes.sum(-1)-1)) <= 1e-10
    assert np.max(np.abs(pairs.sum(-1)-nodes[:, aa])) <= 1e-10
    assert np.max(np.abs(pairs.sum(-2)-nodes[:, bb])) <= 1e-10
    return nodes, pairs


def check_backend(proxy, mean, edges):
    indices = np.linspace(0, len(mean)-1, 8, dtype=int)
    part = dict(proxy, spatial=proxy['spatial'][indices])
    expected = cpu_history_tables(part, mean[indices], edges)
    actual = history_tables(part, mean[indices], edges)
    error = max(float(np.max(np.abs(a-b))) for a, b in zip(actual, expected))
    assert error <= 1e-10, error
    return dict(passed=True, candidate_positions=8, maximum_absolute_table_error=error,
                dtype='float64', algorithm='Same finite sums; only hardware and chunk size differ')
