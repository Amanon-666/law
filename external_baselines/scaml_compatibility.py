"""Synthetic compatibility check of unchanged ScaML-GP, not a localization trial."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def main(reference, output):
    sys.path[:0] = [str(reference), str(reference / 'deps')]
    import torch
    import botorch
    import gpytorch
    from botorch.utils.datasets import SupervisedDataset
    from scamlgp.model import ScaMLGP, meta_fit_scamlgp
    from scamlgp.utils import optimize_marginal_likelihood

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(261027)
    code = {str(reference / 'scamlgp' / f): hashlib.sha256(
        (reference / 'scamlgp' / f).read_bytes()).hexdigest()
        for f in ['model.py', 'utils.py']}
    x = torch.linspace(0., 1., 8).unsqueeze(-1)
    data = {0: SupervisedDataset(x, torch.sin(3*x)),
            1: SupervisedDataset(x, torch.cos(3*x))}
    sources = meta_fit_scamlgp(data, num_restarts_log_likelihood=0, seed=261027)
    query = torch.tensor([[.15], [.45], [.85]])
    records = []
    for k in [1, 3, 5]:
        sx = torch.linspace(.1, .9, k).unsqueeze(-1)
        sy = .7*torch.sin(3*sx)+.2*torch.cos(3*sx)
        model = ScaMLGP(sx, sy, sources)
        optimize_marginal_likelihood(model, num_restarts=0)
        model.eval()
        with torch.no_grad():
            batch = model.posterior(query)
            singles = torch.cat([model.posterior(q[None]).mean for q in query])
        torch.testing.assert_close(batch.mean, singles, atol=1e-8, rtol=1e-8)
        assert torch.isfinite(batch.mean).all() and torch.isfinite(batch.variance).all()
        assert (batch.variance >= 0).all() and (model.weights > 0).all()
        records.append(dict(K=k,weights=model.weights.detach().tolist(),
            minimum_variance=float(batch.variance.min()),
            isolated_query_max_difference=float((batch.mean-singles).abs().max()),
            outcome_mean=model.outcome_transform.means.tolist(),
            outcome_std=model.outcome_transform.stdvs.tolist()))
    for path, sha in code.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    result = dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,
        scope='Synthetic source/target fitting and prediction compatibility only; no real data, no localization scores, no convergence guarantee.',
        versions=dict(python=sys.version,torch=torch.__version__,botorch=botorch.__version__,gpytorch=gpytorch.__version__),
        source_sha256=code,checks=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.reference,a.output)
