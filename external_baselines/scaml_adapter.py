"""RSS-to-coordinate protocol adapter for the unchanged official ScaML core.

Inputs are retained H scans and current S only, in the existing H-centered
coordinate frame. Query scans enter predict only. No location clipping.
"""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
from botorch import fit as botorch_fit
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.utils.datasets import SupervisedDataset
from scamlgp.model import ScaMLGP, meta_fit_scamlgp
from scamlgp.utils import optimize_marginal_likelihood


def encode_rss(rss):
    """Existing -110 dBm nondetection convention, with fixed affine scaling."""
    values = np.where(np.asarray(rss) == 100, -110., rss)
    return torch.as_tensor((values + 110.) / 110., dtype=torch.float64)


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FitTrace:
    """Observe the original optimizer; preserve fitting, warnings and restarts."""
    def __init__(self, maxiter=None):
        self.attempts = []
        self.maxiter = maxiter

    def optimizer(self, mll, **kwargs):
        model = mll.model
        record = dict(model_id=id(model), before=parameter_hash(model), maxiter=self.maxiter)
        if self.maxiter is not None:
            kwargs.setdefault('options', {})['maxiter'] = self.maxiter
        self.attempts.append(record)
        try:
            fitted, info = fit_gpytorch_scipy(mll, **kwargs)
        except Exception as error:
            record.update(exception=type(error).__name__, message=str(error))
            raise
        result = info['OptimizeResult']
        record.update(after=parameter_hash(model), success=bool(result.success),
                      status=int(result.status), message=str(result.message),
                      objective=float(result.fun) if np.isfinite(result.fun) else None, iterations=int(result.nit),
                      seconds=float(info['wall_time']))
        return fitted, info

    @contextmanager
    def observe(self):
        original = botorch_fit.fit_gpytorch_mll
        def observed(mll, **kwargs):
            return original(mll, optimizer=self.optimizer, **kwargs)
        botorch_fit.fit_gpytorch_mll = observed
        try:
            yield self
        finally:
            botorch_fit.fit_gpytorch_mll = original

    def selected(self, model):
        final_hash = parameter_hash(model)
        matches = [i for i, r in enumerate(self.attempts)
                   if r['model_id'] == id(model) and r.get('after') == final_hash]
        if not matches:
            raise RuntimeError('Returned model does not match any recorded optimizer result')
        successful = [r['objective'] for r in self.attempts
                      if r['model_id'] == id(model) and r.get('success') and r['objective'] is not None]
        values = [self.attempts[i]['objective'] for i in matches if self.attempts[i]['objective'] is not None]
        selected_objective = min(values) if values else None
        return dict(parameter_sha256=final_hash, matching_attempts=matches,
                    optimizer_success=any(self.attempts[i]['success'] for i in matches),
                    selected_optimizer_objective=selected_objective,
                    best_successful_objective=min(successful) if successful else None)


def fit_sources(histories, *, seed, restarts=5):
    hx = [encode_rss(h['rss']) for h in histories]
    hy = [torch.as_tensor(h['xy'], dtype=torch.float64) for h in histories]
    sources = []
    for axis in range(2):
        data = {i: SupervisedDataset(x, y[:, axis:axis+1])
                for i, (x, y) in enumerate(zip(hx, hy))}
        sources.append(meta_fit_scamlgp(data, num_restarts_log_likelihood=restarts,
                                       seed=seed+axis))
    return sources


class ScaMLCoordinates:
    def __init__(self, source_models, support, *, seed, restarts=5):
        self.models = []
        self.source_models = source_models
        sx = encode_rss(support['rss'])
        sy = torch.as_tensor(support['xy'], dtype=torch.float64)
        for axis, sources in enumerate(source_models):
            torch.manual_seed(seed+axis)
            model = ScaMLGP(sx, sy[:, axis:axis+1], sources)
            optimize_marginal_likelihood(model, num_restarts=restarts)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, query_rss):
        x = encode_rss(query_rss)
        posteriors = [model.posterior(x) for model in self.models]
        return (torch.cat([p.mean for p in posteriors], dim=-1).numpy(),
                torch.cat([p.variance for p in posteriors], dim=-1).numpy())
