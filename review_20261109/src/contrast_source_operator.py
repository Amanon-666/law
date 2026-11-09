"""Source spatial coefficients invariant to a common RSS shift at fixed masks.

Only the CNN's conditioning features are centered. Original historical GP
levels, source adaptation, censored observations and query RSS remain intact.
This is not invariance of re-fitting the whole censored GP from shifted data.
"""
import torch

from neural_history_operator import history_context


def contrast_features(features):
    """Six raster channels; one shared center across all APs of this source."""
    density = features[:, 0].expm1()
    values, seen = features[:, 2], features[:, 5]
    mass = (density*seen).sum()
    center = (density*values).sum()/mass.clamp_min(torch.finfo(features.dtype).tiny)
    return torch.stack([values-center*seen if i == 2 else features[:, i]
                        for i in range(6)], dim=1)


def contrast_history_context(histories, weights, base, obs, rank=8):
    """Prepare once per H; use with the existing SourcePreservingOperator."""
    context = history_context(histories, weights, base, obs, rank=rank)
    if context is None:
        return None
    return dict(context, histories=[contrast_features(h) for h in context['histories']])
