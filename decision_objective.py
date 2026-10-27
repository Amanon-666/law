"""Decision loss for LAW's existing full-scan, posterior-mean readout.

Callers must sum ALL H/S-selected AP log likelihoods before this interface.
The uniform deployment grid is fixed from permitted H/S information; target
coordinates supervise source training only and never enlarge that grid.
No learning, AP selection, posterior inference, or parameter fitting lives here.
"""
import torch


def decision_terms(scores, grid, target_xy, *, distance_unit, smoothing):
    """Scores [scan,candidate]; return the existing readout and source losses.

    distance_unit and smoothing are explicit source-frozen constants. The
    nearest-grid NLL is diagnostic: it is not an extra training objective and
    is not a continuous-coordinate density score.
    """
    log_probability = scores.log_softmax(-1)
    probability = log_probability.exp()
    xy = probability @ grid
    displacement = xy - target_xy
    smooth_distance = ((displacement.square().sum(-1) + smoothing ** 2).sqrt()
                       - smoothing) / distance_unit
    nearest = (grid[None] - target_xy[:, None]).square().sum(-1).argmin(-1)
    location_nll = -log_probability.gather(1, nearest[:, None]).squeeze(1)
    return dict(xy=xy, smooth_distance=smooth_distance,
                nearest_grid_nll=location_nll)


def source_loss(scores, grid, target_xy, observation_nll, scan_weights, *,
                ap_count, beta, distance_unit, smoothing):
    """Observation NLL is the full scan score at its true source coordinate.

    scan_weights must encode the caller's position-balanced source sampling.
    AP averaging scales only observation loss; it NEVER tempers position scores.
    Building balancing and regularization remain in the existing training loop.
    """
    terms = decision_terms(scores, grid, target_xy,
                           distance_unit=distance_unit, smoothing=smoothing)
    obs = (scan_weights * observation_nll).sum() / ap_count
    decision = (scan_weights * terms['smooth_distance']).sum()
    return dict(loss=obs + beta * decision, observation_loss=obs,
                decision_loss=decision, **terms)


def check():
    """Check the analytic decision derivative and two consequential failures."""
    torch.set_default_dtype(torch.float64)
    grid = torch.tensor([[0., 0.], [1., 0.], [10., 0.]])
    truth = torch.zeros((1, 2))
    scores = torch.tensor([[.2, -.4, .7]], requires_grad=True)
    unit, epsilon = 10., .1
    terms = decision_terms(scores, grid, truth, distance_unit=unit, smoothing=epsilon)
    gradient, = torch.autograd.grad(terms['smooth_distance'].sum(), scores)
    with torch.no_grad():
        p = scores.softmax(-1)
        delta = terms['xy'] - truth
        direction = grid[None] - terms['xy'][:, None]
        analytic = p * (delta[:, None] * direction).sum(-1)
        analytic /= unit * (delta.square().sum(-1) + epsilon ** 2).sqrt()[:, None]
    torch.testing.assert_close(gradient, analytic, atol=1e-13, rtol=1e-12)

    # A better true-position probability can move the deployed mean far away.
    before = decision_terms(torch.log(torch.tensor([[.40, .59, .01]])), grid,
                            truth, distance_unit=unit, smoothing=epsilon)
    after = decision_terms(torch.log(torch.tensor([[.50, .01, .49]])), grid,
                           truth, distance_unit=unit, smoothing=epsilon)
    assert after['nearest_grid_nll'].item() < before['nearest_grid_nll'].item()
    assert after['smooth_distance'].item() > before['smooth_distance'].item()
    torch.testing.assert_close(before['xy'][0, 0], torch.tensor(.69))
    torch.testing.assert_close(after['xy'][0, 0], torch.tensor(4.91))

    # Averaging AP log scores silently changes deployment temperature.
    ap_scores = torch.tensor([[2., 0.], [0., 0.]])
    full = ap_scores.sum(0).softmax(0)
    wrong = ap_scores.mean(0).softmax(0)
    assert abs(full[0].item() - wrong[0].item()) > .1
    obs_nll = torch.tensor([4.])
    pure = source_loss(scores, grid, truth, obs_nll, torch.ones(1),
                       ap_count=2, beta=0., distance_unit=unit, smoothing=epsilon)
    torch.testing.assert_close(pure['loss'], torch.tensor(2.))
    return dict(analytic_gradient_max_error=(gradient - analytic).abs().max().item(),
        nll_improves_but_mean_distance=[before['xy'][0, 0].item(), after['xy'][0, 0].item()],
        full_scan_probability=full.tolist(), incorrectly_AP_averaged_probability=wrong.tolist(),
        beta_zero_observation_only=True, training_updates=0)


if __name__ == '__main__':
    import json
    import socket
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical checks run on ray-web')
    print(json.dumps(check()))
