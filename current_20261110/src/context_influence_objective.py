"""Source feedback for the unchanged projected GP/Tab half-average.

No model, optimizer, inference, file I/O, or training schedule is defined here.
The coordinate cotangent is evaluated at the FULL ensemble prediction, not an
average of separate view losses. Quantile means and ensemble affine scales
are handled by the consumer. GP input is its already projected frozen point.

This objective aligns source optimization with the deployed system's mean
Euclidean error; it is not a new loss or proof of transferable history use.
It can learn compensation for a source-specific GP error, which may not
transfer. Projection limits both reachable outputs and available gradients.
"""
import torch


def system_coordinate_feedback(tab_raw, gp_projected, source_xy, tab_bounds, position_weights):
    """Return source MDE and dL/dt for the complete unprojected Tab mean t.

    All inputs are frozen source-task quantities [N,2] (bounds [2,2], weights
    [N]); weights sum to one and give repeated scans equal total position mass.
    The independent leaf avoids retaining all Tab view graphs simultaneously.
    At zero Euclidean residual PyTorch chooses the zero subgradient. Clamp has
    zero derivative outside bounds and its native boundary subgradient at an
    exact bound. No straight-through gradient or enlarged output box is used.
    """
    with torch.enable_grad():
        point = tab_raw.detach().requires_grad_(True)
        projected = torch.clamp(point, min=tab_bounds[0], max=tab_bounds[1])
        fused = .5*(gp_projected.detach()+projected)
        loss = (position_weights*torch.linalg.vector_norm(fused-source_xy.detach(), dim=1)).sum()
        feedback, = torch.autograd.grad(loss, point)
    return loss.detach(), feedback.detach()
