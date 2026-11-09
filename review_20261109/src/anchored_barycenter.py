"""Fixed-anchor Gaussian Wasserstein fusion, followed by a GP conditional lift.

Research interface, not the default ReFi prior or a demonstrated improvement.
Fusion is computed ONCE on H-defined anchors, never separately on each query
batch. This is a barycenter in anchor Euclidean geometry, not the infinite-
dimensional L2 barycenter. A common reference conditional supplies the spatial
cross-covariances needed for later S conditioning.

Sources must already share coordinates, AP identity and RSS units. Reusing a
source's total weight across identical copies leaves the distribution unchanged;
adding a copy with extra weight generally does not. No source/AP selection,
negative-history completion, target weight fitting or shared learning is done.

Gaussian barycenter equation: Mallasto & Feragen, NeurIPS 2017, Theorem 4;
finite-dimensional fixed-point algorithm: Alvarez-Esteban et al. (2016).
The solver is deliberately preprocessing-only: eigensolver gradients and
end-to-end weight adaptation have NOT been validated.
"""
import torch


def symmetric(matrix):
    return (matrix + matrix.transpose(-1, -2)) / 2


def roots(matrix):
    values, vectors = torch.linalg.eigh(symmetric(matrix))
    if not bool(torch.isfinite(values).all() and (values > 0).all()):
        raise ValueError('Barycenter requires finite positive definite covariance; no clipping')
    root = (vectors * values.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    inverse = (vectors * values.rsqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return root, inverse


@torch.no_grad()
def gaussian_barycenter(means, covariances, weights, tolerance=1e-9, max_steps=100):
    """Return mean/covariance and fixed-point residual; raise on nonconvergence.

    Inputs: means[J,A], covariances[J,A,A], simplex weights[J]. The residual
    checks C = sum_j w_j (sqrt(C) C_j sqrt(C))**.5 at the returned C.
    """
    if means.ndim != 2 or covariances.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError('Incompatible source moments')
    if weights.shape != (len(means),) or not bool(torch.isfinite(weights).all()
            and (weights >= 0).all() and torch.isclose(weights.sum(), weights.new_tensor(1.))):
        raise ValueError('Expected finite simplex weights')
    active = weights > 0
    means, covariances, weights = means[active], covariances[active], weights[active]
    source_roots, _ = roots(covariances)
    covariance = torch.einsum('j,jab->ab', weights, covariances)
    for step in range(max_steps + 1):
        root, inverse = roots(covariance)
        # sqrt(M M') = U diag(s) U'. Forming M M' first squares its
        # condition number and can create negative eigenvalues by roundoff.
        # This SVD computes the SAME square root without that squaring.
        left, singular, _ = torch.linalg.svd(root[None] @ source_roots, full_matrices=False)
        transported = (left * singular.unsqueeze(-2)) @ left.transpose(-1,-2)
        average = torch.einsum('j,jab->ab', weights, transported)
        residual = torch.linalg.matrix_norm(average-covariance) / torch.linalg.matrix_norm(covariance)
        if residual <= tolerance:
            return dict(mean=weights @ means, cov=symmetric(covariance),
                        iterations=step, residual=float(residual))
        if step < max_steps:
            factor = inverse @ average
            covariance = symmetric(factor @ factor.T)
    raise RuntimeError(f'Barycenter did not converge: relative residual={float(residual):g}')


class AnchoredField:
    """q(f_A) lifted using an explicitly chosen shared GP reference conditional.

    f(x) = m0 + B(x)(u-m0) + e(x), u~N(mean,cov),
    B(x)=k(x,A)(K_AA+vI)^-1; cov(e)=k-B k(A,.). This is a valid GP independent
    of which candidate points are requested. Exact single-source recovery
    requires that source to have this same reference conditional given f_A.
    Otherwise it is an explicitly different finite-anchor approximation.

    With v>0 the anchor variable is u=f(A)+epsilon, epsilon~N(0,vI),
    not noiseless f(A). Input covariance must then describe u, including vI.
    This explicit representation noise is not RSS observation noise. Single-
    source recovery is generally approximate at positive v. No adaptive jitter
    or inversion fallback changes this choice silently.
    """
    def __init__(self, anchors, mean, covariance, kernel, reference_mean,
                 reference_noise_variance=0.):
        self.anchors, self.mean, self.cov = anchors, mean, covariance
        self.kernel, self.reference_mean = kernel, reference_mean
        self.latent_reference_mean = reference_mean
        if reference_noise_variance < 0:
            raise ValueError('Reference noise variance must be nonnegative')
        self.reference_noise_variance = reference_noise_variance
        ka = kernel(anchors, anchors)
        self.reference_covariance = ka + reference_noise_variance * torch.eye(
            len(anchors), dtype=ka.dtype, device=ka.device)
        self.chol = torch.linalg.cholesky(self.reference_covariance)
        torch.linalg.cholesky(symmetric(covariance))
        self.covariance_change = symmetric(covariance-self.reference_covariance)

    def projection(self, x):
        return torch.cholesky_solve(self.reference_cross(x).T, self.chol).T

    def reference_cross(self, x):
        return self.kernel(x, self.anchors)

    def moments(self, x, y):
        bx, by = self.projection(x), self.projection(y)
        mean = self.reference_mean + bx @ (self.mean-self.latent_reference_mean)
        return mean, self.kernel(x, y) + bx @ self.covariance_change @ by.T

    def reference_blocks(self, support, grid):
        """Reusable geometry for fields with this identical reference conditional."""
        return dict(bs=self.projection(support),bg=self.projection(grid),
            ss=self.kernel(support,support),gs=self.kernel(grid,support),
            diagonal=torch.stack([self.kernel(x[None],x[None])[0,0] for x in grid]))

    def prior_blocks(self, support, grid, change_kernel=None, intercept_std=0., reference=None):
        """ReFi condition_field-compatible SS/GS/diagonal blocks, no S RSS/Q.

        The optional target discrepancy kernel and AP intercept are independent
        additions ONCE. They must not already be included in the input sources.
        No GxG allocation; singleton/batched calls denote the same random field.
        """
        r = self.reference_blocks(support,grid) if reference is None else reference
        bs,bg = r['bs'],r['bg']
        shift = self.mean-self.latent_reference_mean
        sm,gm = self.reference_mean+bs@shift,self.reference_mean+bg@shift
        sk = r['ss'] + bs @ self.covariance_change @ bs.T
        cross = r['gs'] + bg @ self.covariance_change @ bs.T
        gv = r['diagonal'] + (bg @ self.covariance_change * bg).sum(1)
        if change_kernel is not None:
            sk = sk + change_kernel(support, support)
            cross = cross + change_kernel(grid, support)
            gv = gv + torch.stack([change_kernel(x[None], x[None])[0, 0] for x in grid])
        offset = intercept_std**2
        return dict(sm=sm, sk=sk+offset, gm=gm, cross=cross+offset, gv=gv+offset)


class AnchorSubspace:
    """Numerically resolved orthonormal coordinates of the FULL H anchor set.

    Only eigenvalues below eps * dimension * spectral radius are omitted.
    This is an explicit finite precision approximation, not anchor selection or
    prior whitening: coordinates retain the anchor Euclidean metric. Source
    field preservation must still be measured; the spectral rule alone cannot
    prove posterior approximation accuracy.
    """
    def __init__(self, anchors, kernel):
        self.anchors, self.kernel = anchors, kernel
        k = symmetric(kernel(anchors, anchors))
        self.anchor_kernel = k
        values, vectors = torch.linalg.eigh(k)
        self.threshold = float(torch.finfo(k.dtype).eps*len(k)*values.abs().max())
        if values[0] < -self.threshold:
            raise ValueError('Reference kernel has a materially negative eigenvalue')
        keep = values > self.threshold
        self.values, self.vectors = values[keep], vectors[:,keep]
        self.discarded_absolute_trace = float(values[~keep].abs().sum())

    def encode(self, mean, covariance, reference_mean):
        return (self.vectors.T @ (mean-reference_mean),
                torch.diag(self.values) + symmetric(
                    self.vectors.T @ (covariance-self.anchor_kernel) @ self.vectors))

    def encode_history(self, history, reference_mean):
        """Encode the existing identity History by its reference-relative update.

        Its cubature uses C_Z+epsilon I and its projection uses K_Z+epsilon I,
        so these epsilons cancel: C(x,y)=k(x,y)+P_x(C_Z-K_Z)P_y'.
        This avoids constructing and then subtracting two nearly equal dense
        anchor covariance matrices. None denotes exactly the reference GP.
        """
        if history is None:
            mean = self.values.new_zeros(len(self.values))
            delta = self.values.new_zeros((len(self.values),len(self.values)))
        else:
            projection = self.vectors.T @ history.projection(self.anchors)
            mean = projection @ (history.m-reference_mean)
            delta = symmetric(projection @ (history.c-history.kernel(history.z,history.z)) @ projection.T)
        return mean, torch.diag(self.values)+delta, delta

    def lift(self, mean, covariance, reference_mean, covariance_change=None):
        return SubspaceField(self, mean, covariance, reference_mean, covariance_change)


class SubspaceField(AnchoredField):
    """GP conditional lift of U' (f(A)-m0), with no added anchor noise."""
    def __init__(self, subspace, mean, covariance, reference_mean, covariance_change=None):
        self.subspace = subspace
        self.anchors, self.kernel = subspace.anchors, subspace.kernel
        self.mean, self.cov = mean, covariance
        self.reference_mean, self.latent_reference_mean = reference_mean, 0.
        self.reference_noise_variance = 0.
        self.reference_covariance = torch.diag(subspace.values)
        self.chol = torch.diag(subspace.values.sqrt())
        torch.linalg.cholesky(symmetric(covariance))
        self.covariance_change = (symmetric(covariance-self.reference_covariance)
            if covariance_change is None else covariance_change)

    def reference_cross(self, x):
        return self.kernel(x, self.anchors) @ self.subspace.vectors
