"""Prototype: retained-GP influence relations enter Tab's historical rows.

No GP fit, integration, scan generation, query-coordinate access or prediction
rule lives here. Factorization exposes the existing fixed-state conditional
update. source_cross must contain only persistent history covariance C_j(X,S),
before multiplying by omega_j**2; never include acquisition drift/intercept.
SS retains those terms, as in the established independent-acquisition GP.

The independent encoder receives explicit [H-row, AP, S-anchor, feature]
bindings. It uses the full normalized posterior anchor covariance, then pools
to one residual per historical row. It is a learned compression, not a lossless
encoding of all covariance or information additional to complete H/S. The
standardized GP features are separate attributes; they never pass through the
raw consumer's nonlinear column transforms. Original AP shuffles affect only
that original column stream, not these explicit AP/anchor bindings.

The hook is AFTER the entire row_interactor, BEFORE icl_predictor. In the
installed model the output is [view-batch, all-rows, flattened-row-width];
feature-view permutations preserve H/S/Q row order. Only the first N_H rows
receive a residual. S/Q row representations are untouched at this interface;
subsequent original ICL attention may of course change their predictions.
This prototype requires ordinary uncached model.forward: both official cache
modes are forbidden while enabled, not silently reused or rebuilt. No package
file, original model parameter, raw RSS, label or GP prediction is modified.

Static review limitation: AP-shared encoding followed by AP pooling discards
the binding to the raw Tab columns. Permuting complete AP relation records
while leaving the raw RSS table fixed gives the same residual. Consequently
this prototype can express AP-invariant conditional history-row modulation,
but cannot convey which raw AP column a correction belongs to. Do not use it
as an implementation of AP-directed calibration or as a lossless alternative
to per-AP GP features. It is not connected to a training or prediction runner;
upstream GP moment/row assembly and a justified consumer remain unresolved.

Zero final projection gives an initial zero residual for finite encoder values;
enabled=False attaches nothing and is the exact original computation. These
are initial/disabled computational identities, not trained risk guarantees.
After updating the backbone, disabling returns that current backbone. Save the
encoder's separate state_dict/config with any trained backbone; loading a raw
checkpoint alone does not restore this side path. No runner/checkpoint policy
or model instantiation is provided here.
"""
from contextlib import contextmanager

import torch
from torch import nn

from observations import _chol

# InfluenceEncoder/historical_influence below are the preserved first prototype.
# ColumnInfluenceEncoder/column_influence correct its AP-binding limitation by
# keeping the AP axis until relations meet the native ordered feature groups.


@torch.no_grad()
def factorized_influence(source_cross, weights, sk, sm, posterior_mean, posterior_cov):
    """Return A[J,N,K], e[K], V[K,K] for ONE AP, using original float64 solve.

    source_cross[J,N,K] uses registered, known historical positions X and
    actual support locations S; weights[J] are the already adapted empirical
    source coefficients, not posterior source responsibilities. Every input
    tensor is on one device. sk is the complete support PRIOR latent covariance;
    posterior_mean/cov are existing non-Gaussian posterior moments (not an
    observation-noise diagonal approximation). No noise is added to V.

    Sum_j A_j @ e equals the persistent-history part of the mean update at X.
    With A=sum_j A_j and B=sum_j omega_j**2 C_j, its variance correction is
    diag[A V A^T - B A^T]. Since the solve uses Keff=sk+JITTER*I, this equals
    diag[A(V-Keff)A^T], not A(V-sk)A^T with the jitter omitted. For all-H-
    absent APs the caller supplies zero source_cross, reproducing zero retained
    propagation. K=0 returns empty anchor axes and zero contributions, without
    inventing observations. The solver uses observations._chol's original
    JITTER exactly once; no altered regularizer or numerical retry.
    """
    assert source_cross.ndim == 3
    j, n, k = source_cross.shape
    assert sk.shape == (k, k) and sm.shape == posterior_mean.shape == (k,)
    assert posterior_cov.shape == (k, k)
    tensors = (source_cross, sk, sm, posterior_mean, posterior_cov)
    assert all(t.dtype == torch.float64 and t.device == source_cross.device for t in tensors)
    omega = torch.as_tensor(weights, dtype=torch.float64, device=source_cross.device)
    assert omega.shape == (j,)
    if k:
        cross = source_cross * omega.square()[:, None, None]
        a = torch.cholesky_solve(cross.reshape(j*n, k).T, _chol(sk)).T.reshape(j, n, k)
    else:
        a = source_cross.clone()
    return dict(A=a, e=(posterior_mean-sm).clone(), V=posterior_cov.clone())


class InfluenceEncoder(nn.Module):
    """Small AP-shared, anchor-equivariant relation encoder.

    features: frozen float [N_H,A,K,F], actual H row order, same AP/anchor
    ordering as normalized_covariance[A,K,K]. A row's source identity selects
    A_j and omega_j before packing. A suggested feature schema is
    (A_j, A_j*e/s, e/s, S_detection_count, S_detection_fraction, omega_j,
    (S_x-H_x)/length, (S_y-H_y)/length); raw H masks or total S counts may be
    added explicitly by the caller. s and length must be declared H/S/legal
    source scales. No AP-ID embeddings, Q features, or unknown Q positions.

    normalized_covariance is V/s**2 (per-AP positive scale), NOT merely its
    diagonal and NOT necessarily a correlation matrix. The complete signed
    matrix multiplies anchor embeddings before a shared nonlinear map. AP
    and anchor permutations must also permute covariance and all features.
    Pooling compresses those relations and gives no sufficiency guarantee.
    N_H/A/K may vary; there is no padding, so no undeclared masked mean.

    output_dim is the actual row_interactor C*E width, not input feature count.
    Construction preserves CPU RNG; parameters are initially created on CPU
    and callers explicitly move the module. hidden_dim controls this prototype
    only. Activation cost is O(N_H*A*K*hidden_dim); no runtime/fit claim is made.
    """
    def __init__(self, feature_dim, output_dim, hidden_dim=32):
        super().__init__()
        assert feature_dim > 0 and output_dim > 0 and hidden_dim > 0
        self.config = dict(feature_dim=feature_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        # Preserve original initialization/training RNG when adding this path.
        with torch.random.fork_rng(devices=[]):
            self.token = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
            self.message = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.GELU())
            self.output = nn.Linear(hidden_dim, output_dim, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, features, normalized_covariance):
        assert features.ndim == 4 and features.shape[-1] == self.config['feature_dim']
        n, a, k, _ = features.shape
        assert normalized_covariance.shape == (a, k, k)
        assert features.device == normalized_covariance.device == self.output.weight.device
        if not n or not a or not k:
            return self.output.weight.new_zeros((n, self.config['output_dim']))
        # Model/AMP handles compute precision; GP tensors remain immutable.
        h = self.token(features.to(dtype=self.output.weight.dtype))
        covariance = normalized_covariance.to(dtype=h.dtype)
        message = torch.einsum('akl,nalh->nakh', covariance, h)
        encoded = self.message(torch.cat((h, message), dim=-1)).mean(dim=(1, 2))
        return self.output(encoded)


@contextmanager
def historical_influence(model, encoder, features, normalized_covariance, *, enabled=True):
    """Temporarily inject into a single table's historical prefix, all views.

    model is the actual TabICL torch module (loaded.model_), not sklearn wrapper.
    The caller MUST pack histories first in precisely features' row order;
    support then query follow. B may batch views of this SAME table, not tables
    with different relations. The encoder is an independently owned module,
    which the caller adds explicitly to its optimizer and saved checkpoint.

    Recompute is inside row_interactor; this outer hook is not installed on
    its checkpointed children. Encode afresh on EACH whole-module output, so
    serial axis/view backward never reuses a previous view's freed graph.
    Keep this scope active through any enclosing forward/backward, and do not
    switch table/relations concurrently on the same model. Features/covariance
    are frozen GP data; this prototype does not backpropagate through GP fits.

    Official forward_with_cache is blocked in both store/use modes. Even a
    newly built repr/KV cache requires a separately reviewed integration; none
    is implicitly accepted here. enabled=False makes no model modifications.
    """
    if not enabled:
        yield
        return
    assert not features.requires_grad and not normalized_covariance.requires_grad
    assert features.ndim == 4
    n_history = features.shape[0]
    # Nested scopes would double-add or bind different tables to one model.
    assert not getattr(model, '_history_influence_active', False)
    model._history_influence_active = True
    had_local_cache_method = 'forward_with_cache' in model.__dict__
    local_cache_method = model.__dict__.get('forward_with_cache')

    def forbid_cache(*args, **kwargs):
        raise RuntimeError('Influence side path requires uncached full-table forward; repr/KV cache paths are disabled')

    def inject(module, args, output):
        assert torch.is_tensor(output) and output.ndim == 3
        assert output.shape[1] >= n_history and output.shape[2] == encoder.config['output_dim']
        if not n_history:
            return output
        residual = encoder(features, normalized_covariance).to(dtype=output.dtype, device=output.device)
        # Non-inplace addition; all support/query row values pass unchanged.
        return torch.cat((output[:, :n_history]+residual.unsqueeze(0), output[:, n_history:]), dim=1)

    model.forward_with_cache = forbid_cache
    handle = model.row_interactor.register_forward_hook(inject)
    try:
        yield
    finally:
        handle.remove()
        if had_local_cache_method:
            model.forward_with_cache = local_cache_method
        else:
            del model.forward_with_cache
        del model._history_influence_active


class ColumnInfluenceEncoder(nn.Module):
    """Encode S influence per AP, then bind ordered APs to native column groups.

    Input features and full covariance have the same units/axes as above.
    Anchor pooling is permitted; AP pooling is NOT performed. column_slots
    [B,D] indexes the graph's AP axis in this exact batch's transformed column
    order, with -1 reserved for the original physical-floor feature. The
    consumer obtains this map after the actual constant-column mask and each
    actual feature shuffle, not from sorted AP IDs or a guessed view counter.

    Each group has the installed model's ordered members (g+2**i) mod D.
    Concatenating their separate relation embeddings before projection keeps
    member roles. Permuting relations alone is no longer an architectural
    invariance; no claim of an injective/sufficient compression is made.
    Shared weights need no learned AP-ID lookup, so unseen registered APs have
    the same interface. This provides a conditioning path, not a guarantee
    that training will learn to use it or recover a physical calibration.

    Required moments[N_H,AP,3] are ordered (d,u,b): d=A_j e/s,
    u=A_j V A_j^T/s**2, b=diag(B_j A_j^T)/s**2 with B_j=omega_j**2 C_j.
    These already computed quantities bypass anchor averaging and join each
    AP embedding before native ordered grouping. u-b is only the source's
    variance update, not a complete variance or total target uncertainty.
    This preserves known GP algebra as direct inputs; it is not a new physical
    calibration model, sufficient statistic claim or performance guarantee.
    """
    def __init__(self, feature_dim, output_dim, group_size=3, hidden_dim=32):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, output_dim=output_dim,
                           group_size=group_size, hidden_dim=hidden_dim)
        with torch.random.fork_rng(devices=[]):
            self.token = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
            self.message = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.GELU())
            self.output = nn.Linear(group_size*(hidden_dim+3), output_dim, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, features, normalized_covariance, column_slots, moments):
        n, a, k, f = features.shape
        assert f == self.config['feature_dim']
        assert normalized_covariance.shape == (a, k, k)
        assert moments.shape == (n, a, 3)
        assert column_slots.ndim == 2 and column_slots.shape[1] > 0
        assert features.device == normalized_covariance.device == moments.device == self.output.weight.device
        slots = column_slots.to(device=features.device, dtype=torch.long)
        assert bool(((slots >= -1) & (slots < a)).all())
        b, d = slots.shape
        if not n or not a or not k:
            return self.output.weight.new_zeros((b, n, d, self.config['output_dim']))
        h = self.token(features.to(dtype=self.output.weight.dtype))
        message = torch.einsum('akl,nalh->nakh', normalized_covariance.to(h.dtype), h)
        per_ap = self.message(torch.cat((h, message), dim=-1)).mean(dim=2)
        per_ap = torch.cat((per_ap, moments.to(dtype=per_ap.dtype)), dim=-1)
        # One fixed zero slot for FLOOR. Never assign it an AP relation.
        per_ap = torch.cat((per_ap, per_ap.new_zeros((n, 1, per_ap.shape[-1]))), dim=1)
        slots = torch.where(slots < 0, a, slots)
        shifts = 2 ** torch.arange(self.config['group_size'], device=features.device)
        group_index = (torch.arange(d, device=features.device)[:, None] + shifts) % d
        members = slots[:, group_index]  # [view, group, ordered member]
        grouped = per_ap[:, members].permute(1, 0, 2, 3, 4)
        return self.output(grouped.flatten(-2))


@contextmanager
def column_influence(model, encoder, features, normalized_covariance,
                     column_slots, moments, *, enabled=True):
    """Bind one actual view batch, adding H-only residuals before row attention.

    Only the inspected same-grouping, uncached complete-table path is handled.
    Caller owns the column map and must keep the scope through backward. Raw
    col_embedder output is retained; CLS, S and Q receive zero direct residual.
    Later original attention may change every prediction. The encoder runs
    afresh per call: no view reuses a previous backward's freed graph.
    Required d/u/b moments remain AP-bound and bypass anchor pooling.
    """
    if not enabled:
        yield
        return
    col = model.col_embedder
    assert col.feature_group in (True, 'same')
    assert col.feature_group_size == encoder.config['group_size']
    assert col.embed_dim == encoder.config['output_dim']
    assert not features.requires_grad and not normalized_covariance.requires_grad
    assert not moments.requires_grad
    assert not getattr(model, '_history_influence_active', False)
    n_history = features.shape[0]
    b, d = column_slots.shape
    c = col.reserve_cls_tokens
    model._history_influence_active = True
    had_local = 'forward_with_cache' in model.__dict__
    previous = model.__dict__.get('forward_with_cache')

    def forbid_cache(*args, **kwargs):
        raise RuntimeError('Column influence requires the uncached complete-table path')

    def inject(module, args, output):
        assert output.ndim == 4 and output.shape[0] == b
        assert output.shape[1] >= n_history and output.shape[2] == c+d
        assert output.shape[3] == encoder.config['output_dim']
        residual = encoder(features, normalized_covariance, column_slots, moments).to(
            dtype=output.dtype, device=output.device)
        history = torch.cat((output[:, :n_history, :c],
                             output[:, :n_history, c:]+residual), dim=2)
        return torch.cat((history, output[:, n_history:]), dim=1)

    model.forward_with_cache = forbid_cache
    handle = col.register_forward_hook(inject)
    try:
        yield
    finally:
        handle.remove()
        if had_local:
            model.forward_with_cache = previous
        else:
            del model.forward_with_cache
        del model._history_influence_active
