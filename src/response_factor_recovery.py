"""Experimental reconstruction from product and tomographic components.

The caller supplies observations only. Exact recovery follows the diagonal
algebra reduction; PSD clipping in noisy arithmetic carries no stability claim.
"""

import math

import torch


def recover_factors(theta, row_basis, gram, local, *, eta_z=1.0, tau=1.0,
                    weight_seed=520000, weights=None, rank_tolerance=1e-12):
    l, d = theta.shape
    k = row_basis.shape[0]
    if row_basis.shape != (k, d) or gram.shape != (l, l) or local.shape != (l, k, k):
        raise ValueError('Incompatible observation shapes')
    if k < 2 or k > min(l, d):
        raise ValueError('This experimental reconstruction requires 2 <= K <= min(L,D)')
    if not all(math.isfinite(v) and v > 0 for v in (eta_z, tau, rank_tolerance)):
        raise ValueError('Rates, temperature and tolerance must be positive and finite')
    for value in (theta, row_basis, gram, local):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Observations must be finite')
        if value.dtype != theta.dtype or value.device != theta.device:
            raise ValueError('Observations must share dtype and device')
    eigenvalues, vectors = torch.linalg.eigh((gram + gram.T)/2)
    selected = eigenvalues[-k:]
    if float(selected[0]) <= rank_tolerance * max(1., float(selected[-1])):
        raise ValueError('Measured Gram lacks the required numerical rank')
    x = vectors[:, -k:] * selected.sqrt()
    c = torch.linalg.lstsq(x, theta @ row_basis.T).solution
    singular = torch.linalg.svdvals(c)
    if float(singular[-1]) <= rank_tolerance * max(1., float(singular[0])):
        raise ValueError('Pulled-back basis is numerically singular')
    inverse_c = torch.linalg.inv(c)
    h = inverse_c.T @ (local * (tau*tau/eta_z)) @ inverse_c
    h = (h + h.transpose(-1, -2))/2
    h_eigenvalues, h_vectors = torch.linalg.eigh(h)
    roots = (h_vectors * h_eigenvalues.clamp_min(0).sqrt().unsqueeze(-2)) @ h_vectors.transpose(-1, -2)
    diagonal_family = roots + x.unsqueeze(-1)*x.unsqueeze(-2)
    if weights is None:
        # CPU generation fixes weights independently of the execution device.
        generator = torch.Generator().manual_seed(weight_seed)
        weights = torch.randn(l, generator=generator, dtype=theta.dtype).to(theta.device)
    if weights.shape != (l,) or not bool(torch.isfinite(weights).all()):
        raise ValueError('Weights must be a finite vector of length L')
    combination = torch.einsum('i,ijk->jk', weights, diagonal_family)
    spectrum, rotation = torch.linalg.eigh(combination)
    gap = float(torch.diff(spectrum).min())
    if gap <= rank_tolerance * max(1., float(spectrum.abs().max())):
        raise ValueError('Combined diagonal spectrum is not numerically separated')
    rotated = rotation.T @ diagonal_family @ rotation
    a = torch.diagonal(rotated, dim1=-2, dim2=-1).clone()
    # Solve in K row-space coordinates, avoiding a large least-squares RHS.
    b = torch.linalg.lstsq(a, theta @ row_basis.T).solution @ row_basis
    offdiagonal = rotated - torch.diag_embed(a)
    diagnostics = {
        'combined_eigenvalue_gap': gap,
        'gram_min_eigenvalue': float(eigenvalues[0]),
        'pullback_min_eigenvalue': float(h_eigenvalues.min()),
        'psd_clipped_negative_norm': float(h_eigenvalues.clamp_max(0).norm()),
        'joint_diagonalization_relative_residual': float(offdiagonal.norm()/diagonal_family.norm()),
        'product_relative_residual': float((a@b-theta).norm()/theta.norm()),
        'router_min_entry': float(a.min()),
        'router_row_sum_max_error': float((a.sum(-1)-1).abs().max()),
    }
    return a, b, diagnostics
