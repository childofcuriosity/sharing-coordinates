"""Adversarial numerical checks for the quantitative response proof.

These tests do not prove the theorem.  They independently materialize the
full response operator at small dimensions and check each inequality used in
the proof, including the gauge orientation and the local rounding branch.
"""

import itertools
import math

import pytest
import torch


DTYPE = torch.float64


def _covariance(a: torch.Tensor) -> torch.Tensor:
    return torch.diag(a) - torch.outer(a, a)


def _response_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eta_z: float,
    eta_b: float,
    tau: float,
) -> torch.Tensor:
    """Materialize the (LD)-square response matrix from its exact blocks."""

    depth, _ = a.shape
    dimension = b.shape[1]
    blocks = []
    for i in range(depth):
        row = []
        c_i = _covariance(a[i])
        h_i = b.T @ (c_i @ c_i) @ b
        for j in range(depth):
            block = eta_b * torch.dot(a[i], a[j]) * torch.eye(
                dimension, dtype=DTYPE
            )
            if i == j:
                block = block + (eta_z / tau**2) * h_i
            row.append(block)
        blocks.append(torch.cat(row, dim=1))
    return torch.cat(blocks, dim=0)


def _permutation_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    a_prime: torch.Tensor,
    b_prime: torch.Tensor,
) -> float:
    k = a.shape[1]
    best = math.inf
    for order in itertools.permutations(range(k)):
        p = torch.zeros((k, k), dtype=DTYPE)
        p[torch.arange(k), torch.tensor(order)] = 1.0
        value = torch.sqrt(
            torch.linalg.matrix_norm(a_prime - a @ p) ** 2
            + torch.linalg.matrix_norm(b_prime - p.T @ b) ** 2
        )
        best = min(best, float(value))
    return best


def _proof_quantities(
    a: torch.Tensor,
    b: torch.Tensor,
    m: torch.Tensor,
    *,
    eta_z: float,
    eta_b: float,
    tau: float,
):
    a_prime = a @ m
    b_prime = torch.linalg.solve(m, b)
    depth, k = a.shape

    alpha = float(torch.minimum(a.min(), a_prime.min()))
    s_a = float(
        torch.minimum(torch.linalg.svdvals(a).min(), torch.linalg.svdvals(a_prime).min())
    )
    s_b = float(
        torch.minimum(torch.linalg.svdvals(b).min(), torch.linalg.svdvals(b_prime).min())
    )
    l_a = float(torch.maximum(torch.linalg.matrix_norm(a), torch.linalg.matrix_norm(a_prime)))
    l_b = float(torch.maximum(torch.linalg.matrix_norm(b), torch.linalg.matrix_norm(b_prime)))
    lambda_z = eta_z / tau**2

    l_m = max(1.0, l_a / s_a)
    c_g = depth / (eta_b * s_a**2)
    c_h = 2.0 / (lambda_z * s_b**2)
    mu_star = alpha * (1.0 + 1.0 / (4.0 * l_m**2))
    c_j = l_m**2 * (c_g + c_h) / mu_star
    c_d = math.sqrt(depth * k) * c_j / s_a
    c_row = math.sqrt(k * c_d**2 + (c_g + k * c_d**2) ** 2)
    delta_0 = min(1.0, 1.0 / (2.0 * c_g), 1.0 / (2.0 * math.sqrt(k) * c_row))
    c_local = math.sqrt(l_a**2 + (l_m * l_b) ** 2) * math.sqrt(k) * c_row
    c_global = max(
        c_local,
        2.0 * math.sqrt(l_a**2 + l_b**2) / delta_0,
    )

    response = _response_matrix(a, b, eta_z=eta_z, eta_b=eta_b, tau=tau)
    response_prime = _response_matrix(
        a_prime, b_prime, eta_z=eta_z, eta_b=eta_b, tau=tau
    )
    delta = float(torch.linalg.matrix_norm(response - response_prime, ord=2))

    return {
        "a_prime": a_prime,
        "b_prime": b_prime,
        "alpha": alpha,
        "s_a": s_a,
        "s_b": s_b,
        "l_a": l_a,
        "l_b": l_b,
        "lambda_z": lambda_z,
        "l_m": l_m,
        "c_g": c_g,
        "c_h": c_h,
        "c_j": c_j,
        "c_d": c_d,
        "c_row": c_row,
        "delta_0": delta_0,
        "c_global": c_global,
        "delta": delta,
    }


def _assert_proof_chain(a, b, m, *, eta_z=0.7, eta_b=1.3, tau=0.8):
    q = _proof_quantities(a, b, m, eta_z=eta_z, eta_b=eta_b, tau=tau)
    a_prime = q["a_prime"]
    b_prime = q["b_prime"]
    delta = q["delta"]

    # Exact factor gauge direction and the first almost-orthogonality bound.
    assert torch.allclose(a @ b, a_prime @ b_prime, atol=2e-12, rtol=2e-12)
    assert torch.allclose(torch.linalg.solve(m, b), b_prime)
    gram_error = float(torch.linalg.matrix_norm(torch.eye(m.shape[0]) - m @ m.T, ord=2))
    assert gram_error <= q["c_g"] * delta * (1.0 + 2e-6) + 1e-13

    right_inverse = b.T @ torch.linalg.inv(b @ b.T)
    assert torch.allclose(
        b @ right_inverse,
        torch.eye(b.shape[0], dtype=DTYPE),
        atol=2e-12,
        rtol=2e-12,
    )
    assert float(torch.linalg.matrix_norm(right_inverse, ord=2)) <= 1.0 / q["s_b"] + 2e-12

    # Check the covariance-square pullback, Sylvester linearization, and
    # approximate diagonal algebra separately for every router row.
    for i in range(a.shape[0]):
        c_i = _covariance(a[i])
        c_prime = _covariance(a_prime[i])
        h_i = b.T @ c_i @ c_i @ b
        h_prime = b_prime.T @ c_prime @ c_prime @ b_prime
        raw_square = c_i @ c_i - torch.linalg.solve(
            m.T, (c_prime @ c_prime) @ torch.linalg.inv(m)
        )
        assert float(torch.linalg.matrix_norm(raw_square, ord=2)) <= q["c_h"] * delta * (1.0 + 2e-6) + 1e-13

        y_i = m.T @ c_i @ m
        square_error = float(torch.linalg.matrix_norm(y_i @ y_i - c_prime @ c_prime, ord=2))
        square_bound = q["l_m"] ** 2 * (q["c_g"] + q["c_h"]) * delta
        assert square_error <= square_bound * (1.0 + 2e-6) + 1e-13

        linear_error = float(torch.linalg.matrix_norm(y_i - c_prime, ord=2))
        assert linear_error <= q["c_j"] * delta * (1.0 + 2e-6) + 1e-13

        diagonal_error = m.T @ torch.diag(a[i]) @ m - torch.diag(a_prime[i])
        assert torch.allclose(diagonal_error, y_i - c_prime, atol=2e-12, rtol=2e-12)

        # Independently verify the diagonal-block scalar/PSD separation used
        # to obtain H_i-H_i'.
        t_i = eta_b * (torch.dot(a[i], a[i]) - torch.dot(a_prime[i], a_prime[i]))
        e_i = t_i * torch.eye(b.shape[1], dtype=DTYPE) + q["lambda_z"] * (h_i - h_prime)
        assert float(abs(t_i)) <= float(torch.linalg.matrix_norm(e_i, ord=2)) + 2e-12

    for j in range(m.shape[0]):
        e_jj = torch.zeros_like(m)
        e_jj[j, j] = 1.0
        algebra = m.T @ e_jj @ m
        off = algebra - torch.diag(torch.diag(algebra))
        assert float(torch.linalg.matrix_norm(off)) <= q["c_d"] * delta * (1.0 + 2e-6) + 1e-13

    lhs = _permutation_distance(a, b, a_prime, b_prime)
    assert lhs <= q["c_global"] * delta * (1.0 + 2e-6) + 1e-12

    if delta <= q["delta_0"]:
        rows = []
        for row in m:
            index = int(torch.argmax(row.abs()))
            signed = torch.zeros_like(row)
            signed[index] = torch.sign(row[index])
            rows.append(signed)
            assert float(torch.linalg.vector_norm(row - signed)) <= q["c_row"] * delta * (1.0 + 2e-6) + 1e-12
        signed_matrix = torch.stack(rows)
        assert len(set(torch.argmax(row.abs()).item() for row in signed_matrix)) == m.shape[0]
        assert torch.all(signed_matrix.sum(dim=1) == 1.0)


def test_quantitative_proof_chain_random_small_dimensions():
    generator = torch.Generator().manual_seed(20260823)
    for k in (2, 3, 4):
        depth = k + 1
        dimension = k + 1
        for _ in range(20):
            a = torch.softmax(torch.randn(depth, k, generator=generator, dtype=DTYPE), dim=1)
            b = torch.randn(k, dimension, generator=generator, dtype=DTYPE)
            direction = torch.randn(k, k, generator=generator, dtype=DTYPE)
            direction = direction - direction.mean(dim=1, keepdim=True)
            step = 0.04
            while True:
                m = torch.eye(k, dtype=DTYPE) + step * direction
                a_prime = a @ m
                if (
                    float(a_prime.min()) > 1e-4
                    and float(torch.linalg.cond(m)) < 25.0
                ):
                    break
                step *= 0.5
            _assert_proof_chain(a, b, m)


def test_quantitative_proof_chain_local_rounding_branch():
    a = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.55, 0.45]], dtype=DTYPE)
    b = torch.tensor([[1.0, -0.3, 0.2], [0.1, 0.9, -0.7]], dtype=DTYPE)
    direction = torch.tensor([[1.0, -1.0], [-0.4, 0.4]], dtype=DTYPE)
    for step in (1e-7, 1e-8, 1e-9):
        m = torch.eye(2, dtype=DTYPE) + step * direction
        quantities = _proof_quantities(a, b, m, eta_z=0.7, eta_b=1.3, tau=0.8)
        assert quantities["delta"] <= quantities["delta_0"]
        _assert_proof_chain(a, b, m)


def test_exact_response_permutation_and_nonpermutation_search():
    generator = torch.Generator().manual_seed(260823)
    eta_z, eta_b, tau = 0.7, 1.3, 1.2
    for k in (2, 3, 4):
        a = torch.softmax(torch.randn(k + 2, k, generator=generator, dtype=DTYPE), dim=1)
        b = torch.randn(k, k + 1, generator=generator, dtype=DTYPE)
        reference = _response_matrix(a, b, eta_z=eta_z, eta_b=eta_b, tau=tau)

        order = torch.arange(k - 1, -1, -1)
        p = torch.eye(k, dtype=DTYPE)[:, order]
        permuted = _response_matrix(
            a @ p, p.T @ b, eta_z=eta_z, eta_b=eta_b, tau=tau
        )
        assert torch.allclose(reference, permuted, atol=2e-12, rtol=2e-12)

        for _ in range(50):
            direction = torch.randn(k, k, generator=generator, dtype=DTYPE)
            direction -= direction.mean(dim=1, keepdim=True)
            step = 0.02
            m = torch.eye(k, dtype=DTYPE) + step * direction
            while float((a @ m).min()) <= 1e-4 or float(torch.linalg.cond(m)) >= 20.0:
                step *= 0.5
                m = torch.eye(k, dtype=DTYPE) + step * direction
            candidate = _response_matrix(
                a @ m,
                torch.linalg.solve(m, b),
                eta_z=eta_z,
                eta_b=eta_b,
                tau=tau,
            )
            response_gap = float(torch.linalg.matrix_norm(reference - candidate, ord=2))
            orbit_gap = _permutation_distance(a, b, a @ m, torch.linalg.solve(m, b))
            assert orbit_gap > 1e-7
            assert response_gap > 1e-9


def test_psd_square_root_and_shared_kernel_sylvester_steps():
    """Exercise the two matrix-analysis steps independently of rounding."""

    generator = torch.Generator().manual_seed(314159)
    for k in (2, 3, 4, 5):
        ones = torch.ones(k, dtype=DTYPE)
        projection = torch.eye(k, dtype=DTYPE) - torch.outer(ones, ones) / k
        eigenvalues, eigenvectors = torch.linalg.eigh(projection)
        complement = eigenvectors[:, eigenvalues > 0.5]

        for _ in range(20):
            a = torch.softmax(torch.randn(k, generator=generator, dtype=DTYPE), dim=0)
            c = _covariance(a)

            # A Householder-free construction of an orthogonal map fixing 1.
            random_core = torch.randn(k - 1, k - 1, generator=generator, dtype=DTYPE)
            core, _ = torch.linalg.qr(random_core)
            m = torch.outer(ones, ones) / k + complement @ core @ complement.T
            assert torch.allclose(m @ ones, ones, atol=3e-12, rtol=3e-12)
            assert torch.allclose(m @ m.T, torch.eye(k, dtype=DTYPE), atol=3e-12, rtol=3e-12)

            conjugate = m @ c @ m.T
            squared = conjugate @ conjugate
            values, vectors = torch.linalg.eigh(squared)
            # The exact zero eigenvalue can appear as a positive O(eps)
            # eigensolver residual; project it back to the known common kernel.
            values = torch.where(values.abs() < 1e-14, torch.zeros_like(values), values)
            psd_root = (vectors * values.clamp_min(0.0).sqrt()) @ vectors.T
            assert torch.allclose(psd_root, conjugate, atol=2e-11, rtol=2e-11)

            # Two different interior covariance congruences share exactly the
            # 1-kernel.  Restriction to its orthogonal complement makes the
            # Sylvester inverse Lipschitz with denominator gap(Y)+gap(C').
            a_prime = torch.softmax(
                torch.randn(k, generator=generator, dtype=DTYPE), dim=0
            )
            c_prime = _covariance(a_prime)
            positive = torch.randn(k, k, generator=generator, dtype=DTYPE)
            positive = positive - positive.mean(dim=1, keepdim=True)
            gauge = torch.eye(k, dtype=DTYPE) + 0.03 * positive
            y = gauge.T @ c @ gauge
            y_bar = complement.T @ y @ complement
            c_bar = complement.T @ c_prime @ complement
            d_bar = y_bar - c_bar
            right_side = y_bar @ y_bar - c_bar @ c_bar
            assert torch.allclose(
                y_bar @ d_bar + d_bar @ c_bar,
                right_side,
                atol=2e-11,
                rtol=2e-11,
            )
            gap_sum = float(
                torch.linalg.eigvalsh(y_bar).min()
                + torch.linalg.eigvalsh(c_bar).min()
            )
            assert gap_sum > 0.0
            assert float(torch.linalg.matrix_norm(d_bar, ord=2)) <= (
                float(torch.linalg.matrix_norm(right_side, ord=2)) / gap_sum
                + 2e-11
            )


def test_interior_margin_is_needed_for_a_uniform_linear_inverse():
    """A full-rank boundary sequence has distance/response gap of order 1/t.

    The orthogonal gauges remove the router-Gram response exactly.  Both
    simplex covariances are O(t) near one-hot rows, so their squared response
    is O(t^2), while the factor chart moves O(t).  Router and basis singular
    values and all norms remain uniformly bounded; only the entrywise
    interior margin tends to zero.
    """

    k = 3
    generator = torch.tensor(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
        dtype=DTYPE,
    ) / math.sqrt(3.0)
    identity = torch.eye(k, dtype=DTYPE)
    ones = torch.ones(k, k, dtype=DTYPE)
    b = identity.clone()
    ratios = []
    minima = []
    for t in (2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3):
        m = torch.matrix_exp(t * generator)
        a = t * ones + (1.0 - 3.0 * t) * identity
        a_prime = a @ m
        b_prime = m.T
        assert float(a_prime.min()) > 0.0
        assert float(torch.linalg.svdvals(a).min()) > 0.9
        assert float(torch.linalg.svdvals(a_prime).min()) > 0.9
        assert torch.allclose(a @ b, a_prime @ b_prime, atol=2e-12, rtol=2e-12)

        response = _response_matrix(a, b, eta_z=0.7, eta_b=1.3, tau=1.0)
        response_prime = _response_matrix(
            a_prime, b_prime, eta_z=0.7, eta_b=1.3, tau=1.0
        )
        delta = float(torch.linalg.matrix_norm(response - response_prime, ord=2))
        chart_distance = _permutation_distance(a, b, a_prime, b_prime)
        ratios.append(chart_distance / delta)
        minima.append(float(a_prime.min()))

    assert all(left > right for left, right in zip(minima, minima[1:]))
    assert all(left < right for left, right in zip(ratios, ratios[1:]))
    assert ratios[-1] > 2.5 * ratios[0]


def test_router_singular_margin_is_needed_for_uniform_stability():
    k = 3
    q = torch.tensor(
        [[2.0, -1.0, 2.0], [-1.0, 2.0, 2.0], [2.0, 2.0, -1.0]],
        dtype=DTYPE,
    ) / 3.0
    uniform = torch.ones(k, k, dtype=DTYPE) / k
    identity = torch.eye(k, dtype=DTYPE)
    b = identity.clone()
    ratios = []
    for epsilon in (0.08, 0.04, 0.02, 0.01, 0.005):
        a = uniform + epsilon * (identity - uniform)
        a_prime = a @ q
        b_prime = q.T
        assert float(a_prime.min()) > 0.0
        assert float(torch.linalg.svdvals(a).min()) == pytest.approx(epsilon)
        assert torch.allclose(a @ b, a_prime @ b_prime, atol=2e-12, rtol=2e-12)
        response = _response_matrix(a, b, eta_z=0.7, eta_b=1.3, tau=1.0)
        response_prime = _response_matrix(
            a_prime, b_prime, eta_z=0.7, eta_b=1.3, tau=1.0
        )
        delta = float(torch.linalg.matrix_norm(response - response_prime, ord=2))
        ratios.append(_permutation_distance(a, b, a_prime, b_prime) / delta)
    assert all(left < right for left, right in zip(ratios, ratios[1:]))
    assert ratios[-1] > 5.0 * ratios[0]


def test_basis_singular_margin_is_needed_for_uniform_stability():
    k = 3
    q = torch.tensor(
        [[2.0, -1.0, 2.0], [-1.0, 2.0, 2.0], [2.0, 2.0, -1.0]],
        dtype=DTYPE,
    ) / 3.0
    a = 0.3 * torch.ones(k, k, dtype=DTYPE) + 0.1 * torch.eye(k, dtype=DTYPE)
    a_prime = a @ q
    collapsed = torch.ones(k, 1, dtype=DTYPE) @ torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=DTYPE
    )
    ratios = []
    for epsilon in (0.08, 0.04, 0.02, 0.01, 0.005):
        b = collapsed + epsilon * torch.eye(k, dtype=DTYPE)
        b_prime = q.T @ b
        assert float(torch.linalg.svdvals(b).min()) > 0.0
        assert torch.allclose(a @ b, a_prime @ b_prime, atol=2e-12, rtol=2e-12)
        response = _response_matrix(a, b, eta_z=0.7, eta_b=1.3, tau=1.0)
        response_prime = _response_matrix(
            a_prime, b_prime, eta_z=0.7, eta_b=1.3, tau=1.0
        )
        delta = float(torch.linalg.matrix_norm(response - response_prime, ord=2))
        ratios.append(_permutation_distance(a, b, a_prime, b_prime) / delta)
    assert all(left < right for left, right in zip(ratios, ratios[1:]))
    assert ratios[-1] > 20.0 * ratios[0]
