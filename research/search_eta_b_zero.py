"""Adversarial numerical search for an eta_B=0 response counterexample.

This script is a research diagnostic, not part of the manuscript's evidence.
For K=L=3 it searches for full-rank positive routers A and non-permutation
row-stochastic gauges M satisfying, row by row,

    J(M.T @ a)^2 = M.T @ J(a)^2 @ M.

That is exactly equality of the router-update response after eliminating a
full-row-rank basis when the basis block is frozen (eta_B=0).  Three selected
entries of M are pinned in each solve so that least-squares cannot simply
return the identity gauge.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.optimize import least_squares


def row_stochastic_from_free(values: np.ndarray, k: int) -> np.ndarray:
    first = values.reshape(k, k - 1)
    return np.concatenate([first, 1.0 - first.sum(axis=1, keepdims=True)], axis=1)


def covariance(a: np.ndarray) -> np.ndarray:
    return np.diag(a) - np.outer(a, a)


def independent_symmetric_entries(matrix: np.ndarray) -> np.ndarray:
    # A symmetric matrix killing 1 is determined by its leading (K-1)-square.
    block = matrix[:-1, :-1]
    return np.asarray(
        [block[i, j] for i in range(block.shape[0]) for j in range(i + 1)]
    )


def permutation_distance(matrix: np.ndarray) -> float:
    k = matrix.shape[0]
    best = np.inf
    for order in itertools.permutations(range(k)):
        permutation = np.zeros((k, k))
        permutation[np.arange(k), np.asarray(order)] = 1.0
        best = min(best, np.linalg.norm(matrix - permutation))
    return float(best)


def row_softmax(values: np.ndarray, k: int) -> np.ndarray:
    logits = np.concatenate(
        [values.reshape(k, k - 1), np.zeros((k, 1))], axis=1
    )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def equations(
    variables: np.ndarray,
    k: int,
    pinned_indices: tuple[int, ...],
    pinned_values: np.ndarray,
) -> np.ndarray:
    free_count = k * (k - 1)
    m_free = variables[:free_count]
    a_free = variables[free_count:]
    matrix = row_stochastic_from_free(m_free, k)
    router = row_stochastic_from_free(a_free, k)
    transformed = router @ matrix
    residuals: list[float] = []
    for a, a_prime in zip(router, transformed):
        jacobian = covariance(a)
        jacobian_prime = covariance(a_prime)
        difference = (
            jacobian_prime @ jacobian_prime
            - matrix.T @ (jacobian @ jacobian) @ matrix
        )
        residuals.extend(independent_symmetric_entries(difference))
    residuals.extend(m_free[list(pinned_indices)] - pinned_values)
    return np.asarray(residuals)


def search(seed: int, targets: int, starts: int, max_nfev: int) -> None:
    rng = np.random.default_rng(seed)
    k = 3
    free_count = k * (k - 1)
    pinned_indices = (1, 2, 4)  # M[0,1], M[1,0], M[2,0]
    best = None
    best_norm = np.inf
    for target_index in range(targets):
        # Small but definitely nonzero off-diagonal pins keep feasible A M
        # plausible while excluding every permutation in a neighbourhood.
        pinned_values = rng.uniform(-0.35, 0.35, size=len(pinned_indices))
        if np.linalg.norm(pinned_values) < 0.12:
            pinned_values[0] += 0.2
        for start_index in range(starts):
            base_matrix = np.eye(k) + rng.normal(scale=0.18, size=(k, k))
            base_matrix -= (
                (base_matrix.sum(axis=1, keepdims=True) - 1.0) / k
            )
            m_free = base_matrix[:, :-1].reshape(-1)
            m_free[list(pinned_indices)] = pinned_values
            router = rng.dirichlet(np.ones(k) * 2.0, size=k)
            initial = np.concatenate([m_free, router[:, :-1].reshape(-1)])
            result = least_squares(
                equations,
                initial,
                args=(k, pinned_indices, pinned_values),
                max_nfev=max_nfev,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
            )
            norm = float(np.linalg.norm(result.fun))
            matrix = row_stochastic_from_free(result.x[:free_count], k)
            router = row_stochastic_from_free(result.x[free_count:], k)
            transformed = router @ matrix
            diagnostics = {
                "target": target_index,
                "start": start_index,
                "residual": norm,
                "min_A": float(router.min()),
                "min_A_prime": float(transformed.min()),
                "sigma_A": float(np.linalg.svd(router, compute_uv=False)[-1]),
                "sigma_M": float(np.linalg.svd(matrix, compute_uv=False)[-1]),
                "perm_distance": permutation_distance(matrix),
            }
            if norm < best_norm:
                best_norm = norm
                best = (diagnostics, matrix.copy(), router.copy(), transformed.copy())
            if (
                norm < 1e-9
                and diagnostics["min_A"] > 1e-6
                and diagnostics["min_A_prime"] > 1e-6
                and diagnostics["sigma_A"] > 1e-5
                and diagnostics["sigma_M"] > 1e-5
                and diagnostics["perm_distance"] > 1e-3
            ):
                print("COUNTEREXAMPLE")
                print(diagnostics)
                print("M=", repr(matrix))
                print("A=", repr(router))
                print("A_prime=", repr(transformed))
                return
    print("NO_COUNTEREXAMPLE")
    if best is not None:
        diagnostics, matrix, router, transformed = best
        print(diagnostics)
        print("M=", repr(matrix))
        print("A=", repr(router))
        print("A_prime=", repr(transformed))


def positive_equations(
    variables: np.ndarray,
    k: int,
    diagonal_targets: np.ndarray,
    gauge_target: float,
) -> np.ndarray:
    free_count = k * (k - 1)
    matrix = row_softmax(variables[:free_count], k)
    router = row_softmax(variables[free_count:], k)
    transformed = router @ matrix
    residuals: list[float] = []
    for a, a_prime in zip(router, transformed):
        jacobian = covariance(a)
        jacobian_prime = covariance(a_prime)
        difference = (
            jacobian_prime @ jacobian_prime
            - matrix.T @ (jacobian @ jacobian) @ matrix
        )
        residuals.extend(independent_symmetric_entries(difference))
    # Two router entries discourage the repeated-row escape; one off-diagonal
    # gauge entry excludes every nearby permutation.
    residuals.extend(
        [
            router[0, 0] - diagonal_targets[0],
            router[1, 1] - diagonal_targets[1],
            matrix[0, 1] - gauge_target,
        ]
    )
    return np.asarray(residuals)


def search_positive(seed: int, targets: int, starts: int, max_nfev: int) -> None:
    rng = np.random.default_rng(seed)
    k = 3
    free_count = k * (k - 1)
    best = None
    best_norm = np.inf
    for target_index in range(targets):
        diagonal_targets = rng.uniform(0.45, 0.85, size=2)
        gauge_target = float(rng.uniform(0.08, 0.35))
        for start_index in range(starts):
            initial = rng.normal(scale=1.5, size=2 * free_count)
            result = least_squares(
                positive_equations,
                initial,
                args=(k, diagonal_targets, gauge_target),
                max_nfev=max_nfev,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
            )
            norm = float(np.linalg.norm(result.fun))
            matrix = row_softmax(result.x[:free_count], k)
            router = row_softmax(result.x[free_count:], k)
            transformed = router @ matrix
            diagnostics = {
                "target": target_index,
                "start": start_index,
                "residual": norm,
                "min_A": float(router.min()),
                "min_A_prime": float(transformed.min()),
                "sigma_A": float(np.linalg.svd(router, compute_uv=False)[-1]),
                "sigma_M": float(np.linalg.svd(matrix, compute_uv=False)[-1]),
                "perm_distance": permutation_distance(matrix),
                "router_constraints": diagonal_targets.tolist(),
                "gauge_constraint": gauge_target,
            }
            if norm < best_norm:
                best_norm = norm
                best = (diagnostics, matrix.copy(), router.copy(), transformed.copy())
            if (
                norm < 1e-9
                and diagnostics["sigma_A"] > 1e-5
                and diagnostics["sigma_M"] > 1e-5
                and diagnostics["perm_distance"] > 1e-3
            ):
                print("COUNTEREXAMPLE")
                print(diagnostics)
                print("M=", repr(matrix))
                print("A=", repr(router))
                print("A_prime=", repr(transformed))
                return
    print("NO_COUNTEREXAMPLE")
    if best is not None:
        diagnostics, matrix, router, transformed = best
        print(diagnostics)
        print("M=", repr(matrix))
        print("A=", repr(router))
        print("A_prime=", repr(transformed))


def invariant_target_equations(
    variables: np.ndarray,
    k: int,
    determinant_a_target: float,
    determinant_m_target: float,
    distance_target: float,
) -> np.ndarray:
    free_count = k * (k - 1)
    matrix = row_stochastic_from_free(variables[:free_count], k)
    router = row_softmax(variables[free_count:], k)
    transformed = router @ matrix
    residuals: list[float] = []
    for a, a_prime in zip(router, transformed):
        jacobian = covariance(a)
        jacobian_prime = covariance(a_prime)
        difference = (
            jacobian_prime @ jacobian_prime
            - matrix.T @ (jacobian @ jacobian) @ matrix
        )
        residuals.extend(independent_symmetric_entries(difference))
    residuals.extend(
        [
            np.linalg.det(router) - determinant_a_target,
            np.linalg.det(matrix) - determinant_m_target,
            np.linalg.norm(matrix - np.eye(k)) ** 2 - distance_target**2,
        ]
    )
    return np.asarray(residuals)


def search_invariant_targets(
    seed: int, targets: int, starts: int, max_nfev: int
) -> None:
    rng = np.random.default_rng(seed)
    k = 3
    free_count = k * (k - 1)
    best = None
    best_norm = np.inf
    for target_index in range(targets):
        determinant_a_target = float(rng.uniform(0.03, 0.35))
        determinant_m_target = float(
            rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 2.0)
        )
        distance_target = float(rng.uniform(0.25, 2.5))
        for start_index in range(starts):
            perturbation = rng.normal(scale=distance_target / 3.0, size=(k, k))
            perturbation -= perturbation.sum(axis=1, keepdims=True) / k
            matrix = np.eye(k) + perturbation
            router_logits = rng.normal(scale=1.2, size=free_count)
            initial = np.concatenate([matrix[:, :-1].reshape(-1), router_logits])
            result = least_squares(
                invariant_target_equations,
                initial,
                args=(
                    k,
                    determinant_a_target,
                    determinant_m_target,
                    distance_target,
                ),
                max_nfev=max_nfev,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
            )
            norm = float(np.linalg.norm(result.fun))
            matrix = row_stochastic_from_free(result.x[:free_count], k)
            router = row_softmax(result.x[free_count:], k)
            transformed = router @ matrix
            diagnostics = {
                "target": target_index,
                "start": start_index,
                "residual": norm,
                "min_A": float(router.min()),
                "min_A_prime": float(transformed.min()),
                "sigma_A": float(np.linalg.svd(router, compute_uv=False)[-1]),
                "sigma_M": float(np.linalg.svd(matrix, compute_uv=False)[-1]),
                "det_A": float(np.linalg.det(router)),
                "det_M": float(np.linalg.det(matrix)),
                "perm_distance": permutation_distance(matrix),
                "targets": [
                    determinant_a_target,
                    determinant_m_target,
                    distance_target,
                ],
            }
            if norm < best_norm:
                best_norm = norm
                best = (diagnostics, matrix.copy(), router.copy(), transformed.copy())
            if (
                norm < 1e-9
                and diagnostics["min_A_prime"] > 1e-6
                and diagnostics["sigma_A"] > 1e-5
                and diagnostics["sigma_M"] > 1e-5
                and diagnostics["perm_distance"] > 1e-3
            ):
                print("COUNTEREXAMPLE")
                print(diagnostics)
                print("M=", repr(matrix))
                print("A=", repr(router))
                print("A_prime=", repr(transformed))
                return
    print("NO_COUNTEREXAMPLE")
    if best is not None:
        diagnostics, matrix, router, transformed = best
        print(diagnostics)
        print("M=", repr(matrix))
        print("A=", repr(router))
        print("A_prime=", repr(transformed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--targets", type=int, default=40)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--max-nfev", type=int, default=5000)
    parser.add_argument(
        "--positive",
        action="store_true",
        help="parameterize both A and M as strictly positive row-stochastic matrices",
    )
    parser.add_argument(
        "--invariant-targets",
        action="store_true",
        help="pin det(A), det(M), and distance from identity instead of entries",
    )
    args = parser.parse_args()
    if args.invariant_targets:
        search_invariant_targets(args.seed, args.targets, args.starts, args.max_nfev)
    elif args.positive:
        search_positive(args.seed, args.targets, args.starts, args.max_nfev)
    else:
        search(args.seed, args.targets, args.starts, args.max_nfev)


if __name__ == "__main__":
    main()
