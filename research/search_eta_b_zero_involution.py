"""Search the eta_B=0 boundary in an involutive row-swap family.

Let R swap router rows 0 and 1 and fix row 2, and set
M=A^{-1} R A.  Then A M=R A, M 1=1, and M^2=I.  Consequently the local
response equality for the second swapped row follows from the first.  A valid
counterexample need only satisfy the first swapped-row equality and the fixed
third-row equality, while A and A M remain interior and full rank and M is
not a basis permutation.

This is an adversarial search utility, not manuscript evidence.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.optimize import least_squares


R = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def softmax_rows(values: np.ndarray) -> np.ndarray:
    logits = np.concatenate([values.reshape(3, 2), np.zeros((3, 1))], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def covariance(a: np.ndarray) -> np.ndarray:
    return np.diag(a) - np.outer(a, a)


def tangent_entries(matrix: np.ndarray) -> np.ndarray:
    block = matrix[:2, :2]
    return np.asarray([block[0, 0], block[0, 1], block[1, 1]])


def permutation_distance(matrix: np.ndarray) -> float:
    answer = np.inf
    for order in itertools.permutations(range(3)):
        candidate = np.zeros((3, 3))
        candidate[np.arange(3), np.asarray(order)] = 1.0
        answer = min(answer, np.linalg.norm(matrix - candidate))
    return float(answer)


def residual(values: np.ndarray) -> np.ndarray:
    router = softmax_rows(values)
    try:
        gauge = np.linalg.solve(router, R @ router)
    except np.linalg.LinAlgError:
        return np.ones(6) * 1e3
    transformed = R @ router
    differences = []
    for index in (0, 2):
        native = covariance(router[index])
        changed = covariance(transformed[index])
        differences.extend(
            tangent_entries(changed @ changed - gauge.T @ native @ native @ gauge)
        )
    return np.asarray(differences)


def search(seed: int, starts: int, max_nfev: int) -> None:
    rng = np.random.default_rng(seed)
    best = None
    for start in range(starts):
        initial = rng.normal(scale=1.5, size=6)
        fit = least_squares(
            residual,
            initial,
            max_nfev=max_nfev,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        router = softmax_rows(fit.x)
        try:
            gauge = np.linalg.solve(router, R @ router)
        except np.linalg.LinAlgError:
            continue
        transformed = router @ gauge
        all_errors = []
        for index in range(3):
            native = covariance(router[index])
            changed = covariance(transformed[index])
            all_errors.append(
                np.linalg.norm(
                    changed @ changed - gauge.T @ native @ native @ gauge
                )
            )
        diagnostics = {
            "start": start,
            "search_residual": float(np.linalg.norm(fit.fun)),
            "all_row_max_residual": float(max(all_errors)),
            "min_A": float(router.min()),
            "min_A_prime": float(transformed.min()),
            "sigma_A": float(np.linalg.svd(router, compute_uv=False)[-1]),
            "sigma_M": float(np.linalg.svd(gauge, compute_uv=False)[-1]),
            "permutation_distance": permutation_distance(gauge),
            "M_squared_error": float(np.linalg.norm(gauge @ gauge - np.eye(3))),
            "product_error": float(
                np.linalg.norm(router - transformed @ np.linalg.inv(gauge))
            ),
        }
        item = (diagnostics["all_row_max_residual"], diagnostics, router, gauge)
        if best is None or item[0] < best[0]:
            best = item
        if (
            diagnostics["all_row_max_residual"] < 1e-10
            and diagnostics["min_A"] > 1e-5
            and diagnostics["min_A_prime"] > 1e-5
            and diagnostics["sigma_A"] > 1e-5
            and diagnostics["sigma_M"] > 1e-5
            and diagnostics["permutation_distance"] > 1e-3
        ):
            np.set_printoptions(precision=17, suppress=True)
            print("COUNTEREXAMPLE")
            print(diagnostics)
            print("A=", repr(router))
            print("M=", repr(gauge))
            return
    print("NO_COUNTEREXAMPLE")
    if best is not None:
        _, diagnostics, router, gauge = best
        np.set_printoptions(precision=17, suppress=True)
        print(diagnostics)
        print("A=", repr(router))
        print("M=", repr(gauge))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--starts", type=int, default=500)
    parser.add_argument("--max-nfev", type=int, default=5000)
    args = parser.parse_args()
    search(args.seed, args.starts, args.max_nfev)


if __name__ == "__main__":
    main()
