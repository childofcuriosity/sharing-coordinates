"""Square-system reconnaissance for the signed eta_B=0 compatibility gap.

This is deliberately separate from the least-squares searches in the release.
For K=L=3 it fixes one router coordinate and the first row of a row-stochastic
gauge, leaving nine variables for the nine independent polynomial response
equations.  ``scipy.optimize.root`` is used only to locate candidate algebraic
branches.  A printed small residual is not a certificate; any candidate must
subsequently be recovered and checked in exact arithmetic.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import root


def jacobian(probability: np.ndarray) -> np.ndarray:
    return np.diag(probability) - np.outer(probability, probability)


def unpack(z: np.ndarray, first_gauge_row: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    a, b = first_gauge_row
    gauge = np.array(
        [
            [a, b, 1.0 - a - b],
            [z[0], z[1], 1.0 - z[0] - z[1]],
            [z[2], z[3], 1.0 - z[2] - z[3]],
        ],
        dtype=float,
    )
    router = np.array(
        [
            [0.2, z[4], 0.8 - z[4]],
            [z[5], z[6], 1.0 - z[5] - z[6]],
            [z[7], z[8], 1.0 - z[7] - z[8]],
        ],
        dtype=float,
    )
    return gauge, router


def equations(z: np.ndarray, first_gauge_row: tuple[float, float]) -> np.ndarray:
    gauge, router = unpack(z, first_gauge_row)
    transformed = router @ gauge
    output = []
    for p, q in zip(router, transformed):
        jp = jacobian(p)
        jq = jacobian(q)
        residual = jq @ jq - gauge.T @ (jp @ jp) @ gauge
        output.extend((residual[0, 0], residual[0, 1], residual[1, 1]))
    return np.asarray(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-row", nargs=2, type=float, default=(-0.2, 0.6))
    parser.add_argument("--starts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    fixed = (float(args.first_row[0]), float(args.first_row[1]))
    rng = np.random.default_rng(args.seed)
    accepted = []
    for _ in range(args.starts):
        gauge_tail = rng.normal(loc=1.0 / 3.0, scale=0.65, size=4)
        rows = rng.dirichlet(np.full(3, 1.5), size=3)
        z0 = np.concatenate((gauge_tail, [rows[0, 1]], rows[1:, :2].ravel()))
        result = root(equations, z0, args=(fixed,), method="hybr")
        # Newton/Powell root finding only locates a possible branch; exact
        # certification is still mandatory.  Filter solely to expose plausible
        # branches.
        gauge, router = unpack(result.x, fixed)
        transformed = router @ gauge
        residual = float(np.max(np.abs(equations(result.x, fixed))))
        if (
            result.success
            and residual < 1e-10
            and min(router.min(), transformed.min()) > 1e-5
            and abs(np.linalg.det(router)) > 1e-5
            and abs(np.linalg.det(gauge)) > 1e-5
        ):
            signature = np.round(np.concatenate((gauge.ravel(), router.ravel())), 8)
            if not any(np.max(np.abs(signature - old[0])) < 1e-6 for old in accepted):
                accepted.append((signature, residual, gauge, router, transformed))
                print("residual", residual)
                print("det(M), det(A)", np.linalg.det(gauge), np.linalg.det(router))
                print("min(A), min(AM)", router.min(), transformed.min())
                print("M=\n", repr(gauge))
                print("A=\n", repr(router))
                print("AM=\n", repr(transformed))
                print()
    print("distinct interior candidates:", len(accepted))


if __name__ == "__main__":
    main()
