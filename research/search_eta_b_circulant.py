"""Independent exact-equation search for a cyclic eta_B=0 witness.

This is a research diagnostic.  If the base row ``p`` solves the three
matrix equations below, cyclic equivariance verifies all three rows of the
circulant router, rather than silently checking only one layer.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import least_squares


def cov(p: np.ndarray) -> np.ndarray:
    return np.diag(p) - np.outer(p, p)


def circ(first: np.ndarray) -> np.ndarray:
    # Each successive row is a right cyclic shift.
    return np.stack([np.roll(first, i) for i in range(3)])


def sym3(x: np.ndarray) -> np.ndarray:
    return np.asarray([x[0, 0], x[1, 0], x[1, 1]])


def residual(v: np.ndarray, target: float) -> np.ndarray:
    p = np.asarray([v[0], v[1], 1.0 - v[0] - v[1]])
    first = np.asarray([v[2], v[3], 1.0 - v[2] - v[3]])
    m = circ(first)
    q = m.T @ p
    e = cov(q) @ cov(q) - m.T @ cov(p) @ cov(p) @ m
    return np.r_[sym3(e), first[1] - target]


def permutation_distance(m: np.ndarray) -> float:
    out = np.inf
    for order in itertools.permutations(range(3)):
        p = np.zeros((3, 3))
        p[np.arange(3), order] = 1
        out = min(out, np.linalg.norm(m - p))
    return float(out)


def main() -> None:
    rng = np.random.default_rng(20260823)
    best = None
    for target in np.linspace(-1.5, 1.5, 61):
        for _ in range(100):
            v0 = rng.normal(size=4)
            v0[:2] = rng.dirichlet(np.ones(3))[:2]
            v0[2:] = rng.normal(scale=0.8, size=2)
            ans = least_squares(
                residual, v0, args=(float(target),), max_nfev=10000,
                xtol=1e-14, ftol=1e-14, gtol=1e-14,
            )
            p = np.r_[ans.x[:2], 1 - ans.x[:2].sum()]
            m = circ(np.r_[ans.x[2:], 1 - ans.x[2:].sum()])
            q = m.T @ p
            a = circ(p)
            ap = a @ m
            all_res = []
            for pi, qi in zip(a, ap):
                all_res.append(cov(qi) @ cov(qi) - m.T @ cov(pi) @ cov(pi) @ m)
            diag = {
                "res": float(np.linalg.norm(all_res)),
                "min_A": float(a.min()),
                "min_Ap": float(ap.min()),
                "sA": float(np.linalg.svd(a, compute_uv=False)[-1]),
                "sM": float(np.linalg.svd(m, compute_uv=False)[-1]),
                "pdist": permutation_distance(m),
            }
            if best is None or diag["res"] < best[0]["res"]:
                best = (diag, p, q, m)
            if (
                diag["res"] < 1e-9 and diag["min_A"] > 1e-6
                and diag["min_Ap"] > 1e-6 and diag["sA"] > 1e-6
                and diag["sM"] > 1e-6 and diag["pdist"] > 1e-4
            ):
                print("COUNTEREXAMPLE", diag)
                print("p", repr(p))
                print("q", repr(q))
                print("M", repr(m))
                return
    print("NO_COUNTEREXAMPLE")
    print(best)


if __name__ == "__main__":
    main()
