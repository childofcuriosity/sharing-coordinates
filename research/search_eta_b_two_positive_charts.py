"""Search eta_B=0 equations with both routers interior but a signed gauge.

Unlike ``search_eta_b_zero.py --positive``, this parameterizes A and A'
separately by row softmax and sets M=A^{-1}A'.  Thus M may have negative
entries while both charts remain strictly interior and the product identity
is exact by construction (with B'=M^{-1}B).
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from scipy.optimize import least_squares


def softmax_rows(v: np.ndarray) -> np.ndarray:
    z = np.c_[v.reshape(3, 2), np.zeros(3)]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cov(p: np.ndarray) -> np.ndarray:
    return np.diag(p) - np.outer(p, p)


def sym3(x: np.ndarray) -> list[float]:
    return [float(x[0, 0]), float(x[1, 0]), float(x[1, 1])]


def pdist(m: np.ndarray) -> float:
    ans = np.inf
    for order in itertools.permutations(range(3)):
        p = np.zeros((3, 3)); p[np.arange(3), order] = 1
        ans = min(ans, np.linalg.norm(m - p))
    return float(ans)


def residual(v: np.ndarray, targets: np.ndarray) -> np.ndarray:
    a = softmax_rows(v[:6])
    ap = softmax_rows(v[6:])
    try:
        m = np.linalg.solve(a, ap)
    except np.linalg.LinAlgError:
        return np.full(12, 1e3)
    out: list[float] = []
    for p, q in zip(a, ap):
        e = cov(q) @ cov(q) - m.T @ cov(p) @ cov(p) @ m
        out.extend(sym3(e))
    # Two chart entries prevent the repeated-row/rank-collapse escape; the
    # signed-gauge entry excludes all permutations near the requested target.
    out.extend([a[0, 0]-targets[0], ap[1, 1]-targets[1], m[0, 1]-targets[2]])
    return np.asarray(out)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260823
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng = np.random.default_rng(seed)
    best = None
    for t in range(trials):
        targets = np.r_[rng.uniform(.35,.8,2), rng.uniform(-1.0,.7)]
        if min(abs(targets[2]), abs(targets[2]-1)) < .08:
            targets[2] -= .2
        v0 = rng.normal(scale=1.7, size=12)
        ans = least_squares(residual, v0, args=(targets,), max_nfev=10000,
                            xtol=1e-13, ftol=1e-13, gtol=1e-13)
        a=softmax_rows(ans.x[:6]); ap=softmax_rows(ans.x[6:]); m=np.linalg.solve(a,ap)
        eq = residual(ans.x, targets)[:9]
        d={"trial":t,"res":float(np.linalg.norm(eq)),"full_res":float(np.linalg.norm(ans.fun)),
           "minA":float(a.min()),"minAp":float(ap.min()),
           "sA":float(np.linalg.svd(a,compute_uv=False)[-1]),
           "sAp":float(np.linalg.svd(ap,compute_uv=False)[-1]),
           "sM":float(np.linalg.svd(m,compute_uv=False)[-1]),"pdist":pdist(m),
           "targets":targets.tolist()}
        score=d["full_res"] + 1e-4/(d["sA"]+1e-12)+1e-4/(d["sAp"]+1e-12)
        if best is None or score < best[0]: best=(score,d,a,ap,m)
        if d["full_res"]<1e-9 and d["sA"]>1e-5 and d["sAp"]>1e-5 and d["pdist"]>1e-3:
            print("COUNTEREXAMPLE",d);print("A",repr(a));print("Ap",repr(ap));print("M",repr(m));return
    print("NO_COUNTEREXAMPLE")
    print(best)


if __name__ == "__main__": main()
