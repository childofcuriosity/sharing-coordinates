"""Search the exact diagonal-scaling reformulation of the eta_B=0 gap.

For one positive router row, response equality is equivalent to

    T = diag(p) M diag(q)^{-1}

being a column-stochastic affine Euclidean isometry: its restriction to
``1^perp`` is orthogonal.  Every such matrix has the form

    T = O + t 1^T,

where O is orthogonal with O1=1 and t^T1=0.  A second row with ratio
``r=p'/p`` is compatible exactly when

    T_r = diag(r) T diag(T^T r)^{-1}

has the same property.  This script solves the six square polynomial/rational
conditions for two additional scalings.  It is reconnaissance only: numerical
roots are never treated as proofs and must be converted to exact arithmetic.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.optimize import linprog, root


ONES = np.ones(3)
U = np.ones((3, 3)) / 3.0
PI = np.eye(3) - U
SKEW = np.array(
    [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
) / np.sqrt(3.0)
E = np.array(
    [[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
     [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
     [0.0, -2.0 / np.sqrt(6.0)]]
)
PERMUTATIONS = []
for permutation in itertools.permutations(range(3)):
    candidate = np.zeros((3, 3))
    candidate[np.arange(3), permutation] = 1.0
    PERMUTATIONS.append(candidate)


def affine_isometry(
    theta: float,
    first_translation: float,
    second_translation: float,
    reflection: bool = False,
) -> np.ndarray:
    if reflection:
        tangent = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [np.sin(theta), -np.cos(theta)],
            ]
        )
        rotation = U + E @ tangent @ E.T
    else:
        rotation = U + np.cos(theta) * PI + np.sin(theta) * SKEW
    translation = np.array(
        [first_translation, second_translation, -first_translation - second_translation]
    )
    return rotation + np.outer(translation, ONES)


def scaled_isometry_residual(matrix: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    column_sums = matrix.T @ ratio
    scaled = np.diag(ratio) @ matrix @ np.diag(1.0 / column_sums)
    gram = E.T @ (scaled.T @ scaled) @ E - np.eye(2)
    return np.array([gram[0, 0], gram[0, 1], gram[1, 1]])


def positive_chart_margin(matrix: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Maximize a common strict-positivity margin for q and Tq."""

    # Variables are q_1,q_2,q_3,epsilon.  The six inequalities require
    # q_i>=epsilon and (Tq)_i>=epsilon.
    objective = np.array([0.0, 0.0, 0.0, -1.0])
    inequalities = []
    bounds = []
    for index in range(3):
        row = np.zeros(4)
        row[index] = -1.0
        row[3] = 1.0
        inequalities.append(row)
        bounds.append(0.0)
    for index in range(3):
        row = np.zeros(4)
        row[:3] = -matrix[index]
        row[3] = 1.0
        inequalities.append(row)
        bounds.append(0.0)
    result = linprog(
        objective,
        A_ub=np.asarray(inequalities),
        b_ub=np.asarray(bounds),
        A_eq=np.array([[1.0, 1.0, 1.0, 0.0]]),
        b_eq=np.array([1.0]),
        bounds=[(None, None), (None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not result.success:
        return float("-inf"), None
    return float(result.x[3]), result.x[:3]


def equations(
    z: np.ndarray, fixed_first_ratio: float, reflection: bool = False
) -> np.ndarray:
    theta, ta, tb, second_ratio, third_a, third_b = z
    matrix = affine_isometry(theta, ta, tb, reflection=reflection)
    ratio_two = np.array([fixed_first_ratio, second_ratio, 1.0])
    ratio_three = np.array([third_a, third_b, 1.0])
    return np.concatenate(
        (
            scaled_isometry_residual(matrix, ratio_two),
            scaled_isometry_residual(matrix, ratio_three),
        )
    )


def equations_fixed_translation(
    z: np.ndarray, fixed_translation: float, reflection: bool = False
) -> np.ndarray:
    theta, tb, second_a, second_b, third_a, third_b = z
    matrix = affine_isometry(theta, fixed_translation, tb, reflection=reflection)
    ratio_two = np.array([second_a, second_b, 1.0])
    ratio_three = np.array([third_a, third_b, 1.0])
    return np.concatenate(
        (
            scaled_isometry_residual(matrix, ratio_two),
            scaled_isometry_residual(matrix, ratio_three),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--fixed-translation", type=float)
    parser.add_argument("--reflection", action="store_true")
    parser.add_argument(
        "--probe-fixed",
        nargs=3,
        type=float,
        metavar=("THETA", "T1", "T2"),
        help="hold T fixed and search for a second compatible positive scaling",
    )
    parser.add_argument("--starts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.probe_fixed is not None:
        matrix = affine_isometry(*args.probe_fixed, reflection=args.reflection)
        accepted_ratios = []
        for _ in range(args.starts):
            initial = np.exp(rng.normal(0.0, 1.0, size=2))
            result = root(
                lambda value: scaled_isometry_residual(
                    matrix, np.array([value[0], value[1], 1.0])
                )[:2],
                initial,
                method="hybr",
            )
            if not result.success:
                continue
            ratio = np.array([result.x[0], result.x[1], 1.0])
            column_scale = matrix.T @ ratio
            residual = float(
                np.max(np.abs(scaled_isometry_residual(matrix, ratio)))
            )
            if (
                residual < 1e-9
                and ratio.min() > 1e-6
                and column_scale.min() > 1e-6
                and np.max(np.abs(ratio - 1.0)) > 1e-5
                and not any(
                    np.max(np.abs(ratio - old)) < 1e-5
                    for old in accepted_ratios
                )
            ):
                accepted_ratios.append(ratio)
                print("ratio", repr(ratio), "column scale", repr(column_scale))
                print("residual", residual)
        print("distinct nonconstant positive scalings:", len(accepted_ratios))
        return

    accepted = []
    for _ in range(args.starts):
        if args.fixed_translation is None:
            initial = np.array(
                [
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(-0.3, 0.3),
                    rng.uniform(-0.3, 0.3),
                    np.exp(rng.normal(0.0, 0.7)),
                    np.exp(rng.normal(0.0, 0.7)),
                    np.exp(rng.normal(0.0, 0.7)),
                ]
            )
            result = root(
                equations,
                initial,
                args=(args.ratio, args.reflection),
                method="hybr",
            )
            theta, ta, tb, second_ratio, third_a, third_b = result.x
            ratios = np.array(
                [[1.0, 1.0, 1.0], [args.ratio, second_ratio, 1.0], [third_a, third_b, 1.0]]
            )
            residual = float(
                np.max(np.abs(equations(result.x, args.ratio, args.reflection)))
            )
        else:
            initial = np.array(
                [
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(-0.3, 0.3),
                    np.exp(rng.normal(0.0, 0.7)),
                    np.exp(rng.normal(0.0, 0.7)),
                    np.exp(rng.normal(0.0, 0.7)),
                    np.exp(rng.normal(0.0, 0.7)),
                ]
            )
            result = root(
                equations_fixed_translation,
                initial,
                args=(args.fixed_translation, args.reflection),
                method="hybr",
            )
            theta, tb, second_a, second_b, third_a, third_b = result.x
            ta = args.fixed_translation
            ratios = np.array(
                [[1.0, 1.0, 1.0], [second_a, second_b, 1.0], [third_a, third_b, 1.0]]
            )
            residual = float(
                np.max(
                    np.abs(
                        equations_fixed_translation(
                            result.x, args.fixed_translation, args.reflection
                        )
                    )
                )
            )
        if not result.success:
            continue
        matrix = affine_isometry(theta, ta, tb, reflection=args.reflection)
        column_scales = ratios @ matrix
        chart_margin, base_q = positive_chart_margin(matrix)
        permutation_distance = min(
            np.linalg.norm(matrix - permutation) for permutation in PERMUTATIONS
        )
        if (
            residual < 1e-9
            and chart_margin > 1e-5
            and ratios.min() > 1e-5
            and column_scales.min() > 1e-5
            and abs(np.linalg.det(ratios)) > 1e-5
            and np.max(np.abs(ratios[1:] - 1.0)) > 1e-4
            and permutation_distance > 1e-4
        ):
            signature = np.round(np.concatenate(([theta, ta, tb], ratios[1:].ravel())), 7)
            if any(np.max(np.abs(signature - old)) < 1e-5 for old in accepted):
                continue
            accepted.append(signature)
            print("residual", residual)
            print("theta,t", theta, ta, tb)
            print("T=\n", repr(matrix))
            print("ratios=\n", repr(ratios))
            print("T^T ratios=\n", repr(column_scales))
            print("det ratios", np.linalg.det(ratios))
            print("positive chart margin", chart_margin)
            print("base q,p", base_q, matrix @ base_q)
            print("nearest permutation distance", permutation_distance)
            print()
    print("distinct positive full-rank scaling triples:", len(accepted))


if __name__ == "__main__":
    main()
