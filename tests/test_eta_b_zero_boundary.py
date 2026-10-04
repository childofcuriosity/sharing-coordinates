import itertools

import numpy as np


def _covariance(probabilities: np.ndarray) -> np.ndarray:
    return np.diag(probabilities) - np.outer(probabilities, probabilities)


def _local_residual(
    probabilities: np.ndarray,
    transformed: np.ndarray,
    gauge: np.ndarray,
) -> float:
    before = _covariance(probabilities)
    after = _covariance(transformed)
    difference = after @ after - gauge.T @ before @ before @ gauge
    return float(np.linalg.norm(difference))


def _tangent_basis(dimension: int) -> np.ndarray:
    projector = np.eye(dimension) - np.ones((dimension, dimension)) / dimension
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    return eigenvectors[:, eigenvalues > 0.5]


def _tangent_gram(matrix: np.ndarray) -> np.ndarray:
    basis = _tangent_basis(matrix.shape[0])
    return basis.T @ matrix.T @ matrix @ basis


def test_k2_sum_and_difference_identity_used_by_exact_proof():
    # After subtracting the two eta_B=0 equations, their simultaneous
    # vanishing would force y=(1-d)S/2.  Under that substitution their sum is
    # the displayed positive factor, which can vanish only at d=1.
    p1, p2 = 0.17, 0.73
    d = 0.61
    total = p1 + p2
    delta = p2 - p1
    y = (1.0 - d) * total / 2.0
    q1, q2 = y + d * p1, y + d * p2
    residual_sum = (
        q1 * (1.0 - q1)
        + q2 * (1.0 - q2)
        - d * (p1 * (1.0 - p1) + p2 * (1.0 - p2))
    )
    factorization = (1.0 - d) * (
        total * (2.0 - total) + d * delta**2
    ) / 2.0
    assert np.isclose(residual_sum, factorization, atol=1e-15, rtol=1e-15)
    assert factorization > 0.0


def test_nonnegative_gauge_reduction_has_simplex_column_geometry():
    probabilities = np.array([0.2, 0.3, 0.5])
    gauge = np.array(
        [[0.72, 0.18, 0.10], [0.12, 0.76, 0.12], [0.08, 0.22, 0.70]]
    )
    transformed = gauge.T @ probabilities
    scaled = (
        np.diag(probabilities)
        @ gauge
        @ np.diag(1.0 / transformed)
    )
    assert np.all(scaled >= 0.0)
    assert np.allclose(scaled.sum(axis=0), 1.0, atol=1e-15, rtol=1e-15)
    distances = [
        np.linalg.norm(scaled[:, i] - scaled[:, j])
        for i, j in itertools.combinations(range(3), 2)
    ]
    assert max(distances) < np.sqrt(2.0)
    assert _local_residual(probabilities, transformed, gauge) > 1e-4

    permutation = np.eye(3)[[1, 2, 0]]
    permuted = permutation.T @ probabilities
    perm_scaled = (
        np.diag(probabilities)
        @ permutation
        @ np.diag(1.0 / permuted)
    )
    perm_distances = [
        np.linalg.norm(perm_scaled[:, i] - perm_scaled[:, j])
        for i, j in itertools.combinations(range(3), 2)
    ]
    assert np.allclose(perm_distances, np.sqrt(2.0), atol=1e-15, rtol=1e-15)
    assert _local_residual(probabilities, permuted, permutation) < 1e-15


def test_one_row_cyclic_near_solution_fails_the_other_two_rows():
    router = np.array(
        [
            [1.0 / 6.0, 1.0 / 5.0, 19.0 / 30.0],
            [0.63314916, 0.20079942, 0.16605142],
            [0.12947217, 0.74011599, 0.13041184],
        ]
    )
    row_cycle = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    gauge = np.linalg.solve(router, row_cycle @ router)
    transformed = router @ gauge
    residuals = np.array(
        [
            _local_residual(before, after, gauge)
            for before, after in zip(router, transformed)
        ]
    )
    assert np.linalg.svd(router, compute_uv=False)[-1] > 0.46
    assert min(router.min(), transformed.min()) > 0.12
    assert np.any(gauge < 0.0)
    assert residuals[0] < 1e-8
    assert residuals[1] > 0.06
    assert residuals[2] > 0.05
    assert np.linalg.norm(residuals) > 0.08


def test_exact_diagonal_scaling_and_ratio_transport_identities():
    probability = np.array([0.30, 0.30, 0.40])
    gauge = np.array(
        [[1.10, -0.10, 0.0], [-0.10, 1.10, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = gauge.T @ probability
    scaled = (
        np.diag(probability)
        @ gauge
        @ np.diag(1.0 / transformed)
    )
    assert np.allclose(scaled.sum(axis=0), 1.0, atol=1e-15, rtol=1e-15)
    assert np.allclose(
        _covariance(probability) @ gauge,
        scaled @ _covariance(transformed),
        atol=1e-15,
        rtol=1e-15,
    )

    second = np.array([0.32, 0.28, 0.40])
    second_transformed = gauge.T @ second
    ratio = second / probability
    column_scale = scaled.T @ ratio
    scaled_second_direct = (
        np.diag(second)
        @ gauge
        @ np.diag(1.0 / second_transformed)
    )
    scaled_second_from_ratio = (
        np.diag(ratio)
        @ scaled
        @ np.diag(1.0 / column_scale)
    )
    assert np.all(column_scale > 0.0)
    assert np.allclose(
        second_transformed,
        np.diag(transformed) @ column_scale,
        atol=1e-15,
        rtol=1e-15,
    )
    assert np.allclose(
        scaled_second_direct,
        scaled_second_from_ratio,
        atol=1e-15,
        rtol=1e-15,
    )


def test_coordinate_translation_product_obstruction():
    translation = np.array([0.10, -0.05, -0.02, -0.03])
    matrix = np.eye(4) + np.outer(translation, np.ones(4))
    assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-15, rtol=1e-15)
    assert np.allclose(_tangent_gram(matrix), np.eye(3), atol=1e-14, rtol=1e-14)

    ratio = np.array([1.4, 0.8, 1.1, 0.9])
    column_scale = matrix.T @ ratio
    shift = translation @ ratio
    assert np.allclose(column_scale, ratio + shift, atol=1e-15, rtol=1e-15)
    assert shift > 0.0
    assert np.prod(column_scale) > np.prod(ratio)
    scaled = np.diag(ratio) @ matrix @ np.diag(1.0 / column_scale)
    assert not np.allclose(_tangent_gram(scaled), np.eye(3), atol=1e-8, rtol=1e-8)

    constant_ratio = np.full(4, 1.7)
    constant_scale = matrix.T @ constant_ratio
    constant_scaled = (
        np.diag(constant_ratio)
        @ matrix
        @ np.diag(1.0 / constant_scale)
    )
    assert np.allclose(constant_scaled, matrix, atol=1e-15, rtol=1e-15)


def test_k3_zero_translation_reflection_elimination_factor():
    x, y, z = 1.2, 0.9, 1.7
    ratio = np.array([x, y, z])
    normal = np.array([y - z, z - x, x - y])
    tangent_minus = np.diag(ratio) @ normal
    fixed_normal = np.diag(1.0 / ratio) @ normal
    fixed_normal -= fixed_normal.mean()
    cross = np.cross(tangent_minus, fixed_normal)
    factor = (
        -2.0
        * (x - y)
        * (x - z)
        * (y - z)
        * (x * y + x * z + y * z)
        / (3.0 * x * y * z)
    )
    assert np.allclose(cross, factor * np.ones(3), atol=1e-15, rtol=1e-14)
    assert abs(factor) > 1e-3

    repeated = np.array([1.2, 1.2, 0.7])
    repeated_normal = np.array(
        [
            repeated[1] - repeated[2],
            repeated[2] - repeated[0],
            repeated[0] - repeated[1],
        ]
    )
    reflection = np.eye(3) - 2.0 * np.outer(repeated_normal, repeated_normal) / (
        repeated_normal @ repeated_normal
    )
    transposition = np.eye(3)[[1, 0, 2]]
    assert np.allclose(reflection, transposition, atol=1e-15, rtol=1e-15)
