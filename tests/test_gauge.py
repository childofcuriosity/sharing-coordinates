from src.gauge import exact_argmax_flip_example, run_gauge_counterfactual


def test_gauge_changes_router_but_not_effective_weights():
    result = run_gauge_counterfactual(seed=2)
    assert result["effective_max_abs_error"] < 1e-5
    assert result["post_step_effective_relative_difference"] > 1e-6


def test_exact_example_flips_an_argmax():
    result = exact_argmax_flip_example()
    assert result["effective_max_abs_error"] < 1e-10
    assert result["argmax_before"] != result["argmax_after"]

