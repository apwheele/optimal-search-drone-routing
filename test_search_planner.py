import numpy as np

from search_planner import (
    generate_candidates,
    make_probability_surface,
    path_score,
    plan_from_candidates,
    solve_exact_small,
    validate_paths,
)


def test_surface_is_probability_distribution():
    _, _, probability = make_probability_surface(20)
    assert probability.shape == (20, 20)
    assert np.isclose(probability.sum(), 1.0)
    assert np.all(probability > 0)


def test_candidate_plan_is_valid():
    _, _, probability = make_probability_surface(12)
    candidates = generate_candidates(probability, length=8)
    result = plan_from_candidates(probability, candidates, drones=2, length=8, time_limit=5)
    validate_paths(result.paths, probability.shape, expected_length=8)
    assert np.isclose(result.score, path_score(probability, result.paths))


def test_exact_small_is_certified():
    _, _, probability = make_probability_surface(6)
    result = solve_exact_small(probability, drones=2, length=5, time_limit=20)
    validate_paths(result.paths, probability.shape, expected_length=5)
    assert result.status == "OPTIMAL"
    assert result.candidate_gap < 1e-9

