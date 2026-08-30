import numpy as np

from search_planner import (
    Candidate,
    generate_candidates,
    make_probability_surface,
    path_score,
    plan_from_candidates,
    solve_exact_small,
    update_probability_after_no_detection,
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


def test_translation_refinement_closes_a_packing_gap():
    probability = np.ones((8, 8), dtype=float)
    probability[2:5, :4] = np.array([[4.0], [8.0], [4.0]])
    probability /= probability.sum()
    flat = probability.ravel()
    routes = [(8, 9, 10, 11), (40, 41, 42, 43)]
    candidates = [
        Candidate(path=route, score=float(flat[list(route)].sum()), source="test")
        for route in routes
    ]
    initial = plan_from_candidates(
        probability, candidates, drones=2, length=4, refinement_rounds=0
    )
    refined = plan_from_candidates(
        probability,
        candidates,
        drones=2,
        length=4,
        refinement_rounds=3,
        translation_radius=1,
    )
    validate_paths(refined.paths, probability.shape, expected_length=4)
    assert refined.score > initial.score


def test_negative_search_update_zeros_and_renormalizes():
    prior = np.full((2, 2), 0.25)
    posterior = update_probability_after_no_detection(prior, [[0]], detection_probability=1.0)
    assert posterior[0, 0] == 0.0
    assert np.allclose(posterior.ravel()[1:], 1 / 3)
    assert np.isclose(posterior.sum(), 1.0)

    imperfect = update_probability_after_no_detection(prior, [[0]], detection_probability=0.5)
    assert np.allclose(imperfect.ravel(), [1 / 7, 2 / 7, 2 / 7, 2 / 7])


def test_cell_ids_do_not_overflow_int16_boundary():
    probability = np.ones((200, 200), dtype=float)
    probability /= probability.sum()
    candidates = generate_candidates(probability, length=5)
    all_nodes = [node for candidate in candidates for node in candidate.path]
    assert max(all_nodes) > np.iinfo(np.int16).max
    assert min(all_nodes) >= 0
    assert max(all_nodes) < probability.size
