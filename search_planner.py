"""Probability-weighted, coordinated search paths on a regular grid.

The scalable solver uses a route-generation/set-packing decomposition:

1. Generate many valid elementary paths using compact sweep templates and
   adaptive endpoint beam search.
2. Use CP-SAT to choose exactly k mutually disjoint paths with maximum reward.
3. Add nearby translations of the selected paths and re-solve, closing packing
   gaps caused by pruning routes that were weaker in isolation.

An independent time-indexed CP-SAT model is included for small-instance
optimality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from time import perf_counter
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Candidate:
    """One feasible search route and its collected probability."""

    path: tuple[int, ...]
    score: float
    source: str


@dataclass
class PlanResult:
    """Selected routes and optimization diagnostics."""

    paths: list[list[int]]
    score: float
    status: str
    runtime_seconds: float
    candidate_bound: float
    candidate_gap: float
    global_relaxation_bound: float
    candidate_count: int


def make_probability_surface(
    grid_size: int = 100, extent: float = 10.0, bandwidth: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the three-mode Gaussian KDE surface from Wheeler's scripts.

    The omitted Gaussian normalizing constant cancels when cell weights are
    normalized to sum to one.
    """

    centers = np.array(
        [(1, 5), (6.5, 1), (8, 2), (9.5, 3), (8, 8), (8, 9), (9, 8), (9, 9)],
        dtype=float,
    )
    repeats = np.array([13, 5, 5, 5, 4, 4, 4, 4])
    observations = np.repeat(centers, repeats, axis=0)
    axis = np.linspace(0.0, extent, grid_size)
    x, y = np.meshgrid(axis, axis)
    sample = np.column_stack((x.ravel(), y.ravel()))
    squared_distance = ((sample[:, None, :] - observations[None, :, :]) ** 2).sum(axis=2)
    density = np.exp(-squared_distance / (2.0 * bandwidth**2)).sum(axis=1)
    probability = (density / density.sum()).reshape(grid_size, grid_size)
    return x, y, probability


def _neighbor_ids(node: int, shape: tuple[int, int]) -> list[int]:
    rows, cols = shape
    row, col = divmod(int(node), cols)
    result: list[int] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append(nr * cols + nc)
    return result


def path_score(probability: np.ndarray, paths: Sequence[Sequence[int]]) -> float:
    """Return total probability covered by a collection of paths."""

    flat = probability.ravel()
    return float(sum(flat[node] for path in paths for node in path))


def validate_paths(
    paths: Sequence[Sequence[int]], shape: tuple[int, int], expected_length: int | None = None
) -> None:
    """Raise ValueError unless routes are simple, disjoint, and 8-connected."""

    rows, cols = shape
    all_nodes: list[int] = []
    for drone, path in enumerate(paths):
        if expected_length is not None and len(path) != expected_length:
            raise ValueError(f"drone {drone} has {len(path)} cells, expected {expected_length}")
        if len(set(path)) != len(path):
            raise ValueError(f"drone {drone} revisits a cell")
        for node in path:
            if not 0 <= int(node) < rows * cols:
                raise ValueError(f"drone {drone} contains an out-of-grid cell")
        for left, right in zip(path, path[1:]):
            lr, lc = divmod(int(left), cols)
            rr, rc = divmod(int(right), cols)
            if max(abs(lr - rr), abs(lc - rc)) != 1:
                raise ValueError(f"drone {drone} has a nonadjacent move: {left} -> {right}")
        all_nodes.extend(int(node) for node in path)
    if len(set(all_nodes)) != len(all_nodes):
        raise ValueError("two drones receive credit for the same cell")


def _canonical(path: Sequence[int]) -> tuple[int, ...]:
    route = tuple(int(node) for node in path)
    reverse = route[::-1]
    return min(route, reverse)


def _snake_template(length: int, width: int) -> np.ndarray:
    height = ceil(length / width)
    coordinates: list[tuple[int, int]] = []
    for row in range(height):
        columns: Iterable[int] = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
        coordinates.extend((row, col) for col in columns)
    return np.asarray(coordinates[:length], dtype=np.int16)


def _template_transforms(coordinates: np.ndarray) -> list[np.ndarray]:
    """Return unique rotations/reflections of a relative-coordinate path."""

    variants: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for swap in (False, True):
        for sign_row in (-1, 1):
            for sign_col in (-1, 1):
                transformed = coordinates[:, [1, 0]] if swap else coordinates.copy()
                transformed = transformed * np.array([sign_row, sign_col], dtype=np.int16)
                transformed = transformed - transformed.min(axis=0)
                key = tuple(map(tuple, transformed.tolist()))
                reverse_key = key[::-1]
                canonical_key = min(key, reverse_key)
                variants[canonical_key] = transformed
    return list(variants.values())


def _useful_widths(length: int) -> list[int]:
    limit = min(length, max(10, int(2.2 * sqrt(length))))
    if limit <= 18:
        return list(range(1, limit + 1))
    widths = set(range(1, 11))
    widths.update(np.linspace(12, limit, 10, dtype=int).tolist())
    widths.update({max(1, int(sqrt(length))), max(1, int(round(sqrt(length) / 2)))})
    return sorted(width for width in widths if width <= length)


def _promising_anchors(
    scores: np.ndarray, global_count: int = 20, spatial_bins: int = 4
) -> set[tuple[int, int]]:
    """Keep globally strong anchors plus the best anchor in each spatial bin."""

    flat = scores.ravel()
    count = min(global_count, flat.size)
    indices = np.argpartition(flat, -count)[-count:] if count < flat.size else np.arange(flat.size)
    anchors = {tuple(map(int, np.unravel_index(index, scores.shape))) for index in indices}
    row_edges = np.linspace(0, scores.shape[0], spatial_bins + 1, dtype=int)
    col_edges = np.linspace(0, scores.shape[1], spatial_bins + 1, dtype=int)
    for row_bin in range(spatial_bins):
        for col_bin in range(spatial_bins):
            r0, r1 = row_edges[row_bin], row_edges[row_bin + 1]
            c0, c1 = col_edges[col_bin], col_edges[col_bin + 1]
            if r0 == r1 or c0 == c1:
                continue
            block = scores[r0:r1, c0:c1]
            local = int(np.argmax(block))
            local_row, local_col = np.unravel_index(local, block.shape)
            anchors.add((int(r0 + local_row), int(c0 + local_col)))
    return anchors


def _sweep_candidates(probability: np.ndarray, length: int) -> list[Candidate]:
    rows, cols = probability.shape
    found: dict[tuple[int, ...], Candidate] = {}
    for width in _useful_widths(length):
        base = _snake_template(length, width)
        for template in _template_transforms(base):
            height = int(template[:, 0].max()) + 1
            box_width = int(template[:, 1].max()) + 1
            if height > rows or box_width > cols:
                continue
            score_grid = np.zeros((rows - height + 1, cols - box_width + 1), dtype=float)
            for dr, dc in template:
                score_grid += probability[dr : dr + score_grid.shape[0], dc : dc + score_grid.shape[1]]
            for anchor_row, anchor_col in _promising_anchors(score_grid):
                coords = template + np.array([anchor_row, anchor_col], dtype=np.int16)
                path = tuple((coords[:, 0] * cols + coords[:, 1]).astype(int).tolist())
                key = _canonical(path)
                score = float(score_grid[anchor_row, anchor_col])
                candidate = Candidate(path=path, score=score, source="sweep")
                if key not in found or found[key].score < score:
                    found[key] = candidate
    return list(found.values())


def _diverse_seeds(probability: np.ndarray, length: int, count: int = 28) -> list[int]:
    rows, cols = probability.shape
    separation = max(2, int(sqrt(length) / 3))
    seeds: list[int] = []
    for node in np.argsort(probability.ravel())[::-1]:
        row, col = divmod(int(node), cols)
        if all(max(abs(row - divmod(old, cols)[0]), abs(col - divmod(old, cols)[1])) >= separation for old in seeds):
            seeds.append(int(node))
            if len(seeds) == count:
                break
    # Ensure broad geographic coverage even when most probability lies in one mode.
    for row in np.linspace(0, rows - 1, 5, dtype=int):
        for col in np.linspace(0, cols - 1, 5, dtype=int):
            seeds.append(int(row * cols + col))
    return list(dict.fromkeys(seeds))


def _beam_candidates(
    probability: np.ndarray, length: int, beam_width: int = 18, returns_per_seed: int = 3
) -> list[Candidate]:
    """Grow paths from both endpoints, retaining diverse high-value partial routes."""

    shape = probability.shape
    flat = probability.ravel()
    neighbors = [_neighbor_ids(node, shape) for node in range(flat.size)]
    completed: dict[tuple[int, ...], Candidate] = {}
    for seed in _diverse_seeds(probability, length):
        # Entries are (ranking value, true score, path, used cells).
        beam: list[tuple[float, float, tuple[int, ...], frozenset[int]]] = [
            (float(flat[seed]), float(flat[seed]), (seed,), frozenset((seed,)))
        ]
        for _ in range(1, length):
            expanded: list[tuple[float, float, tuple[int, ...], frozenset[int]]] = []
            signatures: set[tuple[int, int, int]] = set()
            for _, score, path, used in beam:
                moves: list[tuple[int, int]] = []
                moves.extend((0, node) for node in neighbors[path[0]] if node not in used)
                moves.extend((1, node) for node in neighbors[path[-1]] if node not in used)
                # Limit branching to the best moves per partial route.
                moves.sort(key=lambda item: float(flat[item[1]]), reverse=True)
                for side, node in moves[:8]:
                    new_path = (node,) + path if side == 0 else path + (node,)
                    new_used = used | {node}
                    new_score = score + float(flat[node])
                    onward = [n for n in neighbors[node] if n not in new_used]
                    lookahead = max((float(flat[n]) for n in onward), default=-1.0)
                    accessibility = len(onward) * float(flat.mean()) * 0.03
                    ranking = new_score + 0.65 * max(lookahead, 0.0) + accessibility
                    signature = (new_path[0], new_path[-1], hash(new_used))
                    if signature not in signatures:
                        signatures.add(signature)
                        expanded.append((ranking, new_score, new_path, new_used))
            if not expanded:
                beam = []
                break
            expanded.sort(key=lambda state: state[0], reverse=True)
            beam = expanded[:beam_width]
        for _, score, path, _ in sorted(beam, key=lambda state: state[1], reverse=True)[:returns_per_seed]:
            key = _canonical(path)
            completed[key] = Candidate(path=path, score=score, source="beam")
    return list(completed.values())


def generate_candidates(probability: np.ndarray, length: int) -> list[Candidate]:
    """Generate a diverse library of valid fixed-length routes."""

    if not 1 <= length <= probability.size:
        raise ValueError("length must be between 1 and the number of cells")
    candidates = _sweep_candidates(probability, length)
    candidates.extend(_beam_candidates(probability, length))
    unique: dict[tuple[int, ...], Candidate] = {}
    for candidate in candidates:
        key = _canonical(candidate.path)
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    result = sorted(unique.values(), key=lambda candidate: candidate.score, reverse=True)
    for candidate in result:
        validate_paths([candidate.path], probability.shape, length)
    return result


def _candidate_greedy(candidates: Sequence[Candidate], drones: int) -> list[int]:
    selected: list[int] = []
    used: set[int] = set()
    for index, candidate in enumerate(candidates):
        if used.isdisjoint(candidate.path):
            selected.append(index)
            used.update(candidate.path)
            if len(selected) == drones:
                return selected
    return selected


def _translated_candidates(
    probability: np.ndarray,
    selected_paths: Sequence[Sequence[int]],
    radius: int,
) -> list[Candidate]:
    """Generate nearby translations of incumbent paths for master refinement.

    Initial route generation intentionally keeps only a small number of strong
    anchors per template.  That is efficient for one route, but coordinated
    packing can need a slightly lower-scoring translation that fits tightly
    beside another route.  Enriching around the incumbent supplies those
    columns without enumerating every translation of every template.
    """

    rows, cols = probability.shape
    flat = probability.ravel()
    translated: dict[tuple[int, ...], Candidate] = {}
    for path in selected_paths:
        coords = np.array([divmod(int(node), cols) for node in path], dtype=int)
        for row_shift in range(-radius, radius + 1):
            for col_shift in range(-radius, radius + 1):
                if row_shift == 0 and col_shift == 0:
                    continue
                shifted = coords + np.array([row_shift, col_shift])
                if (
                    shifted[:, 0].min() < 0
                    or shifted[:, 0].max() >= rows
                    or shifted[:, 1].min() < 0
                    or shifted[:, 1].max() >= cols
                ):
                    continue
                route = tuple((shifted[:, 0] * cols + shifted[:, 1]).astype(int).tolist())
                key = _canonical(route)
                translated[key] = Candidate(
                    path=route,
                    score=float(flat[list(route)].sum()),
                    source="translation refinement",
                )
    return list(translated.values())


def _solve_candidate_master(
    candidates: Sequence[Candidate],
    cell_count: int,
    drones: int,
    time_limit: float,
    random_seed: int,
) -> tuple[list[int], str, float]:
    """Solve one set-packing master and return indices, status, and bound."""

    if drones == 1:
        return [0], "OPTIMAL", candidates[0].score

    model = cp_model.CpModel()
    choose = [model.new_bool_var(f"route_{index}") for index in range(len(candidates))]
    by_cell: list[list[int]] = [[] for _ in range(cell_count)]
    for index, candidate in enumerate(candidates):
        for node in candidate.path:
            by_cell[node].append(index)
    model.add(sum(choose) == drones)
    for covering in by_cell:
        if len(covering) > 1:
            model.add(sum(choose[index] for index in covering) <= 1)
    scale = 10**12
    rewards = [int(round(candidate.score * scale)) for candidate in candidates]
    model.maximize(sum(reward * variable for reward, variable in zip(rewards, choose)))
    greedy_hint = set(_candidate_greedy(candidates, drones))
    if len(greedy_hint) == drones:
        for index, variable in enumerate(choose):
            model.add_hint(variable, int(index in greedy_hint))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = int(random_seed)
    status_code = solver.solve(model)
    raw_status = solver.status_name(status_code)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = list(greedy_hint)
        if len(selected) != drones:
            raise RuntimeError("candidate library does not contain enough disjoint routes")
        return selected, f"{raw_status}; greedy fallback", float("nan")
    selected = [index for index, variable in enumerate(choose) if solver.value(variable)]
    return selected, raw_status, float(solver.best_objective_bound / scale)


def plan_from_candidates(
    probability: np.ndarray,
    candidates: Sequence[Candidate],
    drones: int,
    length: int,
    time_limit: float = 12.0,
    random_seed: int = 2026,
    refinement_rounds: int = 3,
    translation_radius: int = 3,
) -> PlanResult:
    """Select disjoint routes and iteratively enrich around the incumbent.

    ``time_limit`` applies to each candidate-master solve.  The returned bound
    and status apply only to the final enriched route library, not to all
    mathematically possible grid paths.
    """

    if drones * length > probability.size:
        raise ValueError("the requested disjoint coverage exceeds the grid")
    started = perf_counter()
    working = list(candidates)
    known = {_canonical(candidate.path) for candidate in working}
    selected, raw_status, candidate_bound = _solve_candidate_master(
        working, probability.size, drones, time_limit, random_seed
    )
    for refinement in range(refinement_rounds):
        incumbent_paths = [working[index].path for index in selected]
        additions = []
        for candidate in _translated_candidates(probability, incumbent_paths, translation_radius):
            key = _canonical(candidate.path)
            if key not in known:
                known.add(key)
                additions.append(candidate)
        if not additions:
            break
        working.extend(additions)
        working.sort(key=lambda candidate: candidate.score, reverse=True)
        selected, raw_status, candidate_bound = _solve_candidate_master(
            working,
            probability.size,
            drones,
            time_limit,
            random_seed + refinement + 1,
        )
    paths = [list(working[index].path) for index in selected]
    validate_paths(paths, probability.shape, length)
    score = path_score(probability, paths)
    global_bound = float(np.sort(probability.ravel())[-drones * length :].sum())
    candidate_gap = max(0.0, candidate_bound - score) / candidate_bound if candidate_bound > 0 else 0.0
    if raw_status in {"OPTIMAL", "FEASIBLE"}:
        status = f"{raw_status} (enriched candidate library)"
    else:
        status = raw_status
    return PlanResult(
        paths=paths,
        score=score,
        status=status,
        runtime_seconds=perf_counter() - started,
        candidate_bound=candidate_bound,
        candidate_gap=candidate_gap,
        global_relaxation_bound=global_bound,
        candidate_count=len(working),
    )


def greedy_endpoint_plan(probability: np.ndarray, drones: int, length: int) -> list[list[int]]:
    """Simple sequential endpoint-greedy baseline, similar in spirit to the blog attempt."""

    flat = probability.ravel()
    neighbors = [_neighbor_ids(node, probability.shape) for node in range(flat.size)]
    globally_used: set[int] = set()
    paths: list[list[int]] = []
    for _ in range(drones):
        starts = [node for node in np.argsort(flat)[::-1] if int(node) not in globally_used]
        if not starts:
            break
        path = [int(starts[0])]
        used = {path[0]}
        while len(path) < length:
            moves: list[tuple[float, int, int]] = []
            for side, endpoint in ((0, path[0]), (1, path[-1])):
                for node in neighbors[endpoint]:
                    if node not in globally_used and node not in used:
                        moves.append((float(flat[node]), side, node))
            if not moves:
                raise RuntimeError("greedy baseline became trapped before using its budget")
            _, side, node = max(moves)
            path.insert(0, node) if side == 0 else path.append(node)
            used.add(node)
        paths.append(path)
        globally_used.update(path)
    validate_paths(paths, probability.shape, length)
    return paths


def solve_exact_small(
    probability: np.ndarray,
    drones: int,
    length: int,
    time_limit: float = 60.0,
) -> PlanResult:
    """Solve the full time-indexed model; intended only for small grids."""

    if probability.size > 144:
        raise ValueError("exact model is intentionally limited to at most 144 cells")
    started = perf_counter()
    nodes = range(probability.size)
    model = cp_model.CpModel()
    visit = {
        (drone, step, node): model.new_bool_var(f"x_{drone}_{step}_{node}")
        for drone in range(drones)
        for step in range(length)
        for node in nodes
    }
    for drone in range(drones):
        for step in range(length):
            model.add(sum(visit[drone, step, node] for node in nodes) == 1)
    for node in nodes:
        model.add(sum(visit[drone, step, node] for drone in range(drones) for step in range(length)) <= 1)
    neighbors = [_neighbor_ids(node, probability.shape) for node in nodes]
    for drone in range(drones):
        for step in range(1, length):
            for node in nodes:
                model.add(visit[drone, step, node] <= sum(visit[drone, step - 1, prior] for prior in neighbors[node]))
    # Remove interchangeable-drone symmetry by ordering start-cell IDs.
    starts = []
    for drone in range(drones):
        start = model.new_int_var(0, probability.size - 1, f"start_{drone}")
        model.add(start == sum(node * visit[drone, 0, node] for node in nodes))
        starts.append(start)
    for drone in range(drones - 1):
        model.add(starts[drone] < starts[drone + 1])
    scale = 10**12
    reward = np.rint(probability.ravel() * scale).astype(np.int64)
    model.maximize(
        sum(int(reward[node]) * visit[drone, step, node] for drone in range(drones) for step in range(length) for node in nodes)
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 2026
    status_code = solver.solve(model)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"exact solver returned {solver.status_name(status_code)}")
    paths: list[list[int]] = []
    for drone in range(drones):
        path = []
        for step in range(length):
            path.append(next(node for node in nodes if solver.value(visit[drone, step, node])))
        paths.append(path)
    validate_paths(paths, probability.shape, length)
    score = path_score(probability, paths)
    bound = float(solver.best_objective_bound / scale)
    global_bound = float(np.sort(probability.ravel())[-drones * length :].sum())
    return PlanResult(
        paths=paths,
        score=score,
        status=solver.status_name(status_code),
        runtime_seconds=perf_counter() - started,
        candidate_bound=bound,
        candidate_gap=max(0.0, bound - score) / bound if bound > 0 else 0.0,
        global_relaxation_bound=global_bound,
        candidate_count=0,
    )


def plot_surface(
    probability: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Probability surface",
) -> plt.Axes:
    """Plot the probability surface on the original 0--10 coordinate scale."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.contourf(
        np.linspace(0, 10, probability.shape[1]),
        np.linspace(0, 10, probability.shape[0]),
        probability * probability.size,
        levels=16,
        cmap="magma_r",
    )
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="relative probability (mean = 1)")
    return ax


def plot_paths(
    probability: np.ndarray,
    paths: Sequence[Sequence[int]],
    ax: plt.Axes | None = None,
    title: str = "Coordinated search paths",
) -> plt.Axes:
    """Overlay ordered drone routes on the probability surface."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_surface(probability, ax=ax, title=title)
    colors = plt.colormaps["tab10"]
    rows, cols = probability.shape
    for drone, path in enumerate(paths):
        coordinates = np.array([divmod(int(node), cols) for node in path])
        x = coordinates[:, 1] * 10 / (cols - 1)
        y = coordinates[:, 0] * 10 / (rows - 1)
        line = ax.plot(x, y, color=colors(drone % 10), linewidth=1.7, label=f"drone {drone + 1}")[0]
        line.set_path_effects([patheffects.Stroke(linewidth=3.2, foreground="white"), patheffects.Normal()])
        ax.scatter(x[0], y[0], s=38, color=colors(drone % 10), edgecolor="white", linewidth=0.8, zorder=5)
        ax.scatter(x[-1], y[-1], s=48, marker="X", color=colors(drone % 10), edgecolor="white", linewidth=0.8, zorder=5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    return ax
