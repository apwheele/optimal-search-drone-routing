"""Run the grid/fleet scaling benchmark used in the notebook.

This is intentionally separate from ordinary notebook execution because the
benchmark performs many optimization runs and should not be repeated whenever
the narrative is rendered.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import ortools
import pandas as pd
import psutil

from search_planner import generate_candidates, make_probability_surface, plan_from_candidates


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts"
GRID_SIZES = [10, 50, 100, 200, 500, 1000]
DRONE_COUNTS = [1, 4, 8]
MASTER_TIME_LIMIT_SECONDS = 5.0
REFINEMENT_ROUNDS = 2


def _cpu_name() -> str:
    """Return a useful CPU name on Windows, macOS, or Linux."""

    try:
        if platform.system() == "Windows":
            command = [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
            value = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
            if value:
                return value
        if platform.system() == "Darwin":
            value = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if value:
                return value
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _machine_specification() -> dict[str, object]:
    memory = psutil.virtual_memory()
    return {
        "benchmark_timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operating_system": platform.platform(),
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "installed_memory_gib": round(memory.total / 2**30, 2),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "ortools_version": ortools.__version__,
        "solver": "Google OR-Tools CP-SAT",
        "solver_workers": 8,
        "gpu_used": False,
        "grid_sizes": GRID_SIZES,
        "drone_counts": DRONE_COUNTS,
        "master_time_limit_seconds_per_call": MASTER_TIME_LIMIT_SECONDS,
        "refinement_rounds": REFINEMENT_ROUNDS,
        "timing_repetitions": 1,
        "timing_clock": "time.perf_counter wall-clock seconds",
    }


def _route_length(grid_size: int, drones: int) -> int:
    """Use 100 cells per drone when feasible; fill the 10x10 grid otherwise."""

    return min(100, (grid_size * grid_size) // drones)


def run_benchmark() -> pd.DataFrame:
    OUTPUT.mkdir(exist_ok=True)
    specification = _machine_specification()
    (OUTPUT / "benchmark_machine.json").write_text(
        json.dumps(specification, indent=2), encoding="utf-8"
    )
    process = psutil.Process(os.getpid())
    rows: list[dict[str, object]] = []
    for grid_size in GRID_SIZES:
        started = perf_counter()
        _, _, probability = make_probability_surface(grid_size=grid_size)
        surface_seconds = perf_counter() - started
        libraries: dict[int, tuple[list, float]] = {}
        for drones in DRONE_COUNTS:
            length = _route_length(grid_size, drones)
            if length not in libraries:
                started = perf_counter()
                candidates = generate_candidates(probability, length=length)
                candidate_seconds = perf_counter() - started
                libraries[length] = (candidates, candidate_seconds)
            candidates, candidate_seconds = libraries[length]
            started = perf_counter()
            result = plan_from_candidates(
                probability,
                candidates,
                drones=drones,
                length=length,
                time_limit=MASTER_TIME_LIMIT_SECONDS,
                random_seed=10_000 + grid_size + drones,
                refinement_rounds=REFINEMENT_ROUNDS,
                translation_radius=3,
            )
            planning_seconds = perf_counter() - started
            row = {
                "grid_side": grid_size,
                "grid_cells": grid_size * grid_size,
                "drones": drones,
                "cells_per_drone": length,
                "surface_seconds": surface_seconds,
                "candidate_generation_seconds": candidate_seconds,
                "planning_seconds": planning_seconds,
                "cold_total_seconds": surface_seconds + candidate_seconds + planning_seconds,
                "initial_candidate_count": len(candidates),
                "final_candidate_count": result.candidate_count,
                "covered_probability": result.score,
                "fraction_of_global_bound": result.score / result.global_relaxation_bound,
                "candidate_gap": result.candidate_gap,
                "status": result.status,
                "resident_memory_after_mib": process.memory_info().rss / 2**20,
            }
            rows.append(row)
            # Checkpoint after every case so a later failure does not discard
            # already completed timing measurements.
            pd.DataFrame(rows).to_csv(OUTPUT / "scaling_benchmark.csv", index=False)
            print(
                f"{grid_size:4d}^2, k={drones}: "
                f"candidates={candidate_seconds:7.3f}s, plan={planning_seconds:7.3f}s, "
                f"status={result.status}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    return frame


if __name__ == "__main__":
    result = run_benchmark()
    print(result.to_string(index=False))
