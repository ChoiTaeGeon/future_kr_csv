"""
Tick Window Generator & Coarse-to-Fine Exploration Logic
- Single source of truth for (start, end, step) parameter generation
- Supports dynamic config updates without code refactoring
- 2-Stage Coarse-to-Fine screen logic
"""
from typing import List


def generate_tick_windows(start: int, end: int, step: int) -> List[int]:
    """
    Generate list of tick window sizes from start to end with step.
    Strictly positive and ascending.
    Example: start=100, end=1000, step=10 -> [100, 110, ..., 1000] (91 items)
    """
    if start <= 0 or end < start or step <= 0:
        raise ValueError(f"Invalid tick window configuration: start={start}, end={end}, step={step}")
    return list(range(start, end + 1, step))


def generate_coarse_windows(start: int, end: int, coarse_step: int) -> List[int]:
    """
    Generate a sparse list of representative windows for stage 1 coarse screening.
    Example: start=100, end=1000, coarse_step=100 -> [100, 200, ..., 1000]
    """
    coarse_step = max(coarse_step, 1)
    windows = list(range(start, end + 1, coarse_step))
    if end not in windows:
        windows.append(end)
    return sorted(list(set(windows)))


def generate_fine_neighborhood_windows(
    top_windows: List[int],
    fine_step: int,
    neighborhood_size: int,
    min_bound: int,
    max_bound: int
) -> List[int]:
    """
    For stage 2 fine exploration: expand top-performing coarse windows by ±neighborhood_size
    using the fine_step specified by user.
    """
    fine_set = set()
    for center in top_windows:
        lower = max(min_bound, center - neighborhood_size)
        upper = min(max_bound, center + neighborhood_size)
        for w in range(lower, upper + 1, fine_step):
            fine_set.add(w)
    return sorted(list(fine_set))
