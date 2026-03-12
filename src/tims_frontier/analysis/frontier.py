from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from tims_frontier.storage.records import make_frontier_record


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    metrics: list[str],
    directions: Mapping[str, str],
    epsilon: float,
) -> bool:
    better_or_equal = True
    strictly_better = False
    for metric in metrics:
        lval = float(left[metric])
        rval = float(right[metric])
        direction = directions[metric]
        if direction == "maximize":
            if lval + epsilon < rval:
                better_or_equal = False
                break
            if lval > rval + epsilon:
                strictly_better = True
        else:
            if lval - epsilon > rval:
                better_or_equal = False
                break
            if lval + epsilon < rval:
                strictly_better = True
    return better_or_equal and strictly_better


def extract_frontier_points(
    trial_rows: Iterable[Mapping[str, Any]],
    resolved_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract per-family nondominated points for the primary endpoint."""

    frontier_cfg = resolved_config["analysis"]["frontier"]
    metrics = list(frontier_cfg["metrics"])
    directions = dict(frontier_cfg["directions"])
    epsilon = float(frontier_cfg["dominance_epsilon"])
    completed = [
        row
        for row in trial_rows
        if row.get("status") == "completed"
        and row.get("MC") not in (None, "")
        and row.get("CQ") not in (None, "")
    ]

    key_counts = Counter((row["family"], float(row["MC"]), float(row["CQ"])) for row in completed)
    frontier_rows: list[dict[str, Any]] = []
    for family in {row["family"] for row in completed}:
        family_rows = [row for row in completed if row["family"] == family]
        nondominated: list[Mapping[str, Any]] = []
        for row in family_rows:
            dominated = any(
                _dominates(other, row, metrics=metrics, directions=directions, epsilon=epsilon)
                for other in family_rows
                if other["trial_id"] != row["trial_id"]
            )
            if not dominated:
                nondominated.append(row)
        for index, row in enumerate(nondominated, start=1):
            duplicate_count = key_counts[(row["family"], float(row["MC"]), float(row["CQ"]))]
            frontier_rows.append(
                make_frontier_record(
                    resolved_config=resolved_config,
                    source_trial=row,
                    frontier_point_id=f"{family}_frontier_{index:04d}",
                    duplicate_count=duplicate_count,
                )
            )
    return frontier_rows
