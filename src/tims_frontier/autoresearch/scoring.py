from __future__ import annotations

from typing import Any, Mapping

from .models import ScoreBundle


METRICS = ["MC", "CQ"]
DIRECTIONS = {"MC": "maximize", "CQ": "maximize"}


def _normalize_value(metric: str, value: float, runtime_config: Mapping[str, Any]) -> float:
    normalization = runtime_config["scoring"]["normalization"]
    if normalization["mode"] == "none":
        return float(value)
    bounds = normalization["bounds"][metric]
    denominator = float(bounds["max"]) - float(bounds["min"])
    if denominator <= 0:
        raise ValueError(f"Normalization bounds for {metric} must have max > min")
    return (float(value) - float(bounds["min"])) / denominator


def _normalize_point(point: Mapping[str, Any], runtime_config: Mapping[str, Any]) -> dict[str, float]:
    return {metric: _normalize_value(metric, float(point[metric]), runtime_config) for metric in METRICS}


def _normalize_points(points: list[Mapping[str, Any]], runtime_config: Mapping[str, Any]) -> list[dict[str, float]]:
    return [_normalize_point(point, runtime_config) for point in points]


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    better_or_equal = True
    strictly_better = False
    for metric in METRICS:
        lval = float(left[metric])
        rval = float(right[metric])
        if lval < rval:
            better_or_equal = False
            break
        if lval > rval:
            strictly_better = True
    return better_or_equal and strictly_better


def _extract_nondominated_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    nondominated: list[Mapping[str, Any]] = []
    for row in rows:
        dominated = any(_dominates(other, row) for other in rows if other is not row)
        if not dominated:
            nondominated.append(row)
    return nondominated


def compute_frontier(points: list[Mapping[str, Any]]) -> list[dict[str, float]]:
    frontier = _extract_nondominated_rows(points)
    return sorted(
        ({metric: float(point[metric]) for metric in METRICS} for point in frontier),
        key=lambda point: (point["MC"], -point["CQ"]),
    )


def compute_hypervolume(frontier: list[Mapping[str, Any]], reference_point: Mapping[str, Any]) -> float:
    if not frontier:
        return 0.0
    sorted_frontier = sorted(frontier, key=lambda point: float(point["MC"]))
    ref_mc = float(reference_point["MC"])
    ref_cq = float(reference_point["CQ"])
    hypervolume = 0.0
    previous_mc = ref_mc
    for point in sorted_frontier:
        mc = float(point["MC"])
        cq = float(point["CQ"])
        width = max(0.0, mc - previous_mc)
        height = max(0.0, cq - ref_cq)
        hypervolume += width * height
        previous_mc = max(previous_mc, mc)
    return hypervolume


def _point_beyond_baseline(point: Mapping[str, Any], baseline_frontier: list[Mapping[str, Any]]) -> bool:
    return not any(_dominates(baseline, point) for baseline in baseline_frontier)


def _audit_metrics(
    family_points: list[dict[str, float]],
    family_frontier: list[dict[str, float]],
    baseline_frontier: list[dict[str, float]],
    archive_points: list[dict[str, float]] | None,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    beyond = [point for point in family_frontier if _point_beyond_baseline(point, baseline_frontier)]
    unique_count = len({(point["MC"], point["CQ"]) for point in family_points})
    repeat_consistency = unique_count / len(family_points) if family_points else 1.0
    baseline_mc_max = max((point["MC"] for point in baseline_frontier), default=0.0)
    baseline_cq_max = max((point["CQ"] for point in baseline_frontier), default=0.0)
    family_mc_max = max((point["MC"] for point in family_frontier), default=baseline_mc_max)
    family_cq_max = max((point["CQ"] for point in family_frontier), default=baseline_cq_max)
    score_s_marg = None
    if archive_points:
        archive_frontier = compute_frontier(archive_points)
        archive_hv = compute_hypervolume(archive_frontier, runtime_config["scoring"]["reference_point"])
        archive_plus_frontier = compute_frontier(archive_points + family_points)
        archive_plus_hv = compute_hypervolume(archive_plus_frontier, runtime_config["scoring"]["reference_point"])
        score_s_marg = archive_plus_hv - archive_hv
    return {
        "score_s_marg": score_s_marg,
        "frontier_external_span": {
            "mc_gain": family_mc_max - baseline_mc_max,
            "cq_gain": family_cq_max - baseline_cq_max,
        },
        "repeat_consistency": repeat_consistency,
        "point_count_beyond_baseline": len(beyond),
        "continuous_expansion_flag": len(beyond) >= 2,
    }


def score_family(
    family_results: list[Mapping[str, Any]],
    runtime_config: Mapping[str, Any],
    *,
    baseline_points: list[Mapping[str, Any]] | None,
    archive_points: list[Mapping[str, Any]] | None = None,
) -> ScoreBundle:
    """Score one proposal family relative to the fixed baseline."""

    raw_family_points = [{metric: float(item[metric]) for metric in METRICS} for item in family_results]
    normalized_family_points = _normalize_points(raw_family_points, runtime_config)
    family_frontier = compute_frontier(normalized_family_points)
    reference_point = runtime_config["scoring"]["reference_point"]
    family_hv = compute_hypervolume(family_frontier, reference_point)

    if not baseline_points:
        audit = _audit_metrics(normalized_family_points, family_frontier, family_frontier, archive_points, runtime_config)
        audit["baseline_mode"] = "first_round_family"
        return ScoreBundle(
            baseline_points=normalized_family_points,
            baseline_frontier=family_frontier,
            baseline_hv=family_hv,
            family_points=normalized_family_points,
            family_frontier=family_frontier,
            family_hv=family_hv,
            combined_frontier=family_frontier,
            combined_hv=family_hv,
            score_s_abs=0.0,
            audit_metrics=audit,
        )

    normalized_baseline = [{metric: float(item[metric]) for metric in METRICS} for item in baseline_points]
    baseline_frontier = compute_frontier(normalized_baseline)
    baseline_hv = compute_hypervolume(baseline_frontier, reference_point)
    combined_frontier = compute_frontier(normalized_baseline + normalized_family_points)
    combined_hv = compute_hypervolume(combined_frontier, reference_point)
    audit = _audit_metrics(normalized_family_points, family_frontier, baseline_frontier, archive_points, runtime_config)
    return ScoreBundle(
        baseline_points=normalized_baseline,
        baseline_frontier=baseline_frontier,
        baseline_hv=baseline_hv,
        family_points=normalized_family_points,
        family_frontier=family_frontier,
        family_hv=family_hv,
        combined_frontier=combined_frontier,
        combined_hv=combined_hv,
        score_s_abs=combined_hv - baseline_hv,
        audit_metrics=audit,
    )
