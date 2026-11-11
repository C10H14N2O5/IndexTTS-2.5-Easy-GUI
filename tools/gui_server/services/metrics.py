"""Bounded TensorBoard scalar reads without starting its HTTP server."""
from __future__ import annotations

import math
from pathlib import Path
from pydantic import Field
from ..schemas import ApiModel
from . import projects


class MetricSeries(ApiModel):
    label: str
    points: list[tuple[int, float]]
    count: int


class MetricsResponse(ApiModel):
    project: str
    adapter: str
    logdir: str
    series: list[MetricSeries] = Field(default_factory=list)
    available: bool = False
    error: str | None = None


def sample_points(points: list[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
    """Evenly spaced, deterministic sampling including first and latest points."""
    if len(points) <= limit:
        return points
    if limit == 1:
        return points[-1:]
    return [points[round(i * (len(points) - 1) / (limit - 1))] for i in range(limit)]


def read(name: str, adapter: str | None = None, max_points: int = 2000) -> dict:
    root = projects.output_dir(name, adapter)
    logdir = root / "tensorboard"
    result = dict(project=name, adapter=root.name, logdir=str(logdir), series=[], available=False, error=None)
    _read_events(logdir, result, max_points)
    from . import posthoc
    extra = posthoc.series(name, root)
    if extra is not None:
        result["series"].append({**extra, "points": sample_points(extra["points"], max_points)})
        result["available"] = True
    return result


def _read_events(logdir: Path, result: dict, max_points: int) -> None:
    if not logdir.is_dir() or not any(logdir.glob("events.out.tfevents.*")):
        return
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    try:
        # Reservoir bounded independently of file size. A second deterministic pass
        # caps each returned series and always retains the latest received step.
        accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 2000})
        accumulator.Reload()
        for tag in sorted(accumulator.Tags().get("scalars", [])):
            values = accumulator.Scalars(tag)
            points_by_step = {int(point.step): float(point.value) for point in values if math.isfinite(point.value)}
            points = sorted(points_by_step.items())
            result["series"].append(dict(label=tag, points=sample_points(points, max_points), count=len(points)))
        result["available"] = bool(result["series"])
    except (OSError, ValueError) as exc:
        # An event file can be partially appended while the trainer is running.
        result["error"] = str(exc)
