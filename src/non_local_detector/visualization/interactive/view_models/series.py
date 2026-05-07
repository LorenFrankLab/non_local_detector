"""Generic-series view-models and the user-facing ``MetricSpec`` carrier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    """Discriminated union for ``RunBundle.extra_metrics`` entries.

    Three kinds:

    - ``line``: ``(t, y)`` — rendered as a ``LineSeriesPanel``
      (with optional ``fill_below`` / ``thresholds``).
    - ``scatter``: ``(t, y)`` — rendered as a ``ScatterSeriesPanel``
      (with click-to-recenter on by default).
    - ``intervals``: ``(t_start, t_end)`` — rendered as an
      ``IntervalSeriesPanel``.

    Construct via ``MetricSpec.line(...)``, ``MetricSpec.scatter(...)``,
    or ``MetricSpec.intervals(...)``.
    """

    kind: Literal["line", "scatter", "intervals"]
    name: str
    color: str = "#1f77b4"
    t: np.ndarray | None = None
    y: np.ndarray | None = None
    t_start: np.ndarray | None = None
    t_end: np.ndarray | None = None
    fill_below: bool = False
    thresholds: tuple[float, ...] = ()
    y_range: tuple[float, float] | None = None
    click_recenters: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind in ("line", "scatter"):
            if self.t is None or self.y is None:
                raise ValueError(
                    f"MetricSpec(kind={self.kind!r}) requires `t` and `y`."
                )
            if len(self.t) != len(self.y):
                raise ValueError(
                    f"MetricSpec(kind={self.kind!r}) requires "
                    "len(t) == len(y). "
                    f"Got {len(self.t)} vs {len(self.y)}."
                )
        elif self.kind == "intervals":
            if self.t_start is None or self.t_end is None:
                raise ValueError(
                    "MetricSpec(kind='intervals') requires `t_start` and `t_end`."
                )
            if len(self.t_start) != len(self.t_end):
                raise ValueError(
                    "MetricSpec(kind='intervals') requires "
                    f"len(t_start) == len(t_end). Got "
                    f"{len(self.t_start)} vs {len(self.t_end)}."
                )
        else:
            raise ValueError(
                f"MetricSpec.kind must be 'line', 'scatter', or "
                f"'intervals'. Got {self.kind!r}."
            )

    @classmethod
    def line(
        cls,
        name: str,
        t: np.ndarray,
        y: np.ndarray,
        color: str = "#1f77b4",
        fill_below: bool = False,
        thresholds: tuple[float, ...] = (),
        y_range: tuple[float, float] | None = None,
    ) -> MetricSpec:
        return cls(
            kind="line",
            name=name,
            t=np.asarray(t),
            y=np.asarray(y),
            color=color,
            fill_below=fill_below,
            thresholds=tuple(thresholds),
            y_range=y_range,
        )

    @classmethod
    def scatter(
        cls,
        name: str,
        t: np.ndarray,
        y: np.ndarray,
        color: str = "#1f77b4",
        click_recenters: bool = True,
        y_range: tuple[float, float] | None = None,
    ) -> MetricSpec:
        return cls(
            kind="scatter",
            name=name,
            t=np.asarray(t),
            y=np.asarray(y),
            color=color,
            click_recenters=click_recenters,
            y_range=y_range,
        )

    @classmethod
    def intervals(
        cls,
        name: str,
        t_start: np.ndarray,
        t_end: np.ndarray,
        color: str = "#1f77b4",
    ) -> MetricSpec:
        return cls(
            kind="intervals",
            name=name,
            t_start=np.asarray(t_start),
            t_end=np.asarray(t_end),
            color=color,
        )


# ---------------------------------------------------------------------------
# Series view-models — backend-agnostic data carriers consumed by the four
# generic series panels in panels/qt/series.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineSeriesModel:
    """Single time-series line, optionally with fill-below + threshold lines."""

    name: str
    t: np.ndarray
    y: np.ndarray
    color: str = "#1f77b4"
    fill_below: bool = False
    thresholds: tuple[float, ...] = ()
    y_range: tuple[float, float] | None = None

    @classmethod
    def from_metric_spec(cls, spec: MetricSpec) -> LineSeriesModel:
        if spec.kind != "line":
            raise ValueError(
                f"LineSeriesModel.from_metric_spec requires kind='line'; "
                f"got {spec.kind!r}."
            )
        return cls(
            name=spec.name,
            t=np.asarray(spec.t),
            y=np.asarray(spec.y),
            color=spec.color,
            fill_below=spec.fill_below,
            thresholds=spec.thresholds,
            y_range=spec.y_range,
        )

    def window(self, t_start: float, t_stop: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t_window, y_window)`` clipped to ``[t_start, t_stop]``."""
        mask = (self.t >= t_start) & (self.t <= t_stop)
        return self.t[mask], self.y[mask]


@dataclass(frozen=True)
class MultiLineSeriesModel:
    """Multiple lines on one panel, sharing a common time axis."""

    name: str
    t: np.ndarray
    ys: dict[str, np.ndarray]
    colors: dict[str, str] | None = None
    y_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for label, y in self.ys.items():
            if len(y) != len(self.t):
                raise ValueError(
                    f"MultiLineSeriesModel: ys[{label!r}] has len {len(y)} "
                    f"but t has len {len(self.t)}."
                )

    def window(
        self, t_start: float, t_stop: float
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        mask = (self.t >= t_start) & (self.t <= t_stop)
        return self.t[mask], {label: y[mask] for label, y in self.ys.items()}


@dataclass(frozen=True)
class ScatterSeriesModel:
    """Scatter of ``(t, y)`` points; click-to-recenter on by default."""

    name: str
    t: np.ndarray
    y: np.ndarray
    color: str = "#1f77b4"
    click_recenters: bool = True
    y_range: tuple[float, float] | None = None

    @classmethod
    def from_metric_spec(cls, spec: MetricSpec) -> ScatterSeriesModel:
        if spec.kind != "scatter":
            raise ValueError(
                f"ScatterSeriesModel.from_metric_spec requires "
                f"kind='scatter'; got {spec.kind!r}."
            )
        return cls(
            name=spec.name,
            t=np.asarray(spec.t),
            y=np.asarray(spec.y),
            color=spec.color,
            click_recenters=spec.click_recenters,
            y_range=spec.y_range,
        )

    def window(self, t_start: float, t_stop: float) -> tuple[np.ndarray, np.ndarray]:
        mask = (self.t >= t_start) & (self.t <= t_stop)
        return self.t[mask], self.y[mask]


@dataclass(frozen=True)
class IntervalSeriesModel:
    """Shaded vertical bands per ``(t_start[i], t_end[i])`` pair."""

    name: str
    t_start: np.ndarray
    t_end: np.ndarray
    color: str = "#1f77b4"
    alpha: float = 0.15

    def __post_init__(self) -> None:
        if len(self.t_start) != len(self.t_end):
            raise ValueError(
                "IntervalSeriesModel: t_start and t_end must be the same "
                f"length. Got {len(self.t_start)} vs {len(self.t_end)}."
            )

    @classmethod
    def from_metric_spec(cls, spec: MetricSpec) -> IntervalSeriesModel:
        if spec.kind != "intervals":
            raise ValueError(
                f"IntervalSeriesModel.from_metric_spec requires "
                f"kind='intervals'; got {spec.kind!r}."
            )
        return cls(
            name=spec.name,
            t_start=np.asarray(spec.t_start),
            t_end=np.asarray(spec.t_end),
            color=spec.color,
        )

    def window(self, t_start: float, t_stop: float) -> tuple[np.ndarray, np.ndarray]:
        """Return intervals overlapping ``[t_start, t_stop]``."""
        mask = (self.t_end >= t_start) & (self.t_start <= t_stop)
        return self.t_start[mask], self.t_end[mask]
