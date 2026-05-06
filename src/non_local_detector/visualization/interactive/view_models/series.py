"""Generic-series view-models and the user-facing ``MetricSpec`` carrier.

Phase 1b ships only the ``MetricSpec`` dataclass (used as the type
annotation for ``RunBundle.extra_metrics``). The ``LineSeriesModel`` /
``MultiLineSeriesModel`` / ``ScatterSeriesModel`` / ``IntervalSeriesModel``
implementations land in Phase 3 in this same file.
"""

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
                    "MetricSpec(kind='intervals') requires "
                    "`t_start` and `t_end`."
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
