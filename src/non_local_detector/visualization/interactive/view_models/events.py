"""``EventOverlay`` dataclass for marker-line / shaded-band overlays.

Pure data carrier; the panel-side rendering (``EventOverlayMixin``)
and viewer-level dispatch live in their respective modules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def find_duplicate_overlay_names(overlays: Iterable[EventOverlay]) -> list[str]:
    """Return sorted overlay names that appear more than once."""
    counts = Counter(ovl.name for ovl in overlays)
    return sorted(name for name, count in counts.items() if count > 1)


@dataclass(frozen=True)
class EventOverlay:
    """Set of event marker lines or shaded bands for a time-axis panel.

    Two kinds:

    - ``points``: vertical marker lines at each ``times[i]``.
    - ``intervals``: shaded vertical bands between
      ``t_start[i]`` / ``t_end[i]``.

    Construct via ``EventOverlay.points(...)`` or
    ``EventOverlay.intervals(...)``; the ``__init__`` is private to
    keep the discriminated-union shape consistent.
    """

    name: str
    kind: Literal["points", "intervals"]
    color: str = "#1f77b4"
    alpha: float = 0.15
    times: np.ndarray | None = None
    t_start: np.ndarray | None = None
    t_end: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == "points":
            if self.times is None:
                raise ValueError("EventOverlay(kind='points') requires `times`.")
            if self.t_start is not None or self.t_end is not None:
                raise ValueError(
                    "EventOverlay(kind='points') must not pass `t_start` / `t_end`."
                )
        elif self.kind == "intervals":
            if self.t_start is None or self.t_end is None:
                raise ValueError(
                    "EventOverlay(kind='intervals') requires `t_start` and `t_end`."
                )
            if self.times is not None:
                raise ValueError(
                    "EventOverlay(kind='intervals') must not pass `times`."
                )
            if len(self.t_start) != len(self.t_end):
                raise ValueError(
                    "EventOverlay(kind='intervals') requires "
                    f"len(t_start) == len(t_end). Got "
                    f"{len(self.t_start)} vs {len(self.t_end)}."
                )
        else:
            raise ValueError(
                f"EventOverlay.kind must be 'points' or 'intervals'. Got {self.kind!r}."
            )

    @classmethod
    def points(
        cls,
        name: str,
        times: np.ndarray,
        color: str = "#1f77b4",
        alpha: float = 0.5,
        metadata: dict | None = None,
    ) -> EventOverlay:
        """Construct a points overlay (one vertical line per time)."""
        return cls(
            name=name,
            kind="points",
            color=color,
            alpha=alpha,
            times=np.asarray(times),
            metadata=metadata or {},
        )

    @classmethod
    def intervals(
        cls,
        name: str,
        t_start: np.ndarray,
        t_end: np.ndarray,
        color: str = "#1f77b4",
        alpha: float = 0.15,
        metadata: dict | None = None,
    ) -> EventOverlay:
        """Construct an intervals overlay (shaded band per (start, end))."""
        return cls(
            name=name,
            kind="intervals",
            color=color,
            alpha=alpha,
            t_start=np.asarray(t_start),
            t_end=np.asarray(t_end),
            metadata=metadata or {},
        )
