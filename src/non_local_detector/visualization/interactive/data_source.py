"""In-memory data source for the interactive viewer.

Holds a dict of named ``RunBundle``s, validates cross-bundle
invariants (time-grid alignment, overlay-schema alignment), and exposes
the hot-path window-load methods the viewer's panels consume.

The data source is a thin abstraction so a v2 ``ZarrDecoderDataSource``
can be swapped in without changing panel code.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from non_local_detector.visualization.interactive.view_models.base import RunBundle
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
    find_duplicate_overlay_names,
)


class InMemoryDecoderDataSource:
    """In-memory data source backed by ``RunBundle`` objects.

    Construction validates:

    1. All runs share the same time grid (``results["time"]`` arrays
       must compare equal under ``np.array_equal``). Mismatched grids
       raise immediately with the offending pair named.
    2. Overlay names are unique within each bundle (already enforced
       by ``RunBundle.__post_init__``, re-validated here for the
       multi-run case).
    3. The ``(name, kind)`` overlay-schema set matches across runs.
       Same-named overlays must agree on whether they are
       ``"points"`` or ``"intervals"`` because navigator semantics
       differ between the two.

    Parameters
    ----------
    runs : dict[str, RunBundle]
        Bundles keyed by run name (the M-key swap UI cycles in this
        order). Must contain at least one entry.
    """

    def __init__(self, runs: dict[str, RunBundle]):
        if not runs:
            raise ValueError("InMemoryDecoderDataSource requires at least one run.")
        self._runs = dict(runs)
        self._validate_time_grid_alignment()
        self._validate_overlay_alignment()
        self._active_run_name = next(iter(self._runs))
        # Cache the per-run position interpolated onto the decoder
        # time grid, computed lazily on first ``load_position`` call.
        # Cleared on ``set_active_run`` because the new bundle may
        # have a different position trajectory.
        self._position_cache: dict[str, np.ndarray | None] = {}

    # ------------------------------------------------------------------
    # Construction validators
    # ------------------------------------------------------------------

    def _validate_time_grid_alignment(self) -> None:
        if len(self._runs) <= 1:
            return
        names = list(self._runs)
        reference_name = names[0]
        reference_time = np.asarray(self._runs[reference_name].results["time"].values)
        for other in names[1:]:
            other_time = np.asarray(self._runs[other].results["time"].values)
            if other_time.shape != reference_time.shape or not np.array_equal(
                other_time, reference_time
            ):
                raise ValueError(
                    f"InMemoryDecoderDataSource requires all runs to share "
                    f"the same time grid. Runs {reference_name!r} (length "
                    f"{reference_time.shape[0]}) and {other!r} (length "
                    f"{other_time.shape[0]}) disagree."
                )

    def _validate_overlay_alignment(self) -> None:
        # Re-check within-bundle uniqueness — RunBundle.__post_init__
        # ran this once at construction but bundles are mutable.
        for run_name, bundle in self._runs.items():
            duplicates = find_duplicate_overlay_names(bundle.event_overlays)
            if duplicates:
                raise ValueError(
                    f"Run {run_name!r} has duplicate overlay names: "
                    f"{duplicates}. Each overlay name must be unique "
                    "within a bundle."
                )

        # Across-bundles: same (name, kind) schema set.
        if len(self._runs) <= 1:
            return
        schemas = {
            name: frozenset((ovl.name, ovl.kind) for ovl in b.event_overlays)
            for name, b in self._runs.items()
        }
        names = list(schemas)
        reference = schemas[names[0]]
        mismatched = {n: s for n, s in schemas.items() if s != reference}
        if mismatched:
            raise ValueError(
                "All RunBundles loaded together must declare the same "
                "overlay (name, kind) schema. Got: "
                f"{ {n: sorted(s) for n, s in schemas.items()}!r}."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_single(cls, bundle: RunBundle) -> InMemoryDecoderDataSource:
        """Single-run convenience: wrap one bundle as the only run."""
        return cls({"default": bundle})

    # ------------------------------------------------------------------
    # Run state
    # ------------------------------------------------------------------

    @property
    def run_names(self) -> list[str]:
        return list(self._runs)

    @property
    def active_run_name(self) -> str:
        return self._active_run_name

    @property
    def active_run(self) -> RunBundle:
        return self._runs[self._active_run_name]

    @property
    def available_outputs(self) -> set[str]:
        """Optional output arrays present in the active run's results."""
        return self.active_run.available_outputs

    def set_active_run(self, name: str) -> None:
        if name not in self._runs:
            raise ValueError(f"No run named {name!r}. Available: {self.run_names!r}")
        # RunBundles are mutable — re-validate so a post-construction
        # overlay drift can't leave the navigator in an inconsistent
        # state across runs.
        self._validate_time_grid_alignment()
        self._validate_overlay_alignment()
        self._active_run_name = name

    # ------------------------------------------------------------------
    # Hot-path readers (slice into active run)
    # ------------------------------------------------------------------

    def window_indices(self, t_center: float, t_width: float) -> slice:
        """Return a ``slice`` selecting the visible time-window indices.

        Half-window on each side of ``t_center``.
        """
        if t_width <= 0:
            raise ValueError(f"t_width must be positive. Got {t_width}.")
        time = np.asarray(self.active_run.results["time"].values)
        half = t_width / 2.0
        start = float(t_center - half)
        stop = float(t_center + half)
        i_start = int(np.searchsorted(time, start, side="left"))
        i_stop = int(np.searchsorted(time, stop, side="right"))
        return slice(i_start, i_stop)

    def load_posterior(self, sl: slice) -> np.ndarray:
        """Window slice of ``acausal_posterior``: ``(n_visible, n_state_bins)``."""
        return np.asarray(
            self.active_run.results["acausal_posterior"].isel(time=sl).values,
            dtype=np.float32,
        )

    def load_likelihood(self, sl: slice) -> np.ndarray:
        """Window slice of ``log_likelihood``.

        Raises a clear ``KeyError`` if ``log_likelihood`` is not in
        the active run's results — user should re-run
        ``predict(return_outputs=["log_likelihood"])``.
        """
        if "log_likelihood" not in self.active_run.results:
            raise KeyError(
                "Active run has no `log_likelihood` in results. Re-run "
                "`predict(return_outputs=['log_likelihood'])` and rebuild "
                "the RunBundle."
            )
        return np.asarray(
            self.active_run.results["log_likelihood"].isel(time=sl).values
        )

    def load_acausal(self, sl: slice) -> np.ndarray | None:
        """Window slice of ``acausal_posterior`` (alias of ``load_posterior``).

        Returns ``None`` only if no ``acausal_posterior`` is present;
        in practice ``acausal_posterior`` is always required by
        ``RunBundle.__post_init__``, so this only returns ``None`` for
        tests that bypass validation.
        """
        if "acausal_posterior" not in self.active_run.results:
            return None
        return self.load_posterior(sl)

    def load_predictive(self, sl: slice) -> np.ndarray | None:
        """Window slice of ``predictive_posterior``, or ``None`` if absent."""
        if "predictive_posterior" not in self.active_run.results:
            return None
        return np.asarray(
            self.active_run.results["predictive_posterior"].isel(time=sl).values
        )

    def load_state_probabilities(self, sl: slice) -> np.ndarray:
        """Window slice of ``acausal_state_probabilities``: ``(n_visible, n_states)``.

        ``acausal_state_probabilities`` is required by
        ``RunBundle.__post_init__``, so this method is unconditional —
        no ``None`` return for the canonical construction path.
        """
        return np.asarray(
            self.active_run.results["acausal_state_probabilities"].isel(time=sl).values
        )

    def load_position(self, sl: slice) -> np.ndarray | None:
        """Window slice of the bundle's true position aligned to decoder time.

        Returns a ``(n_visible,)`` 1D array — the bundle's
        ``position`` (sampled at ``position_time``) linearly
        interpolated onto the decoder result's ``time`` grid, then
        sliced to ``sl``. Returns ``None`` when the bundle has no
        position (``position`` is None) or when the position is 2D
        (``position.ndim > 1``); the v1 heatmap overlay supports 1D
        position only.

        Caches the full interpolated array on first call per active
        run; ``set_active_run`` clears the cache because the next
        bundle's position trajectory may differ.
        """
        position_at_decoder_time = self._position_at_decoder_time()
        if position_at_decoder_time is None:
            return None
        return position_at_decoder_time[sl]

    def _position_at_decoder_time(self) -> np.ndarray | None:
        """Return position interpolated onto the decoder time grid (cached).

        Returns ``None`` if the active run has no 1D position to
        align — caller treats that as "skip the position trace".
        """
        cached = self._position_cache.get(self._active_run_name, ...)
        if cached is not ...:
            return cached  # may be a real array or None
        run = self.active_run
        position = np.asarray(run.position) if run.position is not None else None
        position_time = (
            np.asarray(run.position_time) if run.position_time is not None else None
        )
        if position is None or position_time is None or position.ndim != 1:
            self._position_cache[self._active_run_name] = None
            return None
        decoder_time = np.asarray(run.results["time"].values)
        # ``np.interp`` clips to the position-time bounds — the head
        # and tail of the decoder grid get the edge position values
        # rather than NaN, which matches statespacecheck-paper-viewer.
        interpolated = np.interp(
            decoder_time, position_time.astype(float), position.astype(float)
        ).astype(np.float32)
        self._position_cache[self._active_run_name] = interpolated
        return interpolated

    _SLICE_VAR_MAP = {
        "posterior": "acausal_posterior",
        "acausal": "acausal_posterior",
        "likelihood": "log_likelihood",
        "predictive": "predictive_posterior",
    }

    def slice_at_index(
        self,
        t_idx: int,
        which: Literal[
            "posterior", "likelihood", "predictive", "acausal"
        ] = "posterior",
    ) -> np.ndarray | None:
        """Single-row slice for the SlicePanel's per-bin readout."""
        var = self._SLICE_VAR_MAP[which]
        if var not in self.active_run.results:
            if which == "likelihood":
                raise KeyError("Active run has no `log_likelihood` in results.")
            return None
        return np.asarray(self.active_run.results[var].isel(time=t_idx).values)

    def events_in_window(self, _sl: slice) -> list[EventOverlay]:
        """Return the active run's overlays.

        Returned unchanged; the panel renderer maps absolute time to
        x-coordinates and decides what to draw. The ``_sl`` argument
        is accepted to match the future signature when window-clipping
        moves into the data source.
        """
        return list(self.active_run.event_overlays)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def time(self) -> np.ndarray:
        """Active run's time grid (shared across runs after validation)."""
        return np.asarray(self.active_run.results["time"].values)

    @property
    def n_time(self) -> int:
        return int(self.active_run.results.sizes["time"])

    @property
    def results(self) -> xr.Dataset:
        return self.active_run.results
