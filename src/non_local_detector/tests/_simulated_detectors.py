"""Shared helpers for fitting Track 0 simulated detectors.

These helpers are imported by conftest fixtures in tests/analysis/,
tests/visualization/, and tests/interactive/. Keeping the fit logic in
one module means each detector configuration only has to be specified
once; the conftests just declare session-scoped fixture wrappers.

Detectors covered (all four v1 viewer detectors):

- ``NonLocalSortedSpikesDetector(local_position_std=1.0)`` — Track 0
  default, ``bin_sizes_=[n_pos, 1, n_pos, n_pos]``.
- ``ContFragSortedSpikesClassifier()`` — rectangular,
  ``bin_sizes_=[n_pos, n_pos]``.
- ``NoSpikeContFragSortedSpikesClassifier()`` —
  ``bin_sizes_=[1, n_pos, n_pos]``.
- ``SortedSpikesDecoder()`` — single state, ``bin_sizes_=[n_pos]``.

Each helper fits exactly once and returns ``(detector, results)`` where
``results`` is the dataset returned by ``estimate_parameters`` with
``store_log_likelihood=True``. That is sufficient for Phase 1a tests
(which only need the fitted detector + an ``acausal_posterior`` row);
Phase 1b will extend this with explicit ``predict()`` variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from non_local_detector import (
    ContFragSortedSpikesClassifier,
    NonLocalSortedSpikesDetector,
    NoSpikeContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

SEED = 0
N_NEURONS = 25


@dataclass
class SimulatedSession:
    """Container for the simulated session arrays."""

    speed: np.ndarray
    position: np.ndarray
    spike_times: list[np.ndarray]
    time: np.ndarray
    event_times: np.ndarray
    sampling_frequency: int
    is_event: np.ndarray
    place_fields: np.ndarray


def make_session(seed: int = SEED, n_neurons: int = N_NEURONS) -> SimulatedSession:
    """Generate the canonical simulated session via ``make_simulated_data``."""
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=n_neurons, seed=seed)
    return SimulatedSession(
        speed=speed,
        position=position,
        spike_times=spike_times,
        time=time,
        event_times=event_times,
        sampling_frequency=sampling_frequency,
        is_event=is_event,
        place_fields=place_fields,
    )


@dataclass
class FittedDetector:
    """Container for a fitted detector + its in-fit results dataset."""

    detector: object  # one of the four v1 sorted-spikes detector classes
    results: xr.Dataset


def _fit(detector: object, session: SimulatedSession) -> FittedDetector:
    """Fit ``detector`` against ``session`` and return ``(detector, results)``.

    Uses ``return_outputs="log_likelihood"`` so ``results`` carries
    ``log_likelihood`` for the collapse-helper tests in Phase 1a.
    """
    results = detector.estimate_parameters(  # type: ignore[attr-defined]
        position_time=session.time,
        position=session.position,
        spike_times=session.spike_times,
        is_training=~session.is_event,
        time=session.time,
        return_outputs="log_likelihood",
    )
    return FittedDetector(detector=detector, results=results)


def fit_nl_detector(
    session: SimulatedSession, *, local_position_std: float | None = 1.0
) -> FittedDetector:
    """Fit a ``NonLocalSortedSpikesDetector``.

    Defaults to ``local_position_std=1.0`` (Track 0 fixture). Pass
    ``None`` to exercise the singleton-Local schema (slow).
    """
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        non_local_position_penalty=1.0,
        non_local_penalty_std=5.0,
        local_position_std=local_position_std,
    )
    return _fit(detector, session)


def fit_cf_detector(session: SimulatedSession) -> FittedDetector:
    """Fit a ``ContFragSortedSpikesClassifier`` (rectangular schema)."""
    detector = ContFragSortedSpikesClassifier(
        sorted_spikes_algorithm="sorted_spikes_kde",
    )
    return _fit(detector, session)


def fit_nsf_detector(session: SimulatedSession) -> FittedDetector:
    """Fit a ``NoSpikeContFragSortedSpikesClassifier``."""
    detector = NoSpikeContFragSortedSpikesClassifier(
        sorted_spikes_algorithm="sorted_spikes_kde",
    )
    return _fit(detector, session)


def fit_dec_detector(session: SimulatedSession) -> FittedDetector:
    """Fit a ``SortedSpikesDecoder`` (single-state schema)."""
    detector = SortedSpikesDecoder(sorted_spikes_algorithm="sorted_spikes_kde")
    return _fit(detector, session)


# ---------------------------------------------------------------------------
# Predict-variant matrix (Phase 1b: 4 detectors × 3 return_outputs)
# ---------------------------------------------------------------------------

# Predict variants the Phase 1b interactive fixture exposes. Each maps
# to a ``return_outputs`` value passed to ``detector.predict(...)``. The
# resulting ``results`` shapes are:
#
# - default → only acausal_posterior + acausal_state_probabilities.
# - loglik  → + log_likelihood.
# - all     → + filter (causal_*), predictive (predictive_state_probabilities,
#             predictive_posterior), log_likelihood.
PREDICT_VARIANTS: dict[str, str | list[str] | None] = {
    "default": None,
    "loglik": ["log_likelihood"],
    "all": "all",
}


def first_finite_row_index(values: np.ndarray) -> int:
    """Return the first time index with at least one finite value in any column.

    Many tests need to skip the initial-burn-in NaN rows in a
    decoder's ``acausal_posterior`` / ``log_likelihood`` array before
    asserting on a representative row. Centralised here so the
    selection rule is consistent across tests.
    """
    finite_indices = np.flatnonzero(np.isfinite(values).any(axis=-1))
    assert finite_indices.size > 0, "No finite rows in supplied array"
    return int(finite_indices[0])


def predict_variants(
    fitted: FittedDetector, session: SimulatedSession
) -> dict[str, xr.Dataset]:
    """Run ``predict()`` once per variant against the post-fit detector.

    The ``estimate_parameters``-returned dataset is *not* reused —
    its parameters are one M-step behind the post-fit detector state.
    Calling ``predict`` after ``estimate_parameters`` produces results
    consistent with the parameters the detector currently holds.
    """
    variants: dict[str, xr.Dataset] = {}
    for var_name, return_outputs in PREDICT_VARIANTS.items():
        results = fitted.detector.predict(  # type: ignore[attr-defined]
            spike_times=session.spike_times,
            time=session.time,
            position=session.position,
            position_time=session.time,
            return_outputs=return_outputs,
        )
        variants[var_name] = results
    return variants
