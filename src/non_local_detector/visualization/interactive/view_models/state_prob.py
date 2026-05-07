"""``StateProbabilityModel`` — multi-line over discrete states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


class StateProbabilityModel:
    """Stack ``acausal_state_probabilities`` rows for the visible window.

    Output is the input window unchanged (already shaped
    ``(n_visible, n_states)``) plus the active detector's
    ``state_names`` so the panel can label each line. Per-state colors
    are picked by the panel.
    """

    def __init__(self, detector: _DetectorBase) -> None:
        self._bind(detector)

    def _bind(self, detector: _DetectorBase) -> None:
        self._detector = detector
        self._state_names = list(detector.state_names)

    @property
    def detector(self) -> _DetectorBase:
        return self._detector

    @property
    def state_names(self) -> list[str]:
        return self._state_names

    def set_active_run(self, detector: _DetectorBase) -> None:
        """Rebind to a new detector schema."""
        self._bind(detector)

    def update_window(self, state_probs_window: np.ndarray) -> np.ndarray:
        """Validate + return ``(n_visible, n_states)`` for the panel.

        State-prob window comes through unchanged; the panel reads
        ``state_names`` from the model directly.
        """
        if state_probs_window.ndim != 2:
            raise ValueError(
                "StateProbabilityModel expects a 2D window "
                f"(n_visible, n_states). Got shape "
                f"{state_probs_window.shape}."
            )
        n_states = len(self._state_names)
        if state_probs_window.shape[1] != n_states:
            raise ValueError(
                f"StateProbabilityModel expected {n_states} states (per "
                f"detector.state_names), got "
                f"{state_probs_window.shape[1]} columns."
            )
        return state_probs_window
