"""Session-scoped fitted-detector fixtures for the Phase 1a visualization tests.

The static-plot refactor test only needs the NL detector (it's the
only detector ``plot_non_local_model`` consumes).
"""

from __future__ import annotations

import pytest

from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
    fit_nl_detector,
    make_session,
)


@pytest.fixture(scope="session")
def sim_session() -> SimulatedSession:
    """Canonical Track 0 simulated session (n_neurons=25, seed=0)."""
    return make_session()


@pytest.fixture(scope="session")
def nl_fitted(sim_session: SimulatedSession) -> FittedDetector:
    """``NonLocalSortedSpikesDetector(local_position_std=1.0)`` fit + results."""
    return fit_nl_detector(sim_session)
