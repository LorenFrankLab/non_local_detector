"""Session-scoped fitted-detector fixtures for the Phase 1a analysis tests.

Each fixture fits one Track 0 simulated detector exactly once per test
session. Heavy lifting lives in
``non_local_detector.tests._simulated_detectors``.
"""

from __future__ import annotations

import pytest

from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
    fit_cf_detector,
    fit_dec_detector,
    fit_nl_detector,
    fit_nsf_detector,
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


@pytest.fixture(scope="session")
def cf_fitted(sim_session: SimulatedSession) -> FittedDetector:
    """``ContFragSortedSpikesClassifier`` fit + results (rectangular)."""
    return fit_cf_detector(sim_session)


@pytest.fixture(scope="session")
def nsf_fitted(sim_session: SimulatedSession) -> FittedDetector:
    """``NoSpikeContFragSortedSpikesClassifier`` fit + results."""
    return fit_nsf_detector(sim_session)


@pytest.fixture(scope="session")
def dec_fitted(sim_session: SimulatedSession) -> FittedDetector:
    """``SortedSpikesDecoder`` fit + results (single-state)."""
    return fit_dec_detector(sim_session)
