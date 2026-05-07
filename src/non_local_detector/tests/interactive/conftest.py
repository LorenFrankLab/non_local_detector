"""Session-scoped fixtures for the interactive viewer test suite.

Provides:

- ``sim_session`` — Track 0 simulated session.
- ``nl_fitted`` / ``cf_fitted`` / ``nsf_fitted`` / ``dec_fitted`` —
  the four v1 detectors, fit once each via ``estimate_parameters``
  with the EM result discarded.
- ``run_bundles`` — the 4 × 3 matrix (12 entries) of named
  ``RunBundle`` variants, keyed ``<det>_<variant>`` (e.g.
  ``"nl_default"``, ``"nl_loglik"``, ``"nl_all"``).
- ``multi_run_bundles`` — a four-entry dict suitable for the
  Phase 6 model-swap test (``{"nl": ..., "cf": ..., "nsf": ...,
  "dec": ...}``); each value is the ``"all"`` variant so the swap
  exercises every panel.
"""

from __future__ import annotations

import pytest
import xarray as xr

from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
    fit_cf_detector,
    fit_dec_detector,
    fit_nl_detector,
    fit_nsf_detector,
    make_session,
    predict_variants,
)
from non_local_detector.visualization.interactive.view_models.base import RunBundle


@pytest.fixture(scope="session")
def sim_session() -> SimulatedSession:
    return make_session()


@pytest.fixture(scope="session")
def nl_fitted(sim_session: SimulatedSession) -> FittedDetector:
    return fit_nl_detector(sim_session)


@pytest.fixture(scope="session")
def cf_fitted(sim_session: SimulatedSession) -> FittedDetector:
    return fit_cf_detector(sim_session)


@pytest.fixture(scope="session")
def nsf_fitted(sim_session: SimulatedSession) -> FittedDetector:
    return fit_nsf_detector(sim_session)


@pytest.fixture(scope="session")
def dec_fitted(sim_session: SimulatedSession) -> FittedDetector:
    return fit_dec_detector(sim_session)


@pytest.fixture(scope="session")
def nl_singleton_fitted(sim_session: SimulatedSession) -> FittedDetector:
    """``NonLocalSortedSpikesDetector(local_position_std=None)``.

    Fully-singleton schema (``bin_sizes_=[1, 1, n_pos, n_pos]``) for
    the Phase 1c singleton-Local follow-up tests. Tests using this
    fixture should be marked ``@pytest.mark.slow`` — the discrete-Local
    EM is substantially slower than the continuous-Gaussian variant.
    """
    return fit_nl_detector(sim_session, local_position_std=None)


def _build_bundles(
    det_key: str,
    fitted: FittedDetector,
    session: SimulatedSession,
) -> dict[str, RunBundle]:
    """Return ``{<det_key>_<variant>: RunBundle}`` for a single detector."""
    variants: dict[str, xr.Dataset] = predict_variants(fitted, session)
    return {
        f"{det_key}_{variant_name}": RunBundle(
            results=results,
            detector=fitted.detector,
            spike_times=session.spike_times,
            position_time=session.time,
            position=session.position,
            speed=session.speed,
        )
        for variant_name, results in variants.items()
    }


@pytest.fixture(scope="session")
def run_bundles(
    sim_session: SimulatedSession,
    nl_fitted: FittedDetector,
    cf_fitted: FittedDetector,
    nsf_fitted: FittedDetector,
    dec_fitted: FittedDetector,
) -> dict[str, RunBundle]:
    """The full 4 × 3 matrix of (detector, predict variant) RunBundles."""
    bundles: dict[str, RunBundle] = {}
    for key, fitted in (
        ("nl", nl_fitted),
        ("cf", cf_fitted),
        ("nsf", nsf_fitted),
        ("dec", dec_fitted),
    ):
        bundles.update(_build_bundles(key, fitted, sim_session))
    return bundles


@pytest.fixture(scope="session")
def multi_run_bundles(
    run_bundles: dict[str, RunBundle],
) -> dict[str, RunBundle]:
    """Four-entry dict for the model-swap test, keyed by detector code."""
    return {
        "nl": run_bundles["nl_all"],
        "cf": run_bundles["cf_all"],
        "nsf": run_bundles["nsf_all"],
        "dec": run_bundles["dec_all"],
    }


@pytest.fixture(autouse=True)
def _clear_qt_viewer_registry():
    """Close + delete every Qt top-level widget after each test.

    Two paths leave widgets behind:

    1. ``launch_qt(block=False)`` registers in ``viewer.qt._LIVE_VIEWERS``
       so PySide6 doesn't GC non-blocking windows; close those
       explicitly.
    2. Tests that construct ``QtViewer(...)`` directly hold a
       Python local. When the local goes out of scope the GC path
       eventually destroys the C++ widget, but ``cumulative`` Qt
       state (graphics scenes, deferred deletions, font caches
       reachable from undeleted ViewBoxMenu instances) builds up
       across tests and can crash pyqtgraph's ``ViewBoxMenu.setupUi``
       around the 14th–17th constructed viewer. The crash trace
       points at ``axisCtrlTemplate_generic.py``; clearing every
       ``topLevelWidget`` and running a Python GC pass between
       tests keeps the per-test working set bounded.

    Autouse, but no-op if the ``[viewer]`` extra isn't installed.
    """
    import gc

    yield
    try:
        from PySide6 import QtWidgets

        from non_local_detector.visualization.interactive.viewer import (
            qt as qt_mod,
        )
    except ImportError:
        return  # [viewer] extra not installed; nothing to clean.
    for viewer in list(qt_mod._LIVE_VIEWERS):
        viewer.close()
        viewer.deleteLater()
    qt_mod._LIVE_VIEWERS.clear()
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    # Close + deleteLater every other top-level widget the test left
    # behind (e.g. ``QtViewer(ds)`` constructed without going through
    # ``launch_qt``). ``WA_DeleteOnClose`` is set on QtViewer; for
    # other widgets close() is still safe and the Python GC pass
    # below releases C++ ownership.
    for widget in list(app.topLevelWidgets()):
        if widget.isVisible() or not widget.isHidden():
            widget.close()
        widget.deleteLater()
    # processEvents drains queued ``deleteLater`` calls; double pass
    # because some destructions schedule further deferred deletions.
    app.processEvents()
    app.processEvents()
    gc.collect()
    app.processEvents()
