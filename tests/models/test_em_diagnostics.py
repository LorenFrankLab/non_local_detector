"""An ``estimate_parameters`` run's diagnostics belong to that run.

``n_iter_``, ``converged_``, ``em_monotonicity_violations_`` and
``degenerate_timesteps_`` read one object, which a new ``fit`` removes. Stored
detectors keep these values however they were saved.
"""

import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from non_local_detector import (
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import NotFittedError
from non_local_detector.models._em import _EMDiagnostics
from non_local_detector.models.base import (
    ClusterlessDetector,
    SortedSpikesDetector,
    _ModelUnpickler,
)
from tests.models._helpers import SORTED_SIM_RATE, _assert_unchanged, _snapshot

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:EM did not converge after max_iter:UserWarning"),
]

COMPAT_DATA = Path(__file__).parents[1] / "compat_data"
DIAGNOSTICS = ("n_iter_", "converged_", "em_monotonicity_violations_")
ALL_DIAGNOSTICS = (*DIAGNOSTICS, "degenerate_timesteps_")


def _detector():
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
        sampling_frequency=SORTED_SIM_RATE,
    )


def _estimate(detector, sorted_sim, *, max_iter=2):
    time, position, spike_times = sorted_sim
    detector.estimate_parameters(
        time,
        position,
        spike_times,
        time_edges=time_edges_from_centers(time[:200]),
        max_iter=max_iter,
    )
    return detector


def _posterior(detector, sorted_sim):
    time, position, spike_times = sorted_sim
    return detector.predict(
        spike_times=spike_times,
        time_edges=time_edges_from_centers(time[:50]),
        position=position,
        position_time=time,
    ).acausal_posterior.values


@pytest.fixture(scope="module")
def estimated(sorted_sim):
    return _estimate(_detector(), sorted_sim)


def test_estimation_reports_its_run(estimated):
    assert estimated.n_iter_ == 2
    assert estimated.converged_ is False
    assert estimated.em_monotonicity_violations_ == []
    assert estimated.degenerate_timesteps_.size == 0
    assert estimated.degenerate_timesteps_.dtype.kind == "i"


def test_fit_after_estimation_looks_like_a_fresh_fit(estimated, sorted_sim):
    time, position, spike_times = sorted_sim
    refit = estimated.copy().fit(time, position, spike_times)
    fresh = _detector().fit(time, position, spike_times)

    assert sorted(vars(refit)) == sorted(vars(fresh))
    for name in (*ALL_DIAGNOSTICS, "discrete_initial_conditions_"):
        assert not hasattr(refit, name), name
    np.testing.assert_array_equal(
        refit.discrete_state_transitions_, fresh.discrete_state_transitions_
    )
    np.testing.assert_array_equal(refit.initial_conditions_, fresh.initial_conditions_)
    np.testing.assert_array_equal(
        _posterior(refit, sorted_sim), _posterior(fresh, sorted_sim)
    )


def test_reading_a_diagnostic_of_a_plain_fit_names_estimation(sorted_sim):
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    with pytest.raises(NotFittedError, match=r"n_iter_.*estimate_parameters"):
        _ = detector.n_iter_


@pytest.mark.parametrize("name", ALL_DIAGNOSTICS)
def test_diagnostics_are_read_only(estimated, name):
    with pytest.raises(AttributeError):
        setattr(estimated, name, getattr(estimated, name))


def test_list_a_caller_edits_is_not_the_run_s(estimated):
    estimated.em_monotonicity_violations_.append(99)
    assert estimated.em_monotonicity_violations_ == []


def test_a_second_estimation_replaces_the_first(sorted_sim):
    detector = _estimate(_detector(), sorted_sim, max_iter=1)
    assert detector.n_iter_ == 1
    _estimate(detector, sorted_sim, max_iter=2)
    assert detector.n_iter_ == 2


def test_failed_estimation_keeps_the_run_before_it(estimated, sorted_sim, monkeypatch):
    detector = estimated.copy()
    snapshot = _snapshot(detector)

    def failing_step(self, *args, **kwargs):
        msg = "EM step failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(type(detector), "_em_final_e_step", failing_step)
    with pytest.raises(RuntimeError, match="EM step failed"):
        _estimate(detector, sorted_sim)

    _assert_unchanged(detector, snapshot)
    assert detector.n_iter_ == estimated.n_iter_


@pytest.mark.parametrize("transport", ["copy", "pickle", "nld"])
def test_diagnostics_survive_copying_and_saving(
    estimated, sorted_sim, tmp_path, transport
):
    if transport == "copy":
        loaded = estimated.copy()
    elif transport == "pickle":
        loaded = pickle.loads(pickle.dumps(estimated))
    else:
        path = tmp_path / "model.nld"
        estimated.save_model(path)
        loaded = NonLocalSortedSpikesDetector.load_model(path)

    for name in DIAGNOSTICS:
        assert getattr(loaded, name) == getattr(estimated, name), name
        assert type(getattr(loaded, name)) is type(getattr(estimated, name)), name
    np.testing.assert_array_equal(
        loaded.degenerate_timesteps_, estimated.degenerate_timesteps_
    )
    np.testing.assert_array_equal(
        loaded.discrete_initial_conditions_, estimated.discrete_initial_conditions_
    )
    assert not set(ALL_DIAGNOSTICS) & set(vars(loaded))


def test_saved_file_of_a_plain_fit_has_no_diagnostics(sorted_sim, tmp_path):
    time, position, spike_times = sorted_sim
    path = tmp_path / "model.nld"
    _detector().fit(time, position, spike_times).save_model(path)
    with zipfile.ZipFile(path) as archive:
        state = json.loads(archive.read("state.json"))

    assert not {"n_iter", "converged", "em_monotonicity_violations"} & set(state)
    assert not hasattr(NonLocalSortedSpikesDetector.load_model(path), "n_iter_")


@pytest.mark.parametrize(
    "path",
    [
        path
        for version in ("main-66047901", "main-d5f85823")
        for path in sorted((COMPAT_DATA / version).glob("*_model.pkl"))
    ],
    ids=lambda path: f"{path.parent.name}-{path.stem}",
)
def test_old_pickle_keeps_its_diagnostics(path):
    """Real pickles hold the attributes in the detector's own dict."""
    with path.open("rb") as file:
        detector = _ModelUnpickler(file).load()

    assert detector.n_iter_ == 2
    assert isinstance(detector.converged_, bool)
    assert detector.em_monotonicity_violations_ == []
    assert detector.degenerate_timesteps_.size == 0
    assert not set(ALL_DIAGNOSTICS) & set(vars(detector))
    assert isinstance(vars(detector)["_em_diagnostics_"], _EMDiagnostics)


def test_old_pickle_with_some_diagnostics_loads(sorted_sim):
    """An earlier release saved only some of them; the rest are empty."""
    detector = _estimate(_detector(), sorted_sim, max_iter=1)
    state = detector.__getstate__()
    del state["_em_diagnostics_"]
    state["n_iter_"] = 3
    state["converged_"] = True

    loaded = type(detector).__new__(type(detector))
    loaded.__setstate__(state)

    assert loaded.n_iter_ == 3
    assert loaded.converged_ is True
    assert loaded.em_monotonicity_violations_ == []
    assert loaded.degenerate_timesteps_.size == 0


def test_old_pickle_of_a_plain_fit_has_no_diagnostics():
    path = COMPAT_DATA / "v0.6.9" / "nonlocal_sorted_1d_model.pkl"
    with path.open("rb") as file:
        detector = _ModelUnpickler(file).load()
    assert not hasattr(detector, "n_iter_")
    assert "_em_diagnostics_" not in vars(detector)


@pytest.mark.parametrize("name", ["nonlocal_sorted_1d", "nonlocal_clusterless_2d"])
def test_frozen_saved_file_keeps_its_diagnostics(name):
    path = COMPAT_DATA / "format_v1" / f"{name}.nld"
    with zipfile.ZipFile(path) as archive:
        state = json.loads(archive.read("state.json"))
    family = SortedSpikesDetector if "sorted" in name else ClusterlessDetector

    model = family.load_model(path)

    if "n_iter" not in state:
        assert not hasattr(model, "n_iter_")
        return
    assert model.n_iter_ == state["n_iter"]
    assert model.converged_ is state["converged"]
    assert model.em_monotonicity_violations_ == state["em_monotonicity_violations"]
    assert model.degenerate_timesteps_.dtype.kind == "i"


def test_decoder_reports_its_run_too(sorted_sim):
    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
        sampling_frequency=SORTED_SIM_RATE,
    )
    _estimate(decoder, sorted_sim, max_iter=1)
    assert decoder.n_iter_ == 1
