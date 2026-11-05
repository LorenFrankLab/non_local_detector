"""Artifacts Spyglass stored with earlier non_local_detector versions.

Spyglass keeps three things from a decode: the ``DecodingParameters`` row it
built the detector from, the fitted detector (``save_model``) and the results
(``save_results``). Later it reloads the detector and the results to inspect
them, and rebuilds detectors from the rows to fit again or to recompute
environments. ``tests/compat_data`` holds these artifacts as the real 0.6.0,
0.6.9 and recent-main versions wrote them (see ``generate_compat_artifacts.py``
there for the exact versions and how each was made).

Loading and inspecting every artifact must work. Decoding with a detector
fitted before rates were recorded in Hz must be refused with a refit request,
because its rates and transition clocks cannot be converted safely.
"""

import copy
import copyreg
import dataclasses
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from track_linearization import make_track_graph

import non_local_detector as nld
from non_local_detector import analysis
from non_local_detector import continuous_state_transitions as cst
from non_local_detector import discrete_state_transitions as dst
from non_local_detector import initial_conditions as ic
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.models.base import (
    ClusterlessDetector,
    SortedSpikesDetector,
    _ModelUnpickler,
)
from non_local_detector.observation_models import ObservationModel
from tests.models._helpers import type_violations

COMPAT_DATA = Path(__file__).parents[1] / "compat_data"
# Fitted before rates were recorded in Hz: refused at decode.
LEGACY_VERSIONS = ("v0.5.0", "v0.5.12", "v0.5.13", "v0.6.0", "v0.6.9", "main-84259d49")
VERSIONS = (*LEGACY_VERSIONS, "main-66047901", "main-d5f85823")
MODELS = [
    (version, path.name.removesuffix("_model.pkl"))
    for version in VERSIONS
    for path in sorted((COMPAT_DATA / version).glob("*_model.pkl"))
]
# Every saved result file: netCDF4 as Spyglass saves it, and one netCDF3 file
# per version as an environment without netCDF4 saves it.
RESULTS = [
    (version, path.name)
    for version in VERSIONS
    for path in sorted((COMPAT_DATA / version).glob("*_results*.nc"))
]
ROWS = [
    (version, preset, row_format)
    for version in VERSIONS
    for preset, formats in pickle.loads(
        (COMPAT_DATA / version / "decoding_parameters.pkl").read_bytes()
    ).items()
    for row_format in formats
]

pytestmark = [
    pytest.mark.integration,
    # NumPy 1.x pickles name numpy.core, which NumPy 2 reads with a warning.
    pytest.mark.filterwarnings(
        "ignore:numpy.core.numeric is deprecated:DeprecationWarning"
    ),
    # Spyglass in 2024 labelled its transition matrix ("states", "states").
    pytest.mark.filterwarnings("ignore:Duplicate dimension names:UserWarning"),
]


@pytest.fixture(scope="module")
def inputs():
    """The recording every artifact was fitted on (time in s, positions in cm)."""
    return pickle.loads((COMPAT_DATA / "inputs.pkl").read_bytes())


def _load_model(version, artifact):
    """Load a stored model as Spyglass does, with its family's detector class."""
    return _family(artifact).load_model(COMPAT_DATA / version / f"{artifact}_model.pkl")


def _family(artifact):
    return ClusterlessDetector if "clusterless" in artifact else SortedSpikesDetector


def _load_results(version, artifact):
    return SortedSpikesDetector.load_results(
        COMPAT_DATA / version / f"{artifact}_results.nc"
    )


def _row(version, preset, row_format):
    rows = pickle.loads(
        (COMPAT_DATA / version / "decoding_parameters.pkl").read_bytes()
    )
    return rows[preset][row_format]


def _recording(inputs, clusterless):
    """Position and spikes of the artifact's family."""
    if clusterless:
        return inputs["field_position"], {
            "spike_times": inputs["clusterless_spike_times"],
            "spike_waveform_features": inputs["clusterless_features"],
        }
    return inputs["track_position"], {"spike_times": inputs["sorted_spike_times"]}


def _spikes_in(spikes, time_edges):
    """The spikes (and their features) within the decode range."""
    keep = [(t >= time_edges[0]) & (t < time_edges[-1]) for t in spikes["spike_times"]]
    return {
        key: [value[k] for value, k in zip(values, keep, strict=True)]
        for key, values in spikes.items()
    }


def _spyglass_restore(row):
    """Rebuild a detector from a stored row as Spyglass does.

    Mirrors ``restore_classes`` and the ``make()`` construction of Spyglass PR
    #1618 (334d1a6b8, ``spyglass.decoding.v1.dj_decoder_conversion``): only an
    environment's configuration is kept, transitions, initial conditions and
    observation models are rebuilt as ``Class(**stored_attributes)``, and a row
    is rebuilt with its stored class, or for a ``vars(model)`` row without one,
    with the base detector, or the NonLocal detector if only that accepts the
    row's parameters.
    """
    row = copy.deepcopy(row)
    config = Environment().get_params().keys()
    environments = []
    for env in row["environments"]:
        env = {key: value for key, value in env.items() if key in config}
        if env.get("track_graph") is not None:
            env["track_graph"] = make_track_graph(**env["track_graph"])
        environments.append(Environment(**env))
    row["environments"] = environments

    def build(stored, module):
        stored = dict(stored)
        return getattr(module, stored.pop("class_name"))(**stored)

    row["continuous_transition_types"] = [
        [build(t, cst) for t in from_state]
        for from_state in row["continuous_transition_types"]
    ]
    row["discrete_transition_type"] = build(row["discrete_transition_type"], dst)
    row["continuous_initial_conditions_types"] = [
        build(c, ic) for c in row["continuous_initial_conditions_types"]
    ]
    if row["observation_models"] is not None:
        row["observation_models"] = [
            ObservationModel(**obs) for obs in row["observation_models"]
        ]
    class_name = row.pop("class_name", None)
    if class_name is not None:
        return getattr(nld, class_name)(**row)
    row = {key: value for key, value in row.items() if not key.startswith("_")}
    clusterless = "clusterless_algorithm" in row
    base = ClusterlessDetector if clusterless else SortedSpikesDetector
    try:
        return base(**row)
    except TypeError:
        nonlocal_name = (
            "NonLocalClusterlessDetector"
            if clusterless
            else "NonLocalSortedSpikesDetector"
        )
        return getattr(nld, nonlocal_name)(**row)


@pytest.mark.parametrize(("version", "artifact"), MODELS)
def test_saved_model_loads_with_the_attributes_spyglass_reads(version, artifact):
    """``fetch_model`` and the reads of its callers (Spyglass sorted_spikes.py,
    clusterless.py and decoding_merge.py)."""
    model = _load_model(version, artifact)

    clusterless = "clusterless" in artifact
    assert isinstance(
        model, ClusterlessDetector if clusterless else SortedSpikesDetector
    )
    environment = model.environments[0]
    assert (environment.track_graph is not None) == ("1d" in artifact)
    interior_centers = environment.place_bin_centers_[
        environment.is_track_interior_.ravel(order="C")
    ]
    assert interior_centers.shape[0] > 0
    if not clusterless:  # spike_times_sorted_by_place_field_peak
        for key in model.encoding_model_:
            place_fields = np.asarray(model.encoding_model_[key]["place_fields"])
            order = np.argsort(np.nanargmax(place_fields, axis=1).squeeze())
            assert sorted(order) == list(range(place_fields.shape[0]))


@pytest.mark.parametrize(("version", "artifact"), MODELS)
def test_saved_model_keeps_its_fitted_environments(version, artifact):
    """These versions kept a detector's fitted environments in its
    ``environments`` parameter. A loaded model keeps them there, where code
    that inspects old models reads them (``get_params()`` reports them, as it
    always did), and ``environments_`` holds the same objects. Like every
    detector's, ``environments`` is a tuple, so ``clone`` rebuilds it."""
    model = _load_model(version, artifact)

    assert isinstance(model.environments, tuple)
    assert len(model.environments_) == len(model.environments) == 1
    for configured, fitted in zip(model.environments, model.environments_, strict=True):
        assert fitted is configured
        assert fitted.place_bin_centers_ is not None
    for obs in model.observation_models:
        assert model._get_environment_by_name(obs.environment_name) in (
            model.environments_
        )
    cloned = clone(model)
    assert not hasattr(cloned, "environments_")
    assert not cloned.environments[0]._is_fitted


class _StoredState:
    """An object unpickled without its class: keeps the pickled state."""

    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class _StoredStateUnpickler(_ModelUnpickler):
    """Unpickle detectors and environments as their stored states."""

    def find_class(self, module, name):
        if module.startswith("non_local_detector.models") or (
            module,
            name,
        ) == ("non_local_detector.environment", "Environment"):
            return _StoredState
        return super().find_class(module, name)


class _LegacyEnvironmentPickle:
    """Pickles as an ``Environment`` with the given instance ``__dict__``, as
    earlier releases did: unpickling makes an ``Environment`` without calling
    its constructor and passes the state to ``__setstate__``."""

    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        return (copyreg._reconstructor, (Environment, object, None), self.state)


def _stored_environment_states(version, artifact):
    path = COMPAT_DATA / version / f"{artifact}_model.pkl"
    with path.open("rb") as file:
        detector = _StoredStateUnpickler(file).load()
    return [environment.state for environment in detector.state["environments"]]


@pytest.mark.parametrize(("version", "artifact"), MODELS)
def test_environment_pickled_with_fitted_fields_unpickles_into_a_place_grid(
    version, artifact, tmp_path
):
    """``save_environment`` files of these versions hold the fitted attributes
    as instance fields. They load into the current form: the same values,
    read through ``place_grid_``, fitted with the stored configuration."""
    (state,) = _stored_environment_states(version, artifact)
    path = tmp_path / "environment.pkl"
    path.write_bytes(pickle.dumps(_LegacyEnvironmentPickle(state)))

    environment = Environment.load_environment(str(path))

    assert environment._is_fitted
    assert "place_bin_centers_" not in vars(environment)
    assert environment.get_params().keys() == Environment().get_params().keys()
    np.testing.assert_array_equal(
        environment.place_bin_centers_, state["place_bin_centers_"]
    )
    np.testing.assert_array_equal(
        environment.is_track_interior_, state["is_track_interior_"]
    )
    assert environment.centers_shape_ == tuple(state["centers_shape_"])
    for edge, stored in zip(environment.edges_, state["edges_"], strict=True):
        np.testing.assert_array_equal(edge, stored)
    assert environment.distance_between_nodes_ is not None
    centers = environment.place_bin_centers_[environment.is_track_interior_flat_]
    assert environment.get_bin_ind(centers[:3]).shape == (3,)
    restored = pickle.loads(pickle.dumps(environment))
    np.testing.assert_array_equal(
        restored.place_bin_centers_, environment.place_bin_centers_
    )


@pytest.mark.parametrize(
    ("version", "artifact"), [(v, a) for v, a in MODELS if "2d" in a]
)
def test_open_field_pickled_before_its_distances_were_stored_rebuilds_them(
    version, artifact
):
    """Releases before 0.5.13 stored no ``distance_between_nodes_`` or
    ``track_graphDD`` for an open field. Without them, the stored grid loads
    with the graph and distances a fit of it gives, equal to those these
    versions stored."""
    (state,) = _stored_environment_states(version, artifact)
    stored = pickle.loads(pickle.dumps(_LegacyEnvironmentPickle(state)))
    stripped = {
        name: value
        for name, value in state.items()
        if name not in ("distance_between_nodes_", "track_graphDD")
    }

    environment = pickle.loads(pickle.dumps(_LegacyEnvironmentPickle(stripped)))

    expected = stored.distance_between_nodes_
    if isinstance(expected, LazyGraphDistances):
        expected = expected.to_dense()
    np.testing.assert_array_equal(environment.distance_between_nodes_, expected)
    assert set(environment.track_graphDD.edges) == set(stored.track_graphDD.edges)
    assert environment.infer_track_interior_ == state.get(
        "infer_track_interior_", state["infer_track_interior"]
    )


def test_open_field_saved_by_0_5_12_gets_the_distances_of_a_fresh_fit(inputs):
    """0.5.12 stored no graph or distances for an open field (0.5.13 was the
    first to). The rebuilt ones equal those of fitting the stored
    configuration on the recording, and those 0.5.13 stored for it."""
    environment = _load_model("v0.5.12", "nonlocal_clusterless_2d").environments[0]
    (state,) = _stored_environment_states("v0.5.12", "nonlocal_clusterless_2d")
    assert "distance_between_nodes_" not in state
    fresh = Environment(**environment.get_params()).fit_place_grid(
        inputs["field_position"],
        infer_track_interior=environment.infer_track_interior_,
    )
    stored_by_0_5_13 = _load_model("v0.5.13", "nonlocal_clusterless_2d").environments[0]

    np.testing.assert_array_equal(
        environment.place_bin_centers_, fresh.place_bin_centers_
    )
    for expected in (fresh, stored_by_0_5_13):
        np.testing.assert_array_equal(
            environment.distance_between_nodes_, expected.distance_between_nodes_
        )


UNFILTERED_ROWS = [row for row in ROWS if row[0] in ("main-84259d49", "main-66047901")]


@pytest.mark.parametrize(("version", "preset", "row_format"), UNFILTERED_ROWS)
def test_environment_rows_holding_unset_fitted_keys_construct_unfiltered(
    version, preset, row_format
):
    """Rows of main (2025-10 to 2026-10) hold every fitted attribute as None,
    and Spyglass ``master`` passes them to ``Environment`` unfiltered: they
    construct, with a deprecation warning naming the dropped keys."""
    config = Environment().get_params().keys()
    for stored in _row(version, preset, row_format)["environments"]:
        stored = dict(stored)
        if stored.get("track_graph") is not None:
            stored["track_graph"] = make_track_graph(**stored["track_graph"])

        with pytest.warns(DeprecationWarning, match="place_bin_centers_"):
            environment = Environment(**stored)

        assert not environment._is_fitted
        assert environment.get_params().keys() == config
        assert environment.environment_name == stored["environment_name"]


@pytest.mark.parametrize(("version", "artifact"), MODELS)
def test_saved_model_can_be_displayed_and_its_parts_rebuilt(version, artifact):
    """A loaded detector prints (as in a notebook), reports its parameters,
    copies, and its transitions rebuild from their stored attributes."""
    model = _load_model(version, artifact)

    repr(model)
    params = model.get_params(deep=False)
    copy.deepcopy(model)
    for transition in [t for row in model.continuous_transition_types for t in row]:
        if dataclasses.is_dataclass(transition):
            assert type(transition)(**vars(transition)) == transition
    if version.startswith("v"):  # releases, saved before these parameters
        assert params["discrete_transition_prior_weight"] == 0.0
        assert params["frozen_discrete_transition_rows"] is None


def test_random_walk_row_with_the_removed_diffusion_option_is_refused():
    """Rows up to 0.5.12 store ``use_diffusion``; only its default rebuilds,
    because the diffusion random walk it selected no longer exists."""
    assert cst.RandomWalk(use_diffusion=False) == cst.RandomWalk()
    with pytest.raises(ValidationError, match="use_diffusion"):
        cst.RandomWalk(use_diffusion=True)


@pytest.mark.parametrize(("version", "filename"), RESULTS)
def test_saved_results_support_spyglass_result_operations(version, filename):
    """``fetch_results`` and the operations Spyglass applies to it."""
    path = COMPAT_DATA / version / filename
    netcdf3 = filename.endswith("_netcdf3.nc")
    assert path.read_bytes().startswith(b"CDF" if netcdf3 else b"\x89HDF")
    results = SortedSpikesDetector.load_results(path)

    in_range = results.sel(time=slice(-np.inf, np.inf))
    assert in_range.sizes["time"] == results.sizes["time"]
    assert results.coords["state_bins"].size == results.sizes["state_bins"]
    states = np.atleast_1d(results.coords["states"].values)
    assert list(states) == [
        "Local",
        "No-Spike",
        "Non-Local Continuous",
        "Non-Local Fragmented",
    ]
    position_dims = ["x_position", "y_position"] if "2d" in filename else ["position"]
    posterior = results.acausal_posterior.unstack("state_bins").sum("state")
    assert set(posterior.dims) == {"time", *position_dims}
    view = (
        results.acausal_posterior.unstack("state_bins")
        .drop_sel(state=["Local", "No-Spike"], errors="ignore")
        .sum("state")
    )
    assert set(view.dims) == {"time", *position_dims}
    if "interval_labels" in results.coords:
        decoded = results.where(results.interval_labels >= 0, drop=True)
        assert 0 < decoded.sizes["time"] < results.sizes["time"]
    # Each time bin's posterior is a distribution over the state bins
    # (positions outside the track are NaN).
    total = results.acausal_posterior.sum("state_bins", skipna=True)
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


@pytest.mark.parametrize(("version", "artifact"), MODELS)
def test_maximum_a_posteriori_position_of_saved_results(version, artifact):
    """The nld part of Spyglass ``get_ahead_behind_distance`` that needs no
    bin bounds: the MAP position from the loaded model and results."""
    model = _load_model(version, artifact)
    posterior = (
        _load_results(version, artifact)
        .acausal_posterior.unstack("state_bins")
        .sum("state")
    )
    track_graph = model.environments[0].track_graph

    if track_graph is None:
        map_position = analysis.maximum_a_posteriori_estimate(posterior)
    else:
        from non_local_detector.analysis.distance1D import (
            _get_MAP_estimate_2d_position_edges,
        )

        map_position, _ = _get_MAP_estimate_2d_position_edges(
            posterior, track_graph, model
        )

    assert map_position.shape == (posterior.sizes["time"], 2)


@pytest.mark.parametrize(("version", "artifact"), MODELS)
def test_results_without_bin_bounds_cannot_be_aligned_to_tracking(
    version, artifact, inputs
):
    """Results saved before decode bins were explicit have no bin bounds, so
    ``align_tracking_to_results`` raises for them; those saved since align."""
    results = _load_results(version, artifact)
    position, _ = _recording(inputs, "clusterless" in artifact)
    time = inputs["time"]
    tracking = pd.DataFrame(
        {"x": position[:, 0], "y": position[:, 1]}, index=pd.Index(time, name="time")
    )

    def align():
        return analysis.align_tracking_to_results(
            tracking,
            results,
            position_columns=["x", "y"],
            valid_position_intervals=[[time[0], time[-1]]],
        )

    if version in LEGACY_VERSIONS:
        with pytest.raises(ValidationError, match="time_bin_start"):
            align()
    else:
        aligned, valid = align()
        assert len(aligned) == results.sizes["time"]
        assert valid.any()


@pytest.mark.parametrize(("version", "preset", "row_format"), ROWS)
def test_stored_parameter_row_rebuilds_a_detector(version, preset, row_format, inputs):
    """``DecodingParameters.fetch1`` and ``fetch_environments``: the row
    rebuilds a detector under the current validation, and its environments
    initialize from the recording into ``environments_``."""
    detector = _spyglass_restore(_row(version, preset, row_format))

    position, _ = _recording(inputs, "clusterless" in preset)
    detector.initialize_environments(position=position)

    environment = detector.environments_[0]
    assert (environment.track_graph is not None) == ("1d" in preset)
    assert environment.is_track_interior_.any()
    # The rebuilt detector's parameters stay configuration.
    assert not detector.environments[0]._is_fitted


@pytest.mark.parametrize(
    ("version", "artifact"),
    [(v, a) for v, a in MODELS if "kde_log" not in a],
)
def test_stored_row_refits_and_decodes(version, artifact, inputs):
    """``make()`` with an old row: the rebuilt detector fits and decodes."""
    row_format = "vars" if version in LEGACY_VERSIONS else "get_params"
    detector = _spyglass_restore(_row(version, artifact, row_format))
    clusterless = "clusterless" in artifact
    position, spikes = _recording(inputs, clusterless)
    time = inputs["time"]

    detector.fit(
        position_time=time,
        position=position,
        is_training=time <= 1.6,
        valid_position_intervals=[[time[0], time[-1]]],
        **spikes,
    )
    time_edges = detector.calculate_time_edges([time[0], time[-1]], trim=True)
    results = detector.predict(
        **_spikes_in(spikes, time_edges),
        time_edges=time_edges,
        position_time=time,
        position=position,
    )

    total = results.acausal_posterior.sum("state_bins", skipna=True)
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("version", "artifact"), [(v, a) for v, a in MODELS if v in LEGACY_VERSIONS]
)
def test_decoding_with_a_detector_fitted_before_hz_rates_asks_for_a_refit(
    version, artifact, inputs
):
    model = _load_model(version, artifact)
    position, spikes = _recording(inputs, "clusterless" in artifact)
    time = inputs["time"]
    time_edges = model.calculate_time_edges([time[0], time[-1]], trim=True)

    with pytest.raises(ValidationError, match="legacy or unknown time contract"):
        model.predict(
            **_spikes_in(spikes, time_edges),
            time_edges=time_edges,
            position_time=time,
            position=position,
        )


@pytest.mark.parametrize(
    ("version", "artifact"), [(v, a) for v, a in MODELS if v not in LEGACY_VERSIONS]
)
def test_detector_saved_by_recent_main_decodes_as_it_did(version, artifact, inputs):
    """A detector saved by main before encoding models recorded their
    algorithm still decodes, and as the saving version decoded it."""
    model = _load_model(version, artifact)
    position, spikes = _recording(inputs, "clusterless" in artifact)
    time = inputs["time"]
    time_edges = model.calculate_time_edges([time[0], time[-1]], trim=True)

    posterior = model.predict(
        **_spikes_in(spikes, time_edges),
        time_edges=time_edges,
        position_time=time,
        position=position,
    ).acausal_posterior.values

    reference_path = COMPAT_DATA / version / "predict_reference.pkl"
    reference = pickle.loads(reference_path.read_bytes())[artifact]
    # The golden-regression tolerances (test_golden_regression): float32
    # posteriors decoded on another platform or XLA build differ by a few ulp.
    np.testing.assert_allclose(posterior, reference, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("version", "artifact"), [(v, a) for v, a in MODELS if v not in LEGACY_VERSIONS]
)
def test_detector_saved_by_recent_main_has_the_declared_value_types(version, artifact):
    """Loading gives each encoding-model value the type its algorithm now
    declares, whatever type the saving version stored."""
    model = _load_model(version, artifact)
    for encoding_model in model.encoding_model_.values():
        record = {**_SORTED_SPIKES_ALGORITHMS, **_CLUSTERLESS_ALGORITHMS}[
            encoding_model["algorithm"]
        ]
        assert type_violations(encoding_model, record.encoding_model_type) == []


def test_row_with_the_removed_kde_log_algorithm_rebuilds_but_does_not_fit(inputs):
    """``clusterless_kde_log`` rows (main, 2025-10 to 2026-10) rebuild for
    inspection; fitting asks for ``clusterless_kde``."""
    detector = _spyglass_restore(
        _row("main-84259d49", "nonlocal_clusterless_kde_log_2d", "vars")
    )
    position, spikes = _recording(inputs, clusterless=True)
    detector.initialize_environments(position=position)

    with pytest.raises(ValidationError, match="clusterless_kde_log") as error:
        detector.fit(position_time=inputs["time"], position=position, **spikes)
    assert 'Use "clusterless_kde"' in str(error.value)
