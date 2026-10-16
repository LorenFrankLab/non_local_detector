"""No-Spike scores waveform marks against a training-fitted reference density.

Active clusterless states score each spike with a marked intensity, a density
over waveform features. Without a mark density of its own, No-Spike's odds
against those states change with the waveform units.
"""

import pickle

import numpy as np
import pytest
from scipy.stats import norm

from non_local_detector import (
    ClusterlessDecoder,
    Discrete,
    Environment,
    NonLocalClusterlessDetector,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import RATE_EPS_HZ
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
    fit_mark_reference,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.time_edges import time_edges_from_centers

N_TRAIN = 8_000
N_TEST = 400


@pytest.fixture(scope="module")
def run_data():
    sim = make_simulated_run_data(n_runs=1, seed=0)
    time, position = sim.position_time, sim.position
    cut, stop = time[N_TRAIN], time[N_TRAIN + N_TEST]
    train = [times < cut for times in sim.spike_times]
    test = [(times >= cut) & (times < stop) for times in sim.spike_times]
    return {
        "train_time": time[:N_TRAIN],
        "train_position": position[:N_TRAIN],
        "test_time": time[N_TRAIN : N_TRAIN + N_TEST],
        "test_position": position[N_TRAIN : N_TRAIN + N_TEST],
        "train_spikes": [t[m] for t, m in zip(sim.spike_times, train, strict=True)],
        "test_spikes": [t[m] for t, m in zip(sim.spike_times, test, strict=True)],
        "train_marks": [
            f[m] for f, m in zip(sim.spike_waveform_features, train, strict=True)
        ],
        "test_marks": [
            f[m] for f, m in zip(sim.spike_waveform_features, test, strict=True)
        ],
    }


def _state_columns(detector, name):
    """Interior likelihood columns belonging to the named state."""
    interior = detector.state_ind_[detector.is_track_interior_state_bins_]
    return np.flatnonzero(interior == detector.state_names.index(name))


def _log_likelihood(run_data, algorithm, scale, offset):
    """Fit and score with marks and mark bandwidths in transformed units."""
    scale = np.asarray(scale, dtype=float)
    transform = [m * scale + offset for m in run_data["train_marks"]]
    detector = NonLocalClusterlessDetector(
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0 * scale,
            "disable_progress_bar": True,
        },
    ).fit(
        run_data["train_time"],
        run_data["train_position"],
        run_data["train_spikes"],
        transform,
    )
    log_likelihood = detector.compute_log_likelihood(
        run_data["test_time"],
        run_data["test_position"],
        run_data["test_spikes"],
        [m * scale + offset for m in run_data["test_marks"]],
        time_edges=time_edges_from_centers(run_data["test_time"]),
    )
    return detector, np.asarray(log_likelihood)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["clusterless_kde", "clusterless_diffusion"])
def test_state_odds_do_not_depend_on_waveform_units(run_data, algorithm):
    """Every state and position shifts by the same Jacobian per spike."""
    detector, original = _log_likelihood(run_data, algorithm, 1.0, 0.0)
    _, rescaled = _log_likelihood(
        run_data, algorithm, [10.0, 0.5, 3.0, 1.0], [5.0, -2.0, 0.0, 100.0]
    )
    no_spike = _state_columns(detector, "No-Spike")
    assert no_spike.size == 1
    n_spikes = sum(len(t) for t in run_data["test_spikes"])
    assert n_spikes > 20  # the comparison must involve observed marks

    original_odds = original - original[:, no_spike]
    rescaled_odds = rescaled - rescaled[:, no_spike]
    if algorithm == "clusterless_diffusion":
        # The heat kernel resolves its field to about 1e-6 of each spike's peak;
        # entries near that floor are reproduced only approximately by a refit.
        error = np.abs(rescaled_odds - original_odds)
        assert np.mean(error <= 1e-3 + 1e-5 * np.abs(original_odds)) >= 0.99
        assert np.max(error) < 0.1
    else:
        np.testing.assert_allclose(rescaled_odds, original_odds, atol=1e-3, rtol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["clusterless_kde", "clusterless_gmm"])
def test_outlier_spike_favors_no_spike_by_rate_ratio(run_data, algorithm):
    """A spike no state can explain is scored by the shared background, so it
    favors No-Spike over every active state by log(no_spike_rate / RATE_EPS_HZ)."""
    params = (
        {"gmm_components_joint": 8, "gmm_components_occupancy": 4}
        if algorithm == "clusterless_gmm"
        else {"position_std": 6.0, "waveform_std": 24.0}
    )
    detector = NonLocalClusterlessDetector(
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params={**params, "disable_progress_bar": True},
    ).fit(
        run_data["train_time"],
        run_data["train_position"],
        run_data["train_spikes"],
        run_data["train_marks"],
    )
    electrode = next(
        e for e, ref in enumerate(detector.mark_reference_) if ref is not None
    )
    # Broaden the shared reference so the background dominates 1000 µV away.
    broad = MarkReference(np.zeros(4), np.full(4, 3e4))
    detector.mark_reference_[electrode] = broad
    for model in detector.encoding_model_.values():
        model["mark_reference"] = detector.mark_reference_
    spike_time = run_data["test_time"][200]
    spikes = [np.array([])] * len(run_data["test_spikes"])
    marks = [np.zeros((0, 4))] * len(run_data["test_marks"])
    spikes[electrode] = np.array([spike_time])
    marks[electrode] = (run_data["train_marks"][electrode].mean(axis=0) + 1e3)[None]
    edges = time_edges_from_centers(run_data["test_time"])
    with_spike = np.asarray(
        detector.compute_log_likelihood(
            run_data["test_time"],
            run_data["test_position"],
            spikes,
            marks,
            time_edges=edges,
        )
    )
    row = np.searchsorted(edges, spike_time, "right") - 1
    no_spike = _state_columns(detector, "No-Spike")[0]
    silent = np.asarray(
        detector.compute_log_likelihood(
            run_data["test_time"],
            run_data["test_position"],
            [np.array([])] * len(spikes),
            [np.zeros((0, 4))] * len(marks),
            time_edges=edges,
        )
    )
    odds_change = (with_spike[row] - silent[row]) - (
        with_spike[row, no_spike] - silent[row, no_spike]
    )
    active = np.delete(odds_change, no_spike)
    np.testing.assert_allclose(
        active, np.log(RATE_EPS_HZ / detector.no_spike_rate), rtol=1e-4
    )


@pytest.mark.integration
def test_no_spike_adds_training_reference_log_density_per_spike():
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    train_times = [np.array([0.021, 0.067, 0.121, 0.167])]
    train_marks = [np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 30.0], [600.0, 50.0]])]
    # The last training spike falls outside is_training and must not shape the
    # reference; decoding marks never do.
    is_training = time < 0.15
    detector = ClusterlessDecoder(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        observation_models=[ObservationModel(is_no_spike=True)],
        continuous_transition_types=[[Discrete()]],
        infer_track_interior=False,
        no_spike_rate=5.0,
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, train_times, train_marks, is_training=is_training)

    decode_times = [np.array([0.0031, 0.0032, 0.021])]
    decode_marks = [np.array([[1.0, 12.0], [-3.0, 40.0], [500.0, 0.0]])]
    edges = np.arange(21) * 0.002
    result = detector.predict(
        decode_times, decode_marks, time_edges=edges, return_outputs="log_likelihood"
    )

    location = np.array([2.0, 50.0 / 3.0])
    scale = np.sqrt(np.array([8.0 / 3.0, 800.0 / 9.0]))
    widths = np.diff(edges)
    expected = -5.0 * widths
    for t, mark in zip(decode_times[0], decode_marks[0], strict=True):
        row = np.searchsorted(edges, t, side="right") - 1
        expected[row] += np.log(5.0 * widths[row])
        expected[row] += norm.logpdf(mark, loc=location, scale=scale).sum()
    np.testing.assert_allclose(
        result.log_likelihood.to_numpy()[:, 0], expected, rtol=1e-5
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("algorithm", "bandwidth"),
    [
        ("clusterless_kde", {"waveform_std": 7.0}),
        ("clusterless_gmm", {"gmm_reg_covar": 49.0}),
    ],
)
def test_constant_training_coordinate_uses_mark_bandwidth_as_reference_scale(
    algorithm, bandwidth
):
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    detector = ClusterlessDecoder(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        observation_models=[ObservationModel(is_no_spike=True)],
        continuous_transition_types=[[Discrete()]],
        infer_track_interior=False,
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params={**bandwidth, "disable_progress_bar": True},
    ).fit(time, position, [np.array([0.021, 0.067])], [np.array([[3.0], [3.0]])])
    (reference,) = detector.mark_reference_
    np.testing.assert_allclose(reference.location, [3.0])
    np.testing.assert_allclose(reference.scale, [7.0])


@pytest.mark.integration
def test_electrode_without_training_marks_adds_no_reference_term():
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    detector = ClusterlessDecoder(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        observation_models=[ObservationModel(is_no_spike=True)],
        continuous_transition_types=[[Discrete()]],
        infer_track_interior=False,
        no_spike_rate=5.0,
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        time,
        position,
        [np.array([0.021, 0.067]), np.array([])],
        [np.array([[0.0], [2.0]]), np.zeros((0, 1))],
    )
    assert detector.mark_reference_[1] is None
    edges = np.arange(21) * 0.002
    result = detector.predict(
        [np.array([]), np.array([0.021])],
        [np.zeros((0, 1)), np.array([[1e6]])],
        time_edges=edges,
        return_outputs="log_likelihood",
    )
    # Each of the two electrodes fires at no_spike_rate.
    expected = -2 * 5.0 * np.diff(edges)
    expected[10] += np.log(5.0 * np.diff(edges)[10])
    np.testing.assert_allclose(
        result.log_likelihood.to_numpy()[:, 0], expected, rtol=1e-5
    )


@pytest.mark.integration
def test_encoding_refit_keeps_training_reference(run_data):
    detector = NonLocalClusterlessDetector(
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "disable_progress_bar": True,
        },
    )
    time = run_data["train_time"]
    detector.fit(
        time,
        run_data["train_position"],
        run_data["train_spikes"],
        run_data["train_marks"],
    )
    before = pickle.loads(pickle.dumps(detector.mark_reference_))
    detector.estimate_parameters(
        time,
        run_data["train_position"],
        run_data["train_spikes"],
        run_data["train_marks"],
        time_edges=time_edges_from_centers(time),
        estimate_encoding_model=True,
        max_iter=2,
    )
    after = detector.mark_reference_
    assert sum(reference is not None for reference in before) >= 3
    for old, new in zip(before, after, strict=True):
        if old is None:
            assert new is None
            continue
        np.testing.assert_array_equal(old.location, new.location)
        np.testing.assert_array_equal(old.scale, new.scale)


@pytest.mark.integration
def test_saved_model_without_mark_reference_must_be_refit(run_data):
    detector = NonLocalClusterlessDetector(
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        run_data["train_time"],
        run_data["train_position"],
        run_data["train_spikes"],
        run_data["train_marks"],
    )
    legacy = pickle.loads(pickle.dumps(detector))
    del legacy.mark_reference_
    with pytest.raises(ValidationError, match="mark reference"):
        legacy.predict(
            run_data["test_spikes"],
            run_data["test_marks"],
            time_edges=time_edges_from_centers(run_data["test_time"]),
            position_time=run_data["test_time"],
            position=run_data["test_position"],
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("spike_times", "marks", "message"),
    [
        ([np.array([0.021])] * 2, [np.array([[0.1]])] * 2, "electrode"),
        ([np.array([0.021])], [np.array([[0.1, 0.2]])], "waveform features"),
    ],
)
def test_decode_marks_must_match_fitted_reference(spike_times, marks, message):
    time = np.arange(100) * 0.002
    detector = ClusterlessDecoder(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        observation_models=[ObservationModel(is_no_spike=True)],
        continuous_transition_types=[[Discrete()]],
        infer_track_interior=False,
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        time,
        (5 + 4 * np.sin(time * 30))[:, None],
        [np.array([0.021, 0.067])],
        [np.array([[0.0], [2.0]])],
    )
    with pytest.raises(ValidationError, match=message):
        detector.predict(
            spike_times,
            marks,
            time_edges=np.arange(21) * 0.002,
            return_outputs="log_likelihood",
        )


@pytest.mark.unit
@pytest.mark.parametrize(("n_spikes", "value"), [(3, 3.1), (5, 0.1), (7, -87.3)])
def test_inexact_constant_feature_uses_fallback_scale(n_spikes, value):
    # The weighted mean of a constant that is not exactly representable can
    # differ from it by an ulp, leaving a roundoff variance instead of zero.
    marks = np.column_stack(
        [np.full(n_spikes, value), np.arange(n_spikes, dtype=float)]
    )
    reference = fit_mark_reference(marks, np.linspace(1.0, 2.0, n_spikes), 24.0)
    assert reference.scale[0] == 24.0
    assert reference.scale[1] > 0.5


@pytest.mark.integration
def test_mark_term_is_identical_when_rows_are_requested_in_chunks(run_data):
    detector, full = _log_likelihood(run_data, "clusterless_kde", 1.0, 0.0)
    edges = time_edges_from_centers(run_data["test_time"])
    chunks = [
        np.asarray(
            detector.compute_log_likelihood(
                run_data["test_time"],
                run_data["test_position"],
                run_data["test_spikes"],
                run_data["test_marks"],
                time_edges=edges,
                row_slice=rows,
            )
        )
        for rows in (slice(0, 137), slice(137, 399), slice(399, 400))
    ]
    no_spike = _state_columns(detector, "No-Spike")[0]
    np.testing.assert_array_equal(
        np.concatenate(chunks)[:, no_spike], full[:, no_spike]
    )


@pytest.mark.integration
def test_encoding_models_must_share_the_detector_mark_reference(run_data):
    detector = NonLocalClusterlessDetector(
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        run_data["train_time"],
        run_data["train_position"],
        run_data["train_spikes"],
        run_data["train_marks"],
    )
    restored = pickle.loads(pickle.dumps(detector))
    for model in restored.encoding_model_.values():
        assert model["mark_reference"] is restored.mark_reference_
    for model in detector.encoding_model_.values():
        model["mark_reference"] = list(detector.mark_reference_)
    with pytest.raises(ValidationError, match="share the detector's mark reference"):
        detector.predict(
            run_data["test_spikes"],
            run_data["test_marks"],
            time_edges=time_edges_from_centers(run_data["test_time"]),
            position_time=run_data["test_time"],
            position=run_data["test_position"],
        )
