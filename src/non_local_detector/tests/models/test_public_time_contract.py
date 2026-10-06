"""Public decoding calls expose an explicit, recoverable time contract."""

import pickle

import numpy as np
import pytest
import xarray as xr

from non_local_detector import Environment, SortedSpikesDecoder
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin


@pytest.fixture
def recording():
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121])]
    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        infer_track_interior=False,
    ).fit(time, position, spikes)
    return detector, time, position, spikes


@pytest.mark.unit
def test_grid_constructor_is_available_from_package():
    from non_local_detector import calculate_time_edges

    np.testing.assert_allclose(
        calculate_time_edges([0, 0.01], sampling_frequency=500),
        np.arange(6) * 0.002,
    )


@pytest.mark.unit
def test_positional_count_edges_cannot_silently_change_meaning():
    with pytest.raises(TypeError, match="time_edges"):
        get_spikecount_per_time_bin(np.array([0.001]), np.arange(4) * 0.002)


@pytest.mark.unit
def test_old_keyword_explains_centered_grid_migration(recording):
    detector, time, _, spikes = recording
    with pytest.raises(TypeError, match="time_edges_from_centers"):
        detector.predict(spikes, time=time)


@pytest.mark.integration
def test_results_preserve_edges_and_missing_mask_through_pickle_and_concat(recording):
    detector, _, _, spikes = recording
    pieces = []
    for start in (0.01, 0.11):
        edges = start + np.arange(6) * 0.002
        missing = np.array([False, False, True, False, False])
        result = detector.predict(spikes, time_edges=edges, is_missing=missing)
        np.testing.assert_array_equal(result.time_bin_start, edges[:-1])
        np.testing.assert_array_equal(result.time_bin_end, edges[1:])
        np.testing.assert_array_equal(result.is_missing, missing)
        assert (
            result.attrs["time_bin_convention"] == "left_closed_right_open_final_closed"
        )
        assert result.attrs["non_local_detector_version"]
        restored = pickle.loads(pickle.dumps(result))
        xr.testing.assert_identical(restored, result)
        pieces.append(result)
    joined = xr.concat(pieces, dim="time")
    np.testing.assert_array_equal(
        joined.time_bin_start, np.r_[pieces[0].time_bin_start, pieces[1].time_bin_start]
    )
    np.testing.assert_array_equal(
        joined.time_bin_end_inclusive, np.tile([False, False, False, False, True], 2)
    )


@pytest.mark.integration
def test_saved_results_recover_time_metadata_and_spatial_index(recording, tmp_path):
    detector, _, _, spikes = recording
    result = detector.predict(
        spikes,
        time_edges=np.arange(6) * 0.002,
        is_missing=[False, False, True, False, False],
    )
    path = tmp_path / "results.nc"
    detector.save_results(result, path)
    with detector.load_results(path) as restored:
        xr.testing.assert_equal(restored.load(), result)
        assert (
            restored.attrs["time_bin_convention"] == result.attrs["time_bin_convention"]
        )
        assert (
            restored.attrs["non_local_detector_version"]
            == result.attrs["non_local_detector_version"]
        )


@pytest.mark.integration
def test_viterbi_accepts_same_position_free_call_as_predict(recording):
    detector, _, _, spikes = recording
    edges = np.arange(21) * 0.002
    result = detector.predict(spikes, time_edges=edges)
    sequence = detector.most_likely_sequence(spikes, time_edges=edges)
    np.testing.assert_array_equal(sequence.index, result.time)


@pytest.mark.integration
def test_clusterless_diffusion_masks_missing_tracking_for_local_prediction():
    from non_local_detector import NonLocalClusterlessDetector

    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121])]
    features = [np.zeros((3, 1))]
    detector = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        infer_track_interior=False,
        clusterless_algorithm="clusterless_diffusion",
    ).fit(time, position, spikes, features)
    position[40:45] = np.nan
    result = detector.predict(
        spikes,
        features,
        time_edges=np.arange(101) * 0.002,
        position_time=time,
        position=position,
        return_outputs="log_likelihood",
    )
    assert result.is_missing.any()
    missing_ll = (
        result.log_likelihood.sel(time=result.time[result.is_missing])
        .dropna("state_bins")
        .to_numpy()
    )
    np.testing.assert_array_equal(missing_ll, np.zeros_like(missing_ll))


@pytest.mark.integration
def test_single_bin_keeps_a_time_dimension(recording):
    detector, _, _, spikes = recording
    result = detector.predict(spikes, time_edges=np.array([0.0, 0.002]))
    assert result.sizes["time"] == 1
    np.testing.assert_array_equal(result.time_bin_end_inclusive, [True])


@pytest.mark.unit
def test_public_counting_validates_array_like_edges():
    from non_local_detector.exceptions import DataError
    from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

    np.testing.assert_array_equal(
        get_spikecount_per_time_bin(np.array([0.5, 1.0]), time_edges=[0.0, 1.0]), [2]
    )
    for edges in ([0.0, 0.0, 1.0], [0.0, 2.0, 1.0]):
        with pytest.raises(DataError):
            get_spikecount_per_time_bin(np.array([0.5]), time_edges=edges)


@pytest.mark.integration
@pytest.mark.parametrize(
    "method", ["predict", "most_likely_sequence", "compute_log_likelihood", "cached"]
)
def test_unknown_transition_clock_requires_refitting(recording, method):
    from non_local_detector.exceptions import ValidationError

    detector, time, position, spikes = recording
    edges = np.arange(21) * 0.002
    cached = np.asarray(
        detector.predict(
            spikes, time_edges=edges, return_outputs="log_likelihood"
        ).log_likelihood
    )
    del detector.transition_time_bin_width_
    legacy = pickle.loads(pickle.dumps(detector))
    with pytest.raises(ValidationError, match="refit"):
        if method == "cached":
            legacy._predict(edges, log_likelihoods=cached)
        else:
            getattr(legacy, method)(
                spike_times=spikes,
                time_edges=edges,
                position_time=time,
                position=position,
            )


@pytest.mark.unit
@pytest.mark.parametrize(
    "method", ["predict", "most_likely_sequence", "compute_log_likelihood"]
)
def test_unfitted_decoder_explains_fit_requirement(method):
    from non_local_detector.exceptions import ValidationError

    detector = SortedSpikesDecoder()
    with pytest.raises(ValidationError, match="call fit"):
        getattr(detector, method)(
            spike_times=[np.array([])],
            time_edges=np.array([0.0, 0.002]),
            position_time=None,
            position=None,
        )


@pytest.mark.integration
@pytest.mark.parametrize("n_bins", [1, 5])
def test_one_spatial_bin_keeps_its_index_through_save_load(tmp_path, n_bins):
    from non_local_detector import Uniform

    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=10, position_range=((0, 1),)),
        continuous_transition_types=[[Uniform()]],
        infer_track_interior=False,
    ).fit(np.arange(10) * 0.002, np.full((10, 1), 0.5), [np.array([0.005])])
    result = detector.predict(
        [np.array([0.005])], time_edges=np.arange(n_bins + 1) * 0.002
    )
    assert result.sizes["time"] == n_bins
    assert result.sizes["state_bins"] == 1
    path = tmp_path / "one_spatial_bin.nc"
    detector.save_results(result, path)
    with detector.load_results(path) as restored:
        xr.testing.assert_equal(restored.load(), result)


@pytest.mark.integration
@pytest.mark.parametrize("n_transition_rows", [4, 6])
def test_viterbi_rejects_unaligned_covariate_transitions_before_likelihood(
    recording, monkeypatch, n_transition_rows
):
    from non_local_detector.exceptions import ValidationError

    detector, _, _, spikes = recording
    detector.discrete_state_transitions_ = np.broadcast_to(
        detector.discrete_state_transitions_, (n_transition_rows, 1, 1)
    )

    def unexpected_likelihood(*args, **kwargs):
        pytest.fail("Covariate alignment must fail before likelihood evaluation")

    monkeypatch.setattr(detector, "compute_log_likelihood", unexpected_likelihood)
    with pytest.raises(ValidationError, match="covariate"):
        detector.most_likely_sequence(spikes, time_edges=np.arange(6) * 0.002)


@pytest.mark.integration
def test_viterbi_accepts_aligned_fitted_covariate_transitions(recording):
    detector, _, _, spikes = recording
    detector.discrete_state_transitions_ = np.broadcast_to(
        detector.discrete_state_transitions_, (5, 1, 1)
    )
    edges = np.arange(6) * 0.002
    sequence = detector.most_likely_sequence(spikes, time_edges=edges)
    np.testing.assert_array_equal(sequence.index, (edges[:-1] + edges[1:]) / 2)
