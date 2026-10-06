"""Phase 6b/6d acceptance through fitted public detector APIs."""

import copy

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    ClusterlessDecoder,
    Environment,
    SortedSpikesDecoder,
    Uniform,
)
from non_local_detector.exceptions import ValidationError

ALGORITHMS = [
    "sorted_spikes_kde",
    "sorted_spikes_glm",
    "sorted_spikes_diffusion",
    "sorted_spikes_mrf",
    "clusterless_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
]


def _decoder(algorithm, *, calibration=False):
    environment = Environment(
        place_bin_size=1 if calibration else 5,
        position_range=((-1.5, 1.5),) if calibration else ((40, 60),),
    )
    options = {
        "environments": environment,
        "continuous_transition_types": [[Uniform()]],
        "infer_track_interior": False,
    }
    params = {"disable_progress_bar": True}
    if algorithm == "sorted_spikes_glm" and calibration:
        # Natural cubic knots must fit this small spatial domain.
        params["emission_knot_spacing"] = 1.0
    if algorithm == "sorted_spikes_mrf":
        params["penalty"] = 1.0
    if algorithm == "clusterless_gmm":
        params.update(
            gmm_components_occupancy=1,
            gmm_components_gpi=1,
            gmm_components_joint=1,
        )
    if algorithm.startswith("sorted"):
        return SortedSpikesDecoder(
            sorted_spikes_algorithm=algorithm,
            sorted_spikes_algorithm_params=params,
            **options,
        )
    return ClusterlessDecoder(
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params=params,
        **options,
    )


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("position_frequency", [30, 500])
def test_known_rate_is_calibrated_through_public_fit_and_predict(
    algorithm, position_frequency
):
    """Five events over one second predict 0.01 events in an empty 2 ms bin.

    Evaluate the independently known homogeneous rate at its observed position,
    without assuming spatial fits coincide at unobserved locations. Three
    spatial bins avoid the unrelated singleton environment-index limitation;
    zero-centered data avoid ill-conditioned Gaussian mean/covariance arithmetic.
    """
    time = np.arange(position_frequency) / position_frequency
    position = np.zeros((len(time), 1))
    spikes = [np.array([0.1, 0.3, 0.5, 0.7, 0.9])]
    args = (spikes,)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.zeros((5, 1))])
    detector = _decoder(algorithm, calibration=True).fit(time, position, *args)

    for width in (0.002, 0.004):
        result = detector.predict(
            *args,
            time_edges=0.2 + np.arange(3) * width,
            return_outputs="log_likelihood",
        )
        # No event occurs here, so the Poisson/point-process log likelihood is
        # minus the expected count. This reference does not read fitted fields.
        np.testing.assert_allclose(
            result.log_likelihood.sel(position=0.0).to_numpy(),
            -5.0 * width,
            rtol=1e-5,
            atol=0,
        )
        assert np.all(np.isfinite(result.acausal_posterior))
    (encoding,) = detector.encoding_model_.values()
    assert encoding["rate_units"] == "Hz"
    assert encoding["encoding_exposure_seconds"] == pytest.approx(1.0)


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm", [name for name in ALGORITHMS if name != "sorted_spikes_glm"]
)
def test_serializable_backend_round_trip_preserves_units_and_predictions(
    algorithm, tmp_path
):
    """All seven currently serializable families keep exact decode results.

    Patsy DesignInfo serialization remains a separately tracked GLM defect.
    """
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121, 0.167])]
    args = (spikes,)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    detector = _decoder(algorithm).fit(time, position, *args)
    kwargs = {
        "time_edges": np.arange(81) * 0.002,
        "position_time": time,
        "position": position,
        "return_outputs": "log_likelihood",
    }
    expected = detector.predict(*args, **kwargs)
    path = tmp_path / "detector.pkl"
    detector.save_model(path)
    restored = detector.load_model(path)
    actual = restored.predict(*args, **kwargs)
    xr.testing.assert_identical(actual, expected)
    assert restored.time_contract_ == detector.time_contract_
    for model in restored.encoding_model_.values():
        assert model["rate_units"] == "Hz"
        assert model["encoding_exposure_seconds"] == pytest.approx(0.2)


@pytest.fixture(scope="module", params=["sorted_spikes_kde", "clusterless_kde"])
def contract_recording(request):
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121, 0.167])]
    args = (spikes,)
    if request.param.startswith("clusterless"):
        args = (*args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    return _decoder(request.param).fit(time, position, *args), time, position, args


@pytest.mark.integration
@pytest.mark.parametrize(
    "corruption", ["missing_entry_units", "unknown_entry_units", "unknown_version"]
)
@pytest.mark.parametrize("entry_point", ["predict", "viterbi", "likelihood", "cached"])
def test_unknown_contract_is_rejected_before_likelihood(
    contract_recording, corruption, entry_point, monkeypatch
):
    """An apparently current model cannot hide an incompatible encoding entry."""
    fitted, time, position, args = contract_recording
    detector = copy.deepcopy(fitted)
    (encoding,) = detector.encoding_model_.values()
    if corruption == "missing_entry_units":
        del encoding["rate_units"]
    elif corruption == "unknown_entry_units":
        encoding["rate_units"] = "per_position_sample"
    else:
        detector.time_contract_["version"] = 999
    edges = np.arange(6) * 0.002

    if entry_point != "likelihood":

        def unexpected_likelihood(*args, **kwargs):
            pytest.fail("Unknown units must fail before evaluating likelihoods")

        monkeypatch.setattr(detector, "compute_log_likelihood", unexpected_likelihood)

    with pytest.raises(ValidationError, match="refit"):
        if entry_point == "predict":
            detector.predict(*args, time_edges=edges)
        elif entry_point == "viterbi":
            detector.most_likely_sequence(*args, time_edges=edges)
        elif entry_point == "likelihood":
            detector.compute_log_likelihood(time, position, *args, time_edges=edges)
        else:
            detector._predict(
                edges,
                log_likelihoods=np.zeros(
                    (5, detector.is_track_interior_state_bins_.sum())
                ),
            )


@pytest.mark.integration
def test_estimation_refits_legacy_contract_before_decoding(contract_recording):
    """The supported recovery path fits physical units from original recording."""
    fitted, time, position, args = contract_recording
    detector = copy.deepcopy(fitted)
    del detector.time_contract_
    del detector.transition_time_bin_width_
    for encoding in detector.encoding_model_.values():
        del encoding["rate_units"]
    edges = np.arange(81) * 0.002
    result = detector.estimate_parameters(
        time,
        position,
        *args,
        time_edges=edges,
        max_iter=1,
        estimate_discrete_transition=False,
        estimate_encoding_model=False,
    )
    assert np.all(np.isfinite(result.acausal_posterior))
    assert detector.time_contract_ == fitted.time_contract_
    assert detector.transition_time_bin_width_ is None
    for encoding in detector.encoding_model_.values():
        assert encoding["rate_units"] == "Hz"
        assert encoding["encoding_exposure_seconds"] == pytest.approx(0.2)
