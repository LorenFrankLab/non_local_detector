"""Saved-model units are checked for every encoding entry actually dispatched."""

import copy

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    ClusterlessDecoder,
    Discrete,
    Environment,
    SortedSpikesDecoder,
    Uniform,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.tests.models.test_failed_calls_preserve_model import (
    _assert_unchanged,
    _snapshot,
)


@pytest.fixture(
    scope="module",
    params=[
        "sorted-groups",
        "clusterless-groups",
        "sorted-environments",
        "clusterless-environments",
    ],
)
def multi_entry_recording(request):
    family, layout = request.param.split("-")
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    spike_args = ([np.array([0.021, 0.067, 0.121, 0.167])],)
    if family == "clusterless":
        spike_args = (*spike_args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    names = ["", ""] if layout == "groups" else ["first", "second"]
    groups = [0, 1] if layout == "groups" else [0, 0]
    environments = tuple(
        Environment(environment_name=name, place_bin_size=2, position_range=((0, 10),))
        for name in dict.fromkeys(names)
    )
    cls = (
        ContFragClusterlessClassifier
        if family == "clusterless"
        else ContFragSortedSpikesClassifier
    )
    algorithm_param = (
        "clusterless_algorithm_params"
        if family == "clusterless"
        else "sorted_spikes_algorithm_params"
    )
    detector = cls(
        environments=environments,
        observation_models=[
            ObservationModel(name, group)
            for name, group in zip(names, groups, strict=True)
        ],
        continuous_transition_types=[
            [Uniform(source, destination) for destination in names] for source in names
        ],
        infer_track_interior=False,
        **{algorithm_param: {"disable_progress_bar": True}},
    )
    parameter_keys = set(detector.get_params(deep=False))
    labels = {
        "encoding_group_labels": np.repeat(groups, 50),
        "environment_labels": np.repeat(names, 50),
    }
    detector.fit(time, position, *spike_args, **labels)
    assert set(detector.encoding_model_) == set(zip(names, groups, strict=True))
    return (
        detector,
        time,
        position,
        spike_args,
        labels,
        (names[1], groups[1]),
        parameter_keys,
    )


def _call_model(detector, entry, time, position, spike_args):
    edges = np.arange(21) * 0.002
    if entry == "likelihood":
        return detector.compute_log_likelihood(
            time, position, *spike_args, time_edges=edges
        )
    if entry == "cached":
        return detector._predict(
            edges,
            log_likelihoods=np.zeros(
                (20, detector.is_track_interior_state_bins_.sum())
            ),
        )
    method = detector.predict if entry == "predict" else detector.most_likely_sequence
    return method(*spike_args, time_edges=edges, position_time=time, position=position)


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["predict", "viterbi", "likelihood", "cached"])
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_entry",
        "missing_marker",
        "old_marker",
        "unsupported_marker",
        "unsupported_entry",
    ],
)
def test_later_active_entry_rejected_before_any_backend_or_hmm_work(
    multi_entry_recording, monkeypatch, entry, corruption
):
    from non_local_detector.likelihoods import (
        _CLUSTERLESS_ALGORITHMS,
        _SORTED_SPIKES_ALGORITHMS,
    )
    from non_local_detector.models import base

    fitted, time, position, spike_args, _, later_key, _ = multi_entry_recording
    detector = copy.deepcopy(fitted)
    if corruption == "missing_entry":
        del detector.encoding_model_[later_key]
    elif corruption == "missing_marker":
        del detector.encoding_model_[later_key]["rate_units"]
    elif corruption == "old_marker":
        detector.encoding_model_[later_key]["rate_units"] = "per_position_sample"
    elif corruption == "unsupported_marker":
        detector.encoding_model_[later_key]["rate_units"] = np.array(["Hz", "Hz"])
    else:
        detector.encoding_model_[later_key] = None
    snapshot = _snapshot(detector)

    def unexpected_work(*args, **kwargs):
        pytest.fail(
            "All required encoding entries must be checked before likelihood/HMM work"
        )

    if hasattr(detector, "clusterless_algorithm"):
        registry, algorithm = _CLUSTERLESS_ALGORITHMS, detector.clusterless_algorithm
    else:
        registry, algorithm = (
            _SORTED_SPIKES_ALGORITHMS,
            detector.sorted_spikes_algorithm,
        )
    monkeypatch.setitem(registry, algorithm, (registry[algorithm][0], unexpected_work))
    monkeypatch.setattr(base, "chunked_filter_smoother", unexpected_work)
    with pytest.raises(ValidationError, match="refit"):
        _call_model(detector, entry, time, position, spike_args)
    _assert_unchanged(detector, snapshot)


@pytest.mark.integration
def test_unused_legacy_entry_does_not_block_current_observations(multi_entry_recording):
    fitted, time, position, spike_args, _, _, _ = multi_entry_recording
    detector = copy.deepcopy(fitted)
    edges = np.arange(21) * 0.002
    expected = detector.predict(
        *spike_args, time_edges=edges, return_outputs="log_likelihood"
    )
    detector.encoding_model_[("unused", 99)] = {"rate_units": "per_position_sample"}
    actual = detector.predict(
        *spike_args, time_edges=edges, return_outputs="log_likelihood"
    )
    xr.testing.assert_identical(actual, expected)


@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("unused_model", ["legacy", "absent"])
def test_no_spike_state_uses_constructor_hz_rate_without_encoding_entry(
    family, unused_model
):
    time = np.arange(100) * 0.002
    position = (5 + 4 * np.sin(time * 30))[:, None]
    spike_args = ([np.array([0.021])],)
    if family == "clusterless":
        cls = ClusterlessDecoder
        spike_args = (*spike_args, [np.array([[0.1]])])
    else:
        cls = SortedSpikesDecoder
    detector = cls(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        observation_models=[ObservationModel(is_no_spike=True)],
        continuous_transition_types=[[Discrete()]],
        infer_track_interior=False,
        no_spike_rate=5.0,
    ).fit(time, position, *spike_args)
    if unused_model == "legacy":
        detector.encoding_model_[("", 0)]["rate_units"] = "per_position_sample"
    else:
        detector.encoding_model_.clear()
    edges = np.arange(21) * 0.002
    result = detector.predict(
        *spike_args, time_edges=edges, return_outputs="log_likelihood"
    )
    expected = -5.0 * np.diff(edges)
    expected[10] += np.log(5.0 * np.diff(edges)[10])
    np.testing.assert_allclose(
        result.log_likelihood.to_numpy()[:, 0], expected, rtol=1e-5
    )


@pytest.mark.integration
def test_fitted_units_stay_out_of_constructor_parameters(multi_entry_recording):
    fitted, _, _, _, _, _, before_keys = multi_entry_recording
    params = fitted.get_params(deep=False)
    assert set(params) == before_keys
    for field in (
        "time_contract_",
        "encoding_model_",
        "rate_units",
        "encoding_exposure_seconds",
        "transition_time_bin_width_",
    ):
        assert field not in params
    rebuilt = type(fitted)(**copy.deepcopy(params))
    assert not hasattr(rebuilt, "time_contract_")
    assert not hasattr(rebuilt, "encoding_model_")
    assert "rate_units" not in vars(rebuilt)
    assert "encoding_exposure_seconds" not in vars(rebuilt)


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["predict", "viterbi", "likelihood", "cached"])
def test_legacy_file_loads_for_inspection_without_running_constructor(
    multi_entry_recording, tmp_path, monkeypatch, entry
):
    fitted, time, position, spike_args, _, _, _ = multi_entry_recording
    legacy = copy.deepcopy(fitted)
    for name in (
        "time_contract_",
        "transition_time_bin_width_",
        "_transition_time_bin_width_tolerance",
    ):
        legacy.__dict__.pop(name, None)
    for model in legacy.encoding_model_.values():
        model.pop("rate_units", None)
        model.pop("encoding_exposure_seconds", None)
        # Uniform 500 Hz legacy fits stored events per position sample.
        for name in (
            "mean_rates",
            "place_fields",
            "no_spike_part_log_likelihood",
            "summed_ground_process_intensity",
        ):
            if name in model:
                model[name] = np.asarray(model[name]) / 500
    path = tmp_path / "legacy.pkl"
    legacy.save_model(path)

    def unexpected_constructor(*args, **kwargs):
        pytest.fail("Unpickling must preserve legacy state for inspection")

    monkeypatch.setattr(type(legacy), "__init__", unexpected_constructor)
    restored = type(legacy).load_model(path)
    assert not hasattr(restored, "time_contract_")
    for key, model in legacy.encoding_model_.items():
        np.testing.assert_array_equal(
            restored.encoding_model_[key]["mean_rates"], model["mean_rates"]
        )
    with pytest.raises(ValidationError, match="refit"):
        _call_model(restored, entry, time, position, spike_args)


@pytest.mark.integration
def test_encoding_refit_restamps_each_entry_with_weighted_seconds(
    multi_entry_recording,
):
    fitted, time, position, spike_args, labels, _, _ = multi_entry_recording
    detector = copy.deepcopy(fitted)
    del detector.time_contract_
    for model in detector.encoding_model_.values():
        del model["rate_units"]
    detector.fit_encoding_model(
        time, position, *spike_args, **labels, weights=np.full(len(time), 0.25)
    )
    assert detector.time_contract_ == fitted.time_contract_
    for key, model in detector.encoding_model_.items():
        assert model["rate_units"] == "Hz"
        assert model["encoding_exposure_seconds"] == pytest.approx(0.025)
        np.testing.assert_allclose(
            model["mean_rates"], fitted.encoding_model_[key]["mean_rates"], rtol=1e-5
        )


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["cached", "base_viterbi"])
def test_missing_encoding_container_cannot_bypass_concrete_model_guard(
    multi_entry_recording, monkeypatch, entry
):
    from non_local_detector.core import row_slice_aware
    from non_local_detector.models import base

    fitted, _, _, _, _, _, _ = multi_entry_recording
    detector = copy.deepcopy(fitted)
    del detector.encoding_model_
    edges = np.arange(21) * 0.002

    @row_slice_aware
    def unexpected_work(*args, **kwargs):
        pytest.fail(
            "A concrete spike model without encoding state must require fitting"
        )

    monkeypatch.setattr(base, "chunked_filter_smoother", unexpected_work)
    monkeypatch.setattr(detector, "compute_log_likelihood", unexpected_work)
    with pytest.raises(ValidationError, match="fit"):
        if entry == "cached":
            detector._predict(
                edges,
                log_likelihoods=np.zeros(
                    (20, detector.is_track_interior_state_bins_.sum())
                ),
            )
        else:
            base._DetectorBase.most_likely_sequence(detector, edges)


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm",
    [
        "sorted_spikes_kde",
        "sorted_spikes_glm",
        "sorted_spikes_diffusion",
        "sorted_spikes_mrf",
        "clusterless_kde",
        "clusterless_kde_log",
        "clusterless_gmm",
        "clusterless_diffusion",
    ],
)
def test_direct_predictors_validate_supplied_units_and_default_to_hz(algorithm):
    from non_local_detector.likelihoods import (
        _CLUSTERLESS_ALGORITHMS,
        _SORTED_SPIKES_ALGORITHMS,
    )
    from non_local_detector.tests.models.test_rate_units_and_persistence import _decoder

    time = np.arange(500) * 0.002
    position = np.zeros((len(time), 1))
    args = ([np.array([0.1, 0.3, 0.5, 0.7, 0.9])],)
    registry = _SORTED_SPIKES_ALGORITHMS
    if algorithm.startswith("clusterless"):
        args = (*args, [np.zeros((5, 1))])
        registry = _CLUSTERLESS_ALGORITHMS
    detector = _decoder(algorithm, calibration=True).fit(time, position, *args)
    (model,) = detector.encoding_model_.values()
    predictor = registry[algorithm][1]
    edges = 0.09 + np.arange(21) * 0.002
    expected = predictor(time, position, *args, time_edges=edges, **model)
    unmarked = {key: value for key, value in model.items() if key != "rate_units"}
    actual = predictor(time, position, *args, time_edges=edges, **unmarked)
    np.testing.assert_array_equal(actual, expected)
    for units in (None, "per_position_sample"):
        with pytest.raises(ValidationError, match="refit"):
            predictor(
                time, position, *args, time_edges=edges, **unmarked, rate_units=units
            )
