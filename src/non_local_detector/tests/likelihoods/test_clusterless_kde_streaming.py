"""Opt-in linear KDE workspace limits preserve fitted support and row ownership."""

import copy
from contextlib import contextmanager

import jax
import numpy as np
import pytest

import non_local_detector.likelihoods.clusterless_kde as linear
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.exceptions import ValidationError
from non_local_detector.tests.likelihoods.conftest import FLOAT32_ROUNDING

pytestmark = pytest.mark.unit


def data(dtype=np.float32, weight_kind="weighted", empty_encoding=False):
    rng = np.random.default_rng(7451)
    times = np.linspace(0, 2, 19, dtype=dtype)
    position = (2 + 2 * times)[:, None]
    spikes = np.linspace(0.02, 1.98, 17, dtype=dtype)
    marks = rng.normal(size=(17, 2)).astype(dtype)
    if empty_encoding:
        spikes, marks = spikes[:0], marks[:0]
    weights = None
    if weight_kind != "uniform":
        weights = np.zeros(19, dtype=dtype)
        if weight_kind == "weighted":
            weights[2:15] = np.linspace(0.2, 2, 13, dtype=dtype)
    decode = np.array(
        [-0.1, 0, 0.125, 0.125, 0.125, 0.5, 0.74, 0.75, 1, 1, 1.1], dtype=dtype
    )
    features = rng.normal(size=(len(decode), 2)).astype(dtype)
    order = np.array([8, 3, 0, 6, 10, 1, 7, 2, 9, 4, 5])
    return {
        "position_time": times,
        "position": position,
        "training_spikes": [spikes],
        "training_marks": [marks],
        "weights": weights,
        "decode_spikes": [decode[order]],
        "decode_marks": [features[order]],
        "time_edges": np.array([0, 0.125, 0.25, 0.5, 0.75, 1], dtype=dtype),
    }


def fit(environment, values, **policy):
    return linear.fit_clusterless_kde_encoding_model(
        position_time=values["position_time"],
        position=values["position"],
        spike_times=values["training_spikes"],
        spike_waveform_features=values["training_marks"],
        weights=values["weights"],
        environment=environment,
        position_std=0.8,
        waveform_std=0.7,
        block_size=4,
        disable_progress_bar=True,
        encoding_time_range=[0, 2],
        **policy,
    )


def predict(encoding, values, is_local, row_slice=None, **policy):
    return linear.predict_clusterless_kde_log_likelihood(
        position_time=values["position_time"],
        position=values["position"],
        spike_times=values["decode_spikes"],
        spike_waveform_features=values["decode_marks"],
        time_edges=values["time_edges"],
        is_local=is_local,
        row_slice=row_slice,
        **dict(encoding, **policy),
    )


def tolerance(x64):
    return {"rtol": 1e-10, "atol": 1e-10} if x64 else FLOAT32_ROUNDING


@contextmanager
def precision_mode(x64):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("name", ["encoding_block_size", "position_block_size"])
@pytest.mark.parametrize("bad", [0, -1, True, np.bool_(True), 1.5, "4"])
def test_explicit_workspace_limits_reject_invalid_fit_values(
    simple_1d_environment, name, bad
):
    with pytest.raises(ValidationError, match=name):
        fit(simple_1d_environment, data(), **{name: bad})


@pytest.mark.parametrize("name", ["encoding_block_size", "position_block_size"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
@pytest.mark.parametrize("is_local", [False, True])
def test_explicit_workspace_limits_validate_even_empty_prediction(
    simple_1d_environment, name, bad, is_local
):
    values = data()
    encoding = fit(simple_1d_environment, values)
    with pytest.raises(ValidationError, match=name):
        predict(encoding, values, is_local, slice(0, 0), **{name: bad})


@pytest.mark.parametrize(
    "policy",
    [
        {"encoding_block_size": np.int64(4)},
        {"position_block_size": np.int32(5)},
        {"encoding_block_size": 4, "position_block_size": 5},
    ],
)
def test_fitted_workspace_policy_is_carried_to_prediction(
    simple_1d_environment, policy
):
    values = data()
    reference = fit(simple_1d_environment, values)
    encoding = fit(simple_1d_environment, values, **policy)
    for name in ["encoding_block_size", "position_block_size"]:
        assert encoding[name] == policy.get(name)
        if encoding[name] is not None:
            assert type(encoding[name]) is int
    for is_local in [False, True]:
        np.testing.assert_allclose(
            predict(encoding, values, is_local),
            predict(reference, values, is_local),
            **FLOAT32_ROUNDING,
        )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("weight_kind", ["uniform", "weighted", "zero"])
@pytest.mark.parametrize("empty_encoding", [False, True])
def test_streamed_fit_and_local_nonlocal_rows_preserve_original_support(
    simple_1d_environment, x64, weight_kind, empty_encoding
):
    with precision_mode(x64):
        values = data(np.float64 if x64 else np.float32, weight_kind, empty_encoding)
        reference = fit(simple_1d_environment, values)
        encoding = fit(
            simple_1d_environment,
            values,
            encoding_block_size=4,
            position_block_size=5,
        )
        for key in ["occupancy", "summed_ground_process_intensity"]:
            assert encoding[key].dtype == reference[key].dtype
            np.testing.assert_allclose(encoding[key], reference[key], **tolerance(x64))
        assert (
            encoding["encoding_exposure_seconds"]
            == reference["encoding_exposure_seconds"]
        )
        np.testing.assert_array_equal(encoding["mean_rates"], reference["mean_rates"])
        for key in [
            "encoding_weights",
            "encoding_positions",
            "encoding_spike_waveform_features",
        ]:
            for original, tiled in zip(reference[key], encoding[key], strict=True):
                np.testing.assert_array_equal(original, tiled)
        for original, tiled in zip(
            [reference["occupancy_model"], *reference["gpi_models"]],
            [encoding["occupancy_model"], *encoding["gpi_models"]],
            strict=True,
        ):
            assert type(tiled) is type(original)
            np.testing.assert_array_equal(tiled.samples_, original.samples_)
            np.testing.assert_array_equal(tiled.weights_, original.weights_)
        for is_local in [False, True]:
            expected = predict(reference, values, is_local)
            actual = predict(encoding, values, is_local)
            assert actual.dtype == expected.dtype
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, **tolerance(x64))
            for rows in [slice(0, 2), slice(2, 4), slice(4, 5), slice(2, 2)]:
                part = predict(encoding, values, is_local, rows)
                np.testing.assert_allclose(part, expected[rows], **tolerance(x64))


def test_opt_in_fit_and_prediction_never_build_legacy_full_kernel(
    simple_1d_environment, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("opt-in streaming called an unbounded legacy KDE workspace")

    monkeypatch.setattr(linear.KDEModel, "predict", forbidden)
    monkeypatch.setattr(linear, "kde_distance", forbidden)
    monkeypatch.setattr(linear, "block_kde", forbidden)
    monkeypatch.setattr(linear, "block_estimate_log_joint_mark_intensity", forbidden)
    values = data()
    encoding = fit(
        simple_1d_environment, values, encoding_block_size=4, position_block_size=5
    )
    for is_local in [False, True]:
        assert np.isfinite(predict(encoding, values, is_local, slice(1, 5))).all()


def test_default_and_explicit_none_keep_legacy_path_bitwise(
    simple_1d_environment, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("default KDE unexpectedly entered opt-in streamed primitives")

    monkeypatch.setattr(linear, "_sample_tiled_density", forbidden, raising=False)
    monkeypatch.setattr(
        linear, "_streamed_joint_mark_row_sums", forbidden, raising=False
    )
    values = data()
    reference = fit(simple_1d_environment, values)
    explicit = fit(
        simple_1d_environment,
        values,
        encoding_block_size=None,
        position_block_size=None,
    )
    assert reference.keys() == explicit.keys()
    assert "encoding_block_size" not in reference
    assert "position_block_size" not in reference
    for key in ["occupancy", "summed_ground_process_intensity"]:
        np.testing.assert_array_equal(explicit[key], reference[key])
    for is_local in [False, True]:
        np.testing.assert_array_equal(
            predict(
                explicit,
                values,
                is_local,
                encoding_block_size=None,
                position_block_size=None,
            ),
            predict(reference, values, is_local),
        )


def test_native_detector_preserves_workspace_policy_and_posterior(
    simple_1d_environment, tmp_path
):
    values = data(weight_kind="uniform")
    values["time_edges"] = np.linspace(0, 1, 9, dtype=np.float32)
    outputs = []
    for policy in [{}, {"encoding_block_size": 4, "position_block_size": 5}]:
        environment = copy.deepcopy(simple_1d_environment)
        environment.environment_name = ""
        model = NonLocalClusterlessDetector(
            environments=environment,
            infer_track_interior=False,
            clusterless_algorithm_params={
                "position_std": 0.8,
                "waveform_std": 0.7,
                "block_size": 4,
                "disable_progress_bar": True,
                **policy,
            },
        ).fit(
            position_time=values["position_time"],
            position=values["position"],
            spike_times=values["training_spikes"],
            spike_waveform_features=values["training_marks"],
            encoding_time_range=[0, 2],
            transition_representation="structured",
        )
        for encoding in model.encoding_model_.values():
            for name, value in policy.items():
                assert encoding[name] == value
        outputs.append(
            model.predict(
                position_time=values["position_time"],
                position=values["position"],
                spike_times=values["decode_spikes"],
                spike_waveform_features=values["decode_marks"],
                time_edges=values["time_edges"],
                inference_mode="checkpointed",
                chunk_size=2,
                output_mode="compact",
                checkpoint_dir=tmp_path / str(bool(policy)),
            )
        )
    np.testing.assert_allclose(
        outputs[1].acausal_state_probabilities,
        outputs[0].acausal_state_probabilities,
        **FLOAT32_ROUNDING,
    )
    np.testing.assert_allclose(
        outputs[1].attrs["marginal_log_likelihoods"],
        outputs[0].attrs["marginal_log_likelihoods"],
        **FLOAT32_ROUNDING,
    )
