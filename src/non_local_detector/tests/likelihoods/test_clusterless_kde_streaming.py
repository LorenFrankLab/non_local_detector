"""Opt-in KDE workspace limits preserve fitted support and row ownership."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.common import (
    _KDE_BLOCK_KERNEL_BYTES,
)
from non_local_detector.tests.conftest import precision_mode
from non_local_detector.tests.likelihoods.conftest import (
    FLOAT32_ROUNDING,
    tolerance,
    training_mark_references,
)

pytestmark = pytest.mark.unit

WAVEFORM_STD = 0.7
TILING_DEFAULTS = {"enc_tile_size": None, "pos_tile_size": None, "use_streaming": False}
STREAMED_TILING = {"enc_tile_size": 4, "pos_tile_size": 5, "use_streaming": True}
# Tiled log-space reductions regroup float32 sums: measured differences reach
# 2.8e-6 relative to the untiled estimate.
TILED_FLOAT32 = {**FLOAT32_ROUNDING, "rtol": 1e-5}


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
    """Encoding model plus the mark reference of its weighted training spikes."""
    reference = training_mark_references(
        values["position_time"],
        values["training_spikes"],
        values["training_marks"],
        WAVEFORM_STD,
        values["weights"],
        values["position"],
        encoding_time_range=[0, 2],
    )
    encoding = clusterless_kde.fit_clusterless_kde_encoding_model(
        position_time=values["position_time"],
        position=values["position"],
        spike_times=values["training_spikes"],
        spike_waveform_features=values["training_marks"],
        weights=values["weights"],
        environment=environment,
        position_std=0.8,
        waveform_std=WAVEFORM_STD,
        block_size=4,
        disable_progress_bar=True,
        encoding_time_range=[0, 2],
        **policy,
    )
    return {**encoding, "mark_reference": reference}


def predict(encoding, values, is_local, row_slice=None, **policy):
    return clusterless_kde.predict_clusterless_kde_log_likelihood(
        position_time=values["position_time"],
        position=values["position"],
        spike_times=values["decode_spikes"],
        spike_waveform_features=values["decode_marks"],
        time_edges=values["time_edges"],
        is_local=is_local,
        row_slice=row_slice,
        **dict(encoding, **policy),
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"enc_tile_size": 4, "use_streaming": True},
        {"pos_tile_size": 5},
        {"enc_tile_size": 4, "pos_tile_size": 5, "use_streaming": True},
    ],
)
def test_fitted_workspace_policy_is_carried_to_prediction(
    simple_1d_environment, policy
):
    values = data()
    reference = fit(simple_1d_environment, values)
    encoding = fit(simple_1d_environment, values, **policy)
    # The fitted model carries the tiling options to prediction.
    for name, default in TILING_DEFAULTS.items():
        assert encoding[name] == policy.get(name, default)
    for is_local in [False, True]:
        np.testing.assert_allclose(
            predict(encoding, values, is_local),
            predict(reference, values, is_local),
            **TILED_FLOAT32,
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
        encoding = fit(simple_1d_environment, values, **STREAMED_TILING)
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
        tiled_tolerance = tolerance(True) if x64 else TILED_FLOAT32
        for is_local in [False, True]:
            expected = predict(reference, values, is_local)
            actual = predict(encoding, values, is_local)
            assert actual.dtype == expected.dtype
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, **tiled_tolerance)
            for rows in [slice(0, 2), slice(2, 4), slice(4, 5), slice(2, 2)]:
                part = predict(encoding, values, is_local, rows)
                np.testing.assert_allclose(part, expected[rows], **tiled_tolerance)


@pytest.mark.parametrize(
    "n_spikes, n_encoding", [(40_000, 200_000), (400_000, 200_000), (40_000, 2_000_000)]
)
def test_local_terms_keep_compiled_workspace_bounded_without_tiling(
    n_spikes, n_encoding
):
    """Local prediction needs no encoding tiles to bound its kernel workspace.

    The full (spikes x encoding samples) kernel would take 30-300 GiB; the
    compiled per-electrode local step stays within a few kernel budgets for
    any spike or encoding count, even with blocks of 10,000 spikes.
    """

    def shape(*dims, dtype=jnp.float32):
        return jax.ShapeDtypeStruct(dims, dtype)

    n_rows, n_position_dims, n_features = 20_000, 1, 4
    with precision_mode(False):
        compiled = clusterless_kde._add_electrode_local_terms.lower(
            shape(n_rows),  # log_likelihood
            shape(n_rows),  # expected_counts
            shape(n_spikes, n_position_dims),  # spike_positions
            shape(n_spikes, n_features),  # decoding_features
            shape(n_spikes, dtype=jnp.int32),  # row_ids
            shape(n_encoding, n_position_dims),  # encoding_positions
            shape(n_encoding, n_features),  # encoding_features
            shape(n_encoding),  # encoding_weights
            shape(n_position_dims),  # position_std
            shape(n_features),  # waveform_stds
            shape(n_spikes),  # log_occupancy_at_spikes
            shape(n_encoding, n_position_dims),  # gpi_samples
            shape(n_encoding),  # gpi_weights
            shape(n_position_dims),  # gpi_std
            shape(n_rows, n_position_dims),  # positions
            shape(n_rows),  # log_occupancy
            1.0,  # scaled_rate
            1.0,  # mean_rate
            shape(n_spikes),  # log_background
            block_size=10_000,
            gpi_block_size=10_000,
            indices_are_sorted=True,
        ).compile()
    workspace = compiled.memory_analysis().temp_size_in_bytes
    assert workspace <= 8 * _KDE_BLOCK_KERNEL_BYTES, f"{workspace / 2**30:.2f} GiB"


def test_native_detector_preserves_workspace_policy_and_posterior(
    simple_1d_environment, tmp_path
):
    values = data(weight_kind="uniform")
    values["time_edges"] = np.linspace(0, 1, 9, dtype=np.float32)
    outputs = []
    for policy in [{}, STREAMED_TILING]:
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
            for name, default in TILING_DEFAULTS.items():
                assert encoding[name] == policy.get(name, default)
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
