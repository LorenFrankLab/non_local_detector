"""Every registered backend's ``row_slice`` must equal the full-time result sliced.

``predict(n_chunks > 1)`` asks each backend for a contiguous range of rows of
the full-time likelihood. The row-range result therefore has to satisfy two
properties for chunked prediction to be exact:

1. ``predict(time, ..., row_slice=slice(a, b)) == predict(time, ...)[a:b]``
2. concatenating a partition of the rows reproduces the full-time result --
   in particular no spike between two adjacent chunks is lost or double counted.

The decoding spikes below sit at every bin midpoint, so every row partition has
a spike in each of its boundary gaps. One unit has no encoding spikes but does
have decoding spikes, which exercises the zero-rate fast paths.
"""

import jax
import numpy as np
import pytest
from patsy import build_design_matrices

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    EPS,
    get_spikecount_per_time_bin,
    resolve_row_slice,
)
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood
from non_local_detector.likelihoods.sorted_spikes_glm import make_spline_predict_matrix
from non_local_detector.tests.likelihoods.conftest import (
    ALGORITHMS,
    EXACT,
    FLOAT32_ROUNDING,
    fit_registered_backends,
    parity_kwargs,
)

N_TIME = 41
ROW_SLICE = slice(13, 29)
N_PARTITIONS = 5


@pytest.fixture(scope="module")
def decode_data():
    """Simulated encoding/decoding data with boundary spikes and a silent unit."""
    rng = np.random.default_rng(1)
    position_time = np.linspace(0.0, 4.0, 400)
    position = (50.0 + 40.0 * np.sin(2 * np.pi * position_time / 4.0))[:, None]
    environment = Environment(
        environment_name="line",
        place_bin_size=10.0,
        position_range=((0.0, 100.0),),
    ).fit_place_grid(position=position, infer_track_interior=False)

    time = np.linspace(0.0, 4.0, N_TIME)
    bin_midpoints = 0.5 * (time[:-1] + time[1:])

    # Unit 1 has encoding spikes; unit 2 has none (zero-rate fast paths).
    encoding_spike_times = [np.sort(rng.uniform(0.0, 4.0, 40)), np.array([])]
    encoding_features = [
        rng.standard_normal((40, 2)) * 5.0 + 20.0,
        np.zeros((0, 2)),
    ]
    decoding_spike_times = [
        np.sort(np.concatenate([bin_midpoints, rng.uniform(0.0, 4.0, 5)])),
        bin_midpoints.copy(),
    ]
    decoding_features = [
        rng.standard_normal((len(decoding_spike_times[0]), 2)) * 5.0 + 20.0,
        rng.standard_normal((len(decoding_spike_times[1]), 2)) * 5.0 + 20.0,
    ]

    return {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "time": time,
        "encoding_spike_times": encoding_spike_times,
        "encoding_features": encoding_features,
        "decoding_spike_times": decoding_spike_times,
        "decoding_features": decoding_features,
    }


def assert_zero_rate_sentinel(algorithm: str, encoding_model: dict) -> None:
    """The silent unit must be fitted as zero-rate, so the fast paths fire.

    Without this the fixture could drift (e.g. a unit that quietly gains
    encoding spikes) and stop exercising the zero-rate branches while the parity
    assertions still pass.
    """
    if algorithm == "sorted_spikes_glm":
        # The GLM encoding dict carries no mean_rates; a silent neuron's fitted
        # rate map sits at the EPS floor (~1e-15) across every bin instead.
        assert float(np.max(np.asarray(encoding_model["place_fields"])[1])) < 1e-12
        return

    assert float(np.asarray(encoding_model["mean_rates"])[1]) == 0.0
    if algorithm == "clusterless_gmm":
        assert encoding_model["joint_models"][1] is None
    if algorithm == "clusterless_diffusion":
        assert float(np.asarray(encoding_model["weight_total"])[1]) == 0.0


@pytest.fixture(scope="module")
def fitted_backends(decode_data):
    """Fit every registered backend once; return predict callables and arguments."""
    fitted = {}
    for name, predict_func, encoding_model, is_clusterless in fit_registered_backends(
        decode_data
    ):
        assert_zero_rate_sentinel(name, encoding_model)
        args = [
            decode_data["position_time"],
            decode_data["position"],
            decode_data["decoding_spike_times"],
        ]
        if is_clusterless:
            args.append(decode_data["decoding_features"])
        fitted[name] = (predict_func, encoding_model, tuple(args))
    return fitted


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_row_slice_equals_full_time_slice(
    fitted_backends, decode_data, algorithm, is_local
):
    """A requested row range must match the full-time likelihood's rows."""
    predict_func, encoding_model, args = fitted_backends[algorithm]
    time = decode_data["time"]

    full = np.asarray(predict_func(time, *args, **encoding_model, is_local=is_local))
    rows = np.asarray(
        predict_func(
            time, *args, **encoding_model, is_local=is_local, row_slice=ROW_SLICE
        )
    )

    assert rows.shape == full[ROW_SLICE].shape
    np.testing.assert_allclose(
        rows, full[ROW_SLICE], **parity_kwargs(algorithm, is_local=is_local)
    )


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_row_partition_tiles_full_time_result(
    fitted_backends, decode_data, algorithm, is_local
):
    """Concatenating a row partition must reproduce the full-time likelihood.

    This is the property chunked prediction relies on: spikes in the gap between
    two adjacent chunks must be counted exactly once.
    """
    predict_func, encoding_model, args = fitted_backends[algorithm]
    time = decode_data["time"]

    full = np.asarray(predict_func(time, *args, **encoding_model, is_local=is_local))
    chunks = np.array_split(np.arange(len(time)), N_PARTITIONS)
    tiled = np.concatenate(
        [
            np.asarray(
                predict_func(
                    time,
                    *args,
                    **encoding_model,
                    is_local=is_local,
                    row_slice=slice(int(chunk[0]), int(chunk[-1]) + 1),
                )
            )
            for chunk in chunks
        ]
    )

    assert tiled.shape == full.shape
    np.testing.assert_allclose(
        tiled, full, **parity_kwargs(algorithm, is_local=is_local)
    )


@pytest.mark.integration
def test_local_glm_chunks_match_float64_reference(fitted_backends, decode_data):
    """Allow rate-evaluation rounding while independently pinning spike ownership.

    Local GLM rates are recomputed for each chunk's spline matrix. XLA can
    evaluate that float32 matrix-vector product differently for different row
    counts. Check every result against float64 Poisson arithmetic, and require
    the integer counts and float32 spline inputs themselves to match exactly.
    """
    predict, model, args = fitted_backends["sorted_spikes_glm"]
    time = decode_data["time"]
    spikes = decode_data["decoding_spike_times"]
    # Independent 1-D interpolation and Patsy basis evaluation. Match the
    # predictor's working precision, then use float64 for reference arithmetic.
    working_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    position = np.interp(
        time, decode_data["position_time"], decode_data["position"][:, 0]
    ).astype(working_dtype)
    design = np.asarray(
        build_design_matrices([model["emission_design_info"]], {"x0": position})[0],
        dtype=working_dtype,
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    rates = np.maximum(np.exp(design.astype(np.float64) @ coefficients.T), EPS)

    # Derive full-timeline counts independently of the selection/count helpers.
    counts = np.zeros((len(time), len(spikes)), dtype=int)
    for neuron, unit_times in enumerate(spikes):
        in_range = unit_times[(unit_times >= time[0]) & (unit_times <= time[-1])]
        rows = np.searchsorted(time[1:-1], in_range, side="right")
        np.add.at(counts[:, neuron], rows, 1)
    assert not np.any(counts[-1])  # the recorded CI ground-process-only failure
    expected = np.sum(counts * np.log(rates) - rates, axis=1, keepdims=True)

    partitions = [
        slice(int(chunk[0]), int(chunk[-1]) + 1)
        for chunk in np.array_split(np.arange(len(time)), N_PARTITIONS)
    ]
    reference_kwargs = (
        FLOAT32_ROUNDING
        if working_dtype == np.float32
        else {"rtol": 1e-12, "atol": 1e-12}
    )
    for rows in [slice(None), slice(3, 10), *partitions]:
        chunk_design = np.asarray(
            make_spline_predict_matrix(
                model["emission_design_info"], position[rows, None]
            )
        )
        if working_dtype == np.float32:
            np.testing.assert_array_equal(chunk_design, design[rows])
        else:
            # Patsy's float64 constraint projection itself can regroup its
            # reduction by shape (measured <= 1.12e-16 absolute).
            np.testing.assert_allclose(
                chunk_design, design[rows], rtol=1e-14, atol=1e-14
            )
        for neuron, unit_times in enumerate(spikes):
            np.testing.assert_array_equal(
                get_spikecount_per_time_bin(unit_times, time, row_slice=rows),
                counts[rows, neuron],
            )
        actual = np.asarray(
            predict(time, *args, **model, is_local=True, row_slice=rows)
        )
        np.testing.assert_allclose(
            actual, expected[rows], **reference_kwargs, equal_nan=False
        )


@pytest.mark.unit
def test_no_spike_row_slice_matches_full_time(decode_data):
    """The no-spike model must also own spikes on the full timeline."""
    time = decode_data["time"]
    spike_times = decode_data["decoding_spike_times"]

    full = np.asarray(predict_no_spike_log_likelihood(time, spike_times))
    rows = np.asarray(
        predict_no_spike_log_likelihood(time, spike_times, row_slice=ROW_SLICE)
    )
    chunks = np.array_split(np.arange(len(time)), N_PARTITIONS)
    tiled = np.concatenate(
        [
            np.asarray(
                predict_no_spike_log_likelihood(
                    time,
                    spike_times,
                    row_slice=slice(int(chunk[0]), int(chunk[-1]) + 1),
                )
            )
            for chunk in chunks
        ]
    )

    np.testing.assert_allclose(rows, full[ROW_SLICE], **EXACT)
    np.testing.assert_allclose(tiled, full, **EXACT)


@pytest.mark.unit
def test_resolve_row_slice_normalizes_bounds():
    """Open and negative bounds follow ``slice.indices``; empty stays empty."""
    assert resolve_row_slice(None, 10) == (0, 10)
    assert resolve_row_slice(slice(None), 10) == (0, 10)
    assert resolve_row_slice(slice(3, None), 10) == (3, 10)
    assert resolve_row_slice(slice(None, 4), 10) == (0, 4)
    assert resolve_row_slice(slice(-3, None), 10) == (7, 10)
    assert resolve_row_slice(slice(2, -2), 10) == (2, 8)
    assert resolve_row_slice(slice(0, 99), 10) == (0, 10)
    assert resolve_row_slice(slice(1, 1), 10) == (1, 1)
    # A backwards range is an empty one, never a negative row count.
    assert resolve_row_slice(slice(7, 3), 10) == (7, 7)


@pytest.mark.unit
@pytest.mark.parametrize("step", [2, -1])
def test_resolve_row_slice_rejects_non_unit_step(step):
    """Only contiguous ranges are supported; a strided request must not pass."""
    with pytest.raises(ValidationError, match="contiguous"):
        resolve_row_slice(slice(0, 10, step), 10)
