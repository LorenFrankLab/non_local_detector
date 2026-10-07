"""Checkpoint/replay preserves the existing single-sequence dense reference."""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from non_local_detector.checkpointed_inference import checkpointed_forward_backward
from non_local_detector.core import (
    filter,
    filter_covariate_dependent,
    smoother,
    smoother_covariate_dependent,
)
from non_local_detector.result_store import open_result_store

pytestmark = pytest.mark.unit


def problem(covariate=False, n_time=13):
    rng = np.random.default_rng(812)
    n = 4
    initial = np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32)
    transition = rng.uniform(0.05, 1, (n, n)).astype(np.float32)
    transition /= transition.sum(1, keepdims=True)
    ll = rng.normal(size=(n_time, n)).astype(np.float32)
    state_ind = np.array([0, 0, 1, 1])
    edges = np.arange(n_time + 1, dtype=float) * 0.002 + 1700000000.0
    if covariate:
        continuous = np.full((n, n), 0.5, dtype=np.float32)
        weights = rng.uniform(0.05, 1, (n_time, 2, 2)).astype(np.float32)
        weights /= weights.sum(2, keepdims=True)
        (evidence, _), (causal, predictive) = filter_covariate_dependent(
            initial, weights, continuous, state_ind, ll
        )
        acausal = smoother_covariate_dependent(weights, continuous, state_ind, causal)
        kwargs = {
            "discrete_transition_matrix": weights,
            "continuous_transition_matrix": continuous,
        }
    else:
        (evidence, _), (causal, predictive) = filter(initial, transition, ll)
        acausal = smoother(transition, causal)
        kwargs = {"transition_matrix": transition}
    return (
        edges,
        initial,
        ll,
        state_ind,
        kwargs,
        (evidence, causal, predictive, acausal),
    )


@pytest.mark.parametrize("covariate", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 4, 13, 20])
def test_dense_reference_and_global_callbacks(tmp_path, covariate, chunk_size):
    edges, initial, ll, state_ind, kwargs, reference = problem(covariate)
    calls = []

    def likelihood(time_edges, *, row_slice, is_missing):
        assert time_edges is edges
        calls.append((row_slice.start, row_slice.stop))
        assert row_slice.stop - row_slice.start <= chunk_size
        return ll[row_slice]

    result = checkpointed_forward_backward(
        edges,
        initial,
        likelihood,
        state_ind=state_ind,
        chunk_size=chunk_size,
        output_mode="spatial",
        result_path=tmp_path / "result",
        return_outputs=(
            "acausal_posterior",
            "causal_posterior",
            "predictive_posterior",
            "log_likelihood",
        ),
        **kwargs,
    )
    dataset = result.dataset
    for name, values in zip(
        ("causal_posterior", "predictive_posterior", "acausal_posterior"),
        reference[1:],
        strict=True,
    ):
        np.testing.assert_allclose(dataset[name].values, values, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(dataset.log_likelihood, ll)
    np.testing.assert_allclose(
        result.marginal_log_likelihood, reference[0], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(dataset.time_bin_start, edges[:-1])
    np.testing.assert_array_equal(dataset.time_bin_end, edges[1:])
    assert result.diagnostics["likelihood_evaluations"] == 2 * int(
        np.ceil(len(ll) / chunk_size)
    )
    assert result.diagnostics["max_chunk_rows"] <= chunk_size
    assert not list(tmp_path.glob("checkpoints-*"))
    loaded = open_result_store(tmp_path / "result")
    xr.testing.assert_equal(loaded, dataset)
    assert len(calls) == result.diagnostics["likelihood_evaluations"]


@pytest.mark.parametrize("covariate", [False, True])
def test_compact_selected_rows_keep_whole_recording_context(tmp_path, covariate):
    edges, initial, ll, state_ind, kwargs, reference = problem(covariate)
    selected = np.array([1, 2, 8, 12])
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        chunk_size=4,
        selected_rows=selected,
        return_outputs=("causal_state_probabilities", "acausal_state_probabilities"),
        **kwargs,
    )
    assert "acausal_posterior" not in result.dataset
    for name, values in [
        ("causal_state_probabilities", reference[1]),
        ("acausal_state_probabilities", reference[3]),
    ]:
        expected = np.column_stack(
            [np.asarray(values)[:, state_ind == i].sum(1) for i in range(2)]
        )
        np.testing.assert_allclose(
            result.dataset[name], expected[selected], rtol=1e-5, atol=1e-6
        )
    np.testing.assert_array_equal(result.dataset.source_row, selected)
    assert result.dataset.sizes["time"] == len(selected)


@pytest.mark.parametrize("kind", ["missing", "impossible", "nan", "support_mismatch"])
def test_original_degenerate_nan_and_missing_semantics(tmp_path, kind, caplog):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    missing = np.zeros(len(ll), bool)
    if kind == "missing":
        missing[3:5] = True
    elif kind == "impossible":
        ll[4] = -np.inf
    elif kind == "nan":
        ll[4, 1] = np.nan
    else:
        kwargs["transition_matrix"] = np.eye(4, dtype=np.float32)
        initial = np.array([1, 0, 0, 0], dtype=np.float32)
        ll[4] = [-np.inf, 0, 0, 0]
    effective = np.where(missing[:, None], 0, ll)
    (evidence, _), (causal, _) = filter(initial, kwargs["transition_matrix"], effective)
    acausal = smoother(kwargs["transition_matrix"], causal)
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        chunk_size=3,
        is_missing=missing,
        # This test checks the original float32 evidence carry. Stable host
        # accumulation has its own analytic and posterior-equivalence tests.
        evidence_accumulation="reference",
        output_mode="spatial",
        result_path=tmp_path / "result",
        return_outputs=("causal_posterior", "acausal_posterior"),
        **kwargs,
    )
    np.testing.assert_allclose(
        result.dataset.causal_posterior, causal, rtol=1e-5, atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        result.dataset.acausal_posterior, acausal, rtol=1e-5, atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(result.marginal_log_likelihood, evidence, equal_nan=True)
    np.testing.assert_array_equal(result.dataset.is_missing, missing)
    if kind == "impossible":
        assert result.diagnostics["n_degenerate"] == 1
    if kind == "nan":
        assert result.diagnostics["n_nan"] == 1


def test_failure_leaves_no_completed_output_or_checkpoints(tmp_path):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    calls = 0

    def fail_during_replay(edges, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("writer upstream failure")
        return ll[kwargs["row_slice"]]

    with pytest.raises(RuntimeError, match="upstream failure"):
        checkpointed_forward_backward(
            edges,
            initial,
            fail_during_replay,
            state_ind=state_ind,
            chunk_size=4,
            output_mode="spatial",
            result_path=tmp_path / "result",
            **kwargs,
        )
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result-*"))
    assert not list(tmp_path.glob("checkpoints-*"))


def test_operator_forward_backward_protocol(tmp_path):
    edges, initial, ll, state_ind, kwargs, reference = problem()

    class Operator:
        def forward(self, p, discrete_weights=None):
            return p @ jnp.asarray(kwargs["transition_matrix"])

        def backward(self, v, discrete_weights=None):
            return jnp.asarray(kwargs["transition_matrix"]) @ v

    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        transition_operator=Operator(),
        state_ind=state_ind,
        chunk_size=4,
        output_mode="spatial",
        result_path=tmp_path / "result",
    )
    np.testing.assert_allclose(
        result.dataset.acausal_posterior, reference[3], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_bad_chunk_size_rejected_before_callback(bad):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    with pytest.raises(ValueError, match="chunk_size"):
        checkpointed_forward_backward(
            edges,
            initial,
            lambda *a, **kw: pytest.fail("called"),
            chunk_size=bad,
            **kwargs,
        )


def test_declared_zero_bin_state_and_no_implicit_causal_retention():
    edges, initial, ll, state_ind, kwargs, _ = problem()
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        n_states=3,
        **kwargs,
    )
    assert set(result.dataset.data_vars) == {"acausal_state_probabilities"}
    assert result.dataset.sizes["states"] == 3
    np.testing.assert_array_equal(result.dataset.acausal_state_probabilities[:, 2], 0)


def test_native_metadata_is_published_atomically_and_compact_has_no_spatial_dimension(
    tmp_path,
):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    metadata = {
        "_native_metadata": {
            "state_names": ["continuous", "fragmented"],
            "interior_mask": [True, False, True],
        }
    }
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        result_path=tmp_path / "result",
        result_attrs=metadata,
        **kwargs,
    )
    assert "state_bins" not in result.dataset.dims
    assert result.dataset.attrs["_native_metadata"] == metadata["_native_metadata"]
    assert (
        open_result_store(tmp_path / "result").attrs["_native_metadata"]
        == metadata["_native_metadata"]
    )


def test_changed_likelihood_replay_fails_atomically(tmp_path):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    counts = {}

    def callback(edges, *, row_slice, is_missing):
        counts[row_slice.start] = counts.get(row_slice.start, 0) + 1
        return ll[row_slice] + counts[row_slice.start]

    with pytest.raises(ValueError, match="changed during checkpoint replay"):
        checkpointed_forward_backward(
            edges,
            initial,
            callback,
            state_ind=state_ind,
            result_path=tmp_path / "result",
            chunk_size=4,
            **kwargs,
        )
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result-*"))


def test_checkpoint_directory_failure_leaves_no_output_staging(tmp_path):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    bad = tmp_path / "not-a-directory"
    bad.write_text("keep")
    with pytest.raises(FileExistsError):
        checkpointed_forward_backward(
            edges,
            initial,
            lambda edges, **kw: ll[kw["row_slice"]],
            state_ind=state_ind,
            result_path=tmp_path / "result",
            checkpoint_dir=bad,
            **kwargs,
        )
    assert bad.read_text() == "keep"
    assert not list(tmp_path.glob(".result-*"))


def test_float64_matches_the_original_dense_precision_path(tmp_path):
    import jax

    if not jax.config.x64_enabled:
        pytest.skip(
            "requires actual enabled float64; separately exercised with JAX_ENABLE_X64=1"
        )
    edges, initial, ll, state_ind, kwargs, _ = problem()
    initial = initial.astype("float64")
    ll = ll.astype("float64")
    matrix = kwargs["transition_matrix"].astype("float64")
    (evidence, _), (causal, _) = filter(initial, matrix, ll)
    acausal = smoother(matrix, causal)
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        transition_matrix=matrix,
        chunk_size=4,
        output_mode="spatial",
        result_path=tmp_path / "result",
        dtype=np.float64,
        evidence_accumulation="reference",
    )
    assert result.dataset.acausal_posterior.dtype == np.float64
    np.testing.assert_array_equal(result.dataset.acausal_posterior, acausal)
    np.testing.assert_array_equal(result.marginal_log_likelihood, evidence)


def test_result_centers_use_original_overflow_safe_native_formula():
    from non_local_detector.likelihoods.common import decode_bin_centers

    edges = np.array([1e308, 1e308 + 2e300, 1e308 + 4e300])
    result = checkpointed_forward_backward(
        edges,
        np.array([0.5, 0.5], np.float32),
        lambda edges, **kw: np.zeros(
            (kw["row_slice"].stop - kw["row_slice"].start, 2), np.float32
        ),
        state_ind=np.array([0, 1]),
        transition_matrix=np.eye(2, dtype=np.float32),
        chunk_size=1,
    )
    assert np.isfinite(result.dataset.time).all()
    np.testing.assert_array_equal(result.dataset.time, decode_bin_centers(edges, 0, 2))


def test_boundary_events_and_global_final_edge_keep_native_ownership(tmp_path):
    from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

    edges = np.arange(7, dtype=float) * 0.002
    spikes = edges.copy()
    rates = np.array([2, 20], dtype=np.float32)
    counts = np.array([1, 1, 1, 1, 1, 2])
    values = counts[:, None] * np.log(rates) - np.diff(edges)[:, None] * rates

    def likelihood(original_edges, *, row_slice, is_missing):
        selected = get_spikecount_per_time_bin(
            spikes, row_slice=row_slice, time_edges=original_edges
        )
        np.testing.assert_array_equal(selected, counts[row_slice])
        return (
            selected[:, None] * np.log(rates)
            - np.diff(original_edges)[row_slice, None] * rates
        )

    result = checkpointed_forward_backward(
        edges,
        np.array([0.5, 0.5], np.float32),
        likelihood,
        state_ind=np.array([0, 1]),
        transition_matrix=np.eye(2, dtype=np.float32),
        chunk_size=2,
        output_mode="spatial",
        result_path=tmp_path / "result",
        return_outputs=["acausal_posterior", "log_likelihood"],
    )
    (evidence, _), (causal, _) = filter(
        np.array([0.5, 0.5], np.float32),
        np.eye(2, dtype=np.float32),
        values.astype(np.float32),
    )
    np.testing.assert_array_equal(
        result.dataset.log_likelihood, values.astype(np.float32)
    )
    np.testing.assert_allclose(
        result.dataset.acausal_posterior,
        smoother(np.eye(2, dtype=np.float32), causal),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.marginal_log_likelihood, evidence, rtol=1e-5, atol=1e-6
    )


def test_global_diagnostic_indices_are_exact_once_and_packed_for_store(tmp_path):
    edges, initial, ll, state_ind, kwargs, _ = problem()
    ll[[1, 4, 11]] = -np.inf
    ll[7, 0] = np.nan
    missing = np.zeros(len(ll), bool)
    missing[4] = True
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        is_missing=missing,
        chunk_size=4,
        selected_rows=np.array([8, 9]),
        result_path=tmp_path / "result",
        **kwargs,
    )
    np.testing.assert_array_equal(result.diagnostics["degenerate_indices"], [1, 11])
    np.testing.assert_array_equal(result.diagnostics["nan_indices"], [7])
    assert result.diagnostics["n_degenerate"] == 2
    assert result.diagnostics["n_nan"] == 1
    assert result.diagnostics["degenerate_indices"].dtype.kind == "i"
    assert "degenerate_indices" not in result.dataset.attrs
    loaded = open_result_store(tmp_path / "result")

    def indices(key):
        packed = np.frombuffer(bytes.fromhex(loaded.attrs[key]), dtype=np.uint8)
        return np.flatnonzero(np.unpackbits(packed, bitorder="little", count=len(ll)))

    np.testing.assert_array_equal(indices("degenerate_row_mask_hex"), [1, 11])
    np.testing.assert_array_equal(indices("nan_row_mask_hex"), [7])


def test_all_impossible_diagnostics_use_numpy_metadata_not_per_row_python_objects():
    edges, initial, ll, state_ind, kwargs, _ = problem()
    ll[:] = -np.inf
    result = checkpointed_forward_backward(
        edges,
        initial,
        lambda edges, **kw: ll[kw["row_slice"]],
        state_ind=state_ind,
        chunk_size=1,
        **kwargs,
    )
    np.testing.assert_array_equal(
        result.diagnostics["degenerate_indices"], np.arange(len(ll))
    )
    assert (
        result.diagnostics["degenerate_indices"].nbytes
        == len(ll) * np.dtype(np.intp).itemsize
    )
    assert len(bytes.fromhex(result.dataset.attrs["degenerate_row_mask_hex"])) == int(
        np.ceil(len(ll) / 8)
    )


@pytest.mark.parametrize(
    "budget", [0, -1, True, False, 1.5, np.float64(512), "1024", None]
)
@pytest.mark.parametrize(
    "mode,path_enabled", [("compact", False), ("compact", True), ("spatial", True)]
)
def test_read_budget_is_validated_before_callbacks_or_files(
    tmp_path, budget, mode, path_enabled
):
    path = tmp_path / "result" if path_enabled else None
    with pytest.raises(ValueError, match="max_read_bytes"):
        checkpointed_forward_backward(
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5], np.float32),
            lambda *args, **kwargs: pytest.fail("callback called before validation"),
            state_ind=np.array([0, 1]),
            transition_matrix=np.eye(2, dtype=np.float32),
            output_mode=mode,
            result_path=path,
            max_read_bytes=budget,
        )
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result-*"))


def test_numpy_integer_read_budget_is_normalized(tmp_path):
    result = checkpointed_forward_backward(
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5], np.float32),
        lambda *args, **kwargs: np.zeros((1, 2), np.float32),
        state_ind=np.array([0, 1]),
        transition_matrix=np.eye(2, dtype=np.float32),
        result_path=tmp_path / "result",
        max_read_bytes=np.int64(16),
    )
    np.testing.assert_array_equal(
        result.dataset.acausal_state_probabilities, [[0.5, 0.5]]
    )


def test_production_grid_uniform_marginals_preserve_pairwise_normalization(tmp_path):
    n_spatial = 182 * 182
    sizes = (1, 1, n_spatial, n_spatial)
    state_ind = np.repeat(np.arange(4), sizes)
    initial = np.full(len(state_ind), 1 / len(state_ind), dtype=np.float32)

    class Identity:
        def forward(self, values, discrete_weights=None):
            return values

        def backward(self, values, discrete_weights=None):
            return values

    result = checkpointed_forward_backward(
        np.array([0.0, 0.002]),
        initial,
        lambda edges, **kwargs: np.zeros((1, len(initial)), dtype=np.float32),
        state_ind=state_ind,
        transition_operator=Identity(),
        chunk_size=1,
        output_mode="spatial",
        result_path=tmp_path / "result",
        return_outputs=["acausal_posterior", "acausal_state_probabilities"],
    )
    spatial = result.dataset.acausal_posterior.values[0]
    expected = np.array(
        [np.sum(spatial[state_ind == state], dtype=np.float32) for state in range(4)]
    )
    actual = result.dataset.acausal_state_probabilities.values[0]
    np.testing.assert_allclose(
        np.sum(spatial, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(actual, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6
    )


def test_descending_unsigned_selected_rows_are_rejected_before_callbacks():
    with pytest.raises(ValueError, match="unique, increasing"):
        checkpointed_forward_backward(
            np.arange(7, dtype=float),
            np.ones(1, np.float32),
            lambda *args, **kwargs: pytest.fail("callback before validation"),
            state_ind=np.zeros(1, np.int32),
            transition_matrix=np.ones((1, 1), np.float32),
            selected_rows=np.array([5, 1], np.uint8),
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("chunk_size", [1, 4, 13, 20])
@pytest.mark.parametrize("kind", ["finite", "impossible", "nan", "positive_inf"])
def test_stable_preserves_dense_probabilities(tmp_path, dtype, chunk_size, kind):
    import jax

    if dtype == np.float64 and not jax.config.x64_enabled:
        pytest.skip("requires actual enabled float64")
    rng = np.random.default_rng(7701)
    initial = np.array([0.2, 0.3, 0.1, 0.4], dtype=dtype)
    transition = rng.uniform(0.05, 1, (4, 4)).astype(dtype)
    transition /= transition.sum(1, keepdims=True)
    ll = rng.normal(size=(13, 4)).astype(dtype)
    if kind == "impossible":
        ll[4] = -np.inf
    elif kind == "nan":
        ll[4, 1] = np.nan
    elif kind == "positive_inf":
        ll[4, 1] = np.inf
    (evidence, _), (causal, predictive) = filter(initial, transition, ll)
    acausal = smoother(transition, causal)
    result = checkpointed_forward_backward(
        np.arange(14, dtype=float) * 0.002,
        initial,
        lambda edges, row_slice, **kwargs: ll[row_slice],
        state_ind=np.array([0, 0, 1, 1]),
        transition_matrix=transition,
        chunk_size=chunk_size,
        output_mode="spatial",
        result_path=tmp_path / "stable",
        dtype=dtype,
        return_outputs=[
            "causal_posterior",
            "predictive_posterior",
            "acausal_posterior",
        ],
    )
    reference = checkpointed_forward_backward(
        np.arange(14, dtype=float) * 0.002,
        initial,
        lambda edges, row_slice, **kwargs: ll[row_slice],
        state_ind=np.array([0, 0, 1, 1]),
        transition_matrix=transition,
        chunk_size=chunk_size,
        output_mode="spatial",
        result_path=tmp_path / "reference",
        evidence_accumulation="reference",
        dtype=dtype,
        return_outputs=[
            "causal_posterior",
            "predictive_posterior",
            "acausal_posterior",
        ],
    )
    for name, dense in [
        ("causal_posterior", causal),
        ("predictive_posterior", predictive),
        ("acausal_posterior", acausal),
    ]:
        np.testing.assert_array_equal(
            result.dataset[name].values, reference.dataset[name].values
        )
        np.testing.assert_allclose(
            result.dataset[name].values, dense, rtol=1e-5, atol=1e-6, equal_nan=True
        )
    np.testing.assert_allclose(
        result.marginal_log_likelihood, evidence, rtol=1e-5, atol=1e-6, equal_nan=True
    )
    assert result.dataset.attrs["evidence_accumulation"] == "stable"
    assert result.dataset.attrs["evidence_dtype"] == "float64"
    assert result.dataset.attrs["state_probability_dtype"] == np.dtype(dtype).name


@pytest.mark.parametrize(
    "chunks",
    [
        [1.0, 2.0, 3.0],
        [-np.inf, 1.0],
        [np.inf, 1.0],
        [-np.inf, np.inf],
        [np.nan, -np.inf],
        [np.inf, np.nan],
    ],
)
def test_ieee_nonfinite_sum(chunks):
    from non_local_detector.checkpointed_inference import _sum_evidence

    with np.errstate(invalid="ignore"):
        expected = np.sum(chunks, dtype=np.float64)
    np.testing.assert_allclose(_sum_evidence(iter(chunks)), expected, equal_nan=True)


@pytest.mark.parametrize("chunk_size", [1, 256, 16384])
def test_million_row_streaming_accumulator(chunk_size):
    from non_local_detector.checkpointed_inference import _sum_evidence

    n_rows = 1_800_000
    increment = np.float32(-0.02)

    def chunks():
        for start in range(0, n_rows, chunk_size):
            rows = min(chunk_size, n_rows - start)
            yield np.sum(np.full(rows, increment, dtype=np.float32), dtype=np.float64)

    expected = n_rows * float(increment)
    assert _sum_evidence(chunks()) == expected


@pytest.mark.parametrize("chunk_size", [256, 16384])
def test_million_row_full_driver(chunk_size):
    n_rows = 1_800_000
    increment = np.float32(-0.02)
    result = checkpointed_forward_backward(
        np.arange(n_rows + 1, dtype=float) * 0.002,
        np.ones(1, np.float32),
        lambda edges, row_slice, **kwargs: np.full(
            (row_slice.stop - row_slice.start, 1), increment, np.float32
        ),
        transition_matrix=np.ones((1, 1), np.float32),
        state_ind=np.zeros(1, np.int32),
        chunk_size=chunk_size,
        selected_rows=np.array([n_rows - 1]),
    )
    assert result.marginal_log_likelihood == n_rows * float(increment)
    np.testing.assert_array_equal(result.dataset.acausal_state_probabilities, [[1.0]])


@pytest.mark.parametrize("tail", [[], [np.nan], [-np.inf], [np.inf]])
def test_stable_fsum_overflow_exhausts_input_and_preserves_ieee_total(tail):
    from non_local_detector.checkpointed_inference import _sum_evidence

    values = [1e308, 1e308, *tail]
    visited = []
    reference = 0.0
    for value in values:
        reference += value

    def chunks():
        for value in values:
            visited.append(value)
            yield value

    np.testing.assert_allclose(_sum_evidence(chunks()), reference, equal_nan=True)
    assert len(visited) == len(values)


@pytest.mark.parametrize("tail", [[], [np.nan], [-np.inf]])
def test_float64_evidence_overflow_keeps_full_replay_and_cleanup(tmp_path, tail):
    import jax

    if not jax.config.x64_enabled:
        pytest.skip("requires actual enabled float64")
    ll = np.array([1e308, 1e308, *tail], np.float64)[:, None]
    calls = []

    def likelihood(edges, row_slice, **kwargs):
        calls.append(row_slice.start)
        return ll[row_slice]

    (reference, _), (causal, _) = filter(
        np.ones(1, np.float64), np.ones((1, 1), np.float64), ll
    )
    expected = smoother(np.ones((1, 1), np.float64), causal)
    result = checkpointed_forward_backward(
        np.arange(len(ll) + 1, dtype=float) * 0.002,
        np.ones(1, np.float64),
        likelihood,
        transition_matrix=np.ones((1, 1), np.float64),
        state_ind=np.zeros(1, np.int32),
        chunk_size=1,
        dtype=np.float64,
        checkpoint_dir=tmp_path,
    )
    np.testing.assert_allclose(
        result.marginal_log_likelihood, reference, equal_nan=True
    )
    np.testing.assert_array_equal(result.dataset.acausal_state_probabilities, expected)
    assert len(calls) == 2 * len(ll)
    assert not list(tmp_path.glob("checkpoints-*"))
