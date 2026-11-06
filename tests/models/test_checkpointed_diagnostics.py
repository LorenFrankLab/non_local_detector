"""Dense and checkpointed predictions keep exact global diagnostic row ownership."""

from functools import wraps

import numpy as np
import pytest

from non_local_detector.models._em import _EMDiagnostics

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("output_mode", ["compact", "spatial"])
@pytest.mark.parametrize("neutralize_missing", [False, True])
def test_native_diagnostics_keep_rows_outside_selected_outputs(
    checkpoint_recording,
    tmp_path,
    monkeypatch,
    family,
    output_mode,
    neutralize_missing,
):
    model, fit, predict = checkpoint_recording(family)
    model.fit(**fit)
    original = model.compute_log_likelihood

    @wraps(original)
    def damaged(*args, **kwargs):
        values = np.asarray(original(*args, **kwargs)).copy()
        rows = kwargs.get("row_slice")
        start = 0 if rows is None else rows.start
        stop = 100 if rows is None else rows.stop
        for row in (7, 45):
            if start <= row < stop:
                values[row - start, :] = -np.inf
        return values

    monkeypatch.setattr(model, "compute_log_likelihood", damaged)
    # The result describes the predicted recording; the public final-E-step
    # attribute continues to describe training data.
    model._em_diagnostics_ = _EMDiagnostics(0, False, (), np.array([98]))
    missing = np.zeros(100, bool)
    missing[45] = neutralize_missing
    expected = np.array([7] if neutralize_missing else [7, 45])
    path = tmp_path / "result"
    result = model.predict(
        **predict,
        is_missing=missing,
        inference_mode="checkpointed",
        output_mode=output_mode,
        result_path=path,
        chunk_size=13,
        selected_intervals=[[0.1, 0.2]],
    )
    np.testing.assert_array_equal(model.degenerate_timesteps_, [98])
    np.testing.assert_array_equal(result.source_row, np.arange(5, 10))
    assert result.attrs["n_degenerate"] == len(expected)
    assert result.attrs["diagnostics_are_global"] is True
    loaded = model.load_results(path)
    packed = np.frombuffer(
        bytes.fromhex(loaded.attrs["degenerate_row_mask_hex"]), dtype=np.uint8
    )
    actual = np.flatnonzero(np.unpackbits(packed, bitorder="little", count=100))
    np.testing.assert_array_equal(actual, expected)
    if output_mode == "spatial":
        assert not loaded.acausal_posterior.variable._in_memory


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
# max_iter=0 runs only the final E-step, which never converges.
@pytest.mark.filterwarnings("ignore:EM did not converge after max_iter=0:UserWarning")
def test_dense_chunked_estimation_records_degenerate_rows_of_every_chunk(
    checkpoint_recording, monkeypatch, family
):
    """Dense EM's final E-step collects all-``-inf`` rows from every chunk.

    With ``n_chunks=3`` the likelihood is evaluated per chunk and not cached,
    so the indices come only from the driver's ``degenerate_indices``.
    """
    model, fit, predict = checkpoint_recording(family)
    model.fit(**fit)
    original = model.compute_log_likelihood
    degenerate_rows = [7, 45, 80]  # chunks [0, 34), [34, 67), [67, 100)

    @wraps(original)
    def damaged(*args, **kwargs):
        values = np.asarray(original(*args, **kwargs)).copy()
        rows = kwargs.get("row_slice")
        start = 0 if rows is None else rows.start
        stop = 100 if rows is None else rows.stop
        for row in degenerate_rows:
            if start <= row < stop:
                values[row - start, :] = -np.inf
        requested.append((start, stop))
        return values

    requested = []
    monkeypatch.setattr(model, "compute_log_likelihood", damaged)
    model.estimate_parameters(
        **predict,
        transition_representation="dense",
        n_chunks=3,
        cache_likelihood=False,
        max_iter=0,
    )
    assert requested == [(0, 34), (34, 67), (67, 100)]
    np.testing.assert_array_equal(model.degenerate_timesteps_, degenerate_rows)
    assert model.degenerate_timesteps_.dtype.kind == "i"
