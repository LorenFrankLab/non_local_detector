"""Native checkpointed predictions keep exact global diagnostic row ownership."""

from functools import wraps

import numpy as np
import pytest

from non_local_detector.tests.models.test_phase7_prediction import recording

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("output_mode", ["compact", "spatial"])
@pytest.mark.parametrize("neutralize_missing", [False, True])
def test_native_diagnostics_keep_rows_outside_selected_outputs(
    tmp_path,
    monkeypatch,
    family,
    output_mode,
    neutralize_missing,
):
    model, fit, predict = recording(family)
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
        if start <= 73 < stop:
            values[73 - start, 0] = np.nan
        return values

    monkeypatch.setattr(model, "compute_log_likelihood", damaged)
    # Prediction refreshes the private diagnostic for its own recording, while
    # the public final-E-step attribute continues to describe training data.
    model.degenerate_timesteps_ = np.array([98])
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
    np.testing.assert_array_equal(model._degenerate_timesteps_, expected)
    np.testing.assert_array_equal(model.degenerate_timesteps_, [98])
    np.testing.assert_array_equal(result.source_row, np.arange(5, 10))
    assert result.attrs["n_degenerate"] == len(expected)
    assert result.attrs["n_nan"] == 1
    assert result.attrs["diagnostics_are_global"] is True
    loaded = model.load_results(path)
    for key, indices in [
        ("degenerate_row_mask_hex", expected),
        ("nan_row_mask_hex", [73]),
    ]:
        packed = np.frombuffer(bytes.fromhex(loaded.attrs[key]), dtype=np.uint8)
        actual = np.flatnonzero(np.unpackbits(packed, bitorder="little", count=100))
        np.testing.assert_array_equal(actual, indices)
    if output_mode == "spatial":
        assert not loaded.acausal_posterior.variable._in_memory
