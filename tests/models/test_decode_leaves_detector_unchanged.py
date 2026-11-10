"""Decoding reads a fitted detector and never changes it.

``predict``, ``most_likely_sequence`` and ``compute_log_likelihood`` return
their outputs; every attribute of the detector stays the same object with the
same value, so one fitted detector can decode several recordings in turn.
"""

import pickle

import numpy as np
import pytest

from tests.models._helpers import _assert_unchanged, _snapshot

pytestmark = pytest.mark.integration


def _decode(model, predict, method, tmp_path):
    if method == "predict":
        return model.predict(**predict, return_outputs="all")
    if method == "predict_chunked":
        return model.predict(**predict, n_chunks=3, cache_likelihood=False)
    if method == "predict_checkpointed_compact":
        return model.predict(
            **predict, inference_mode="checkpointed", output_mode="compact"
        )
    if method == "predict_checkpointed_spatial":
        return model.predict(
            **predict,
            inference_mode="checkpointed",
            output_mode="spatial",
            result_path=tmp_path / "result",
            chunk_size=13,
            return_outputs="all",
        )
    if method == "most_likely_sequence":
        return model.most_likely_sequence(**predict)
    arguments = dict(predict)
    edges = arguments.pop("time_edges")
    return model.compute_log_likelihood(
        arguments.pop("position_time"),
        arguments.pop("position"),
        *arguments.values(),
        time_edges=edges,
    )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize(
    "method",
    [
        "predict",
        "predict_chunked",
        "predict_checkpointed_compact",
        "predict_checkpointed_spatial",
        "most_likely_sequence",
        "compute_log_likelihood",
    ],
)
def test_decoding_leaves_every_attribute_unchanged(
    checkpoint_recording, tmp_path, family, method
):
    model, fit, predict = checkpoint_recording(family)
    model.fit(**fit)
    # Values as well as identities: an in-place edit keeps the identity.
    snapshot = _snapshot(model)
    values = pickle.dumps(vars(model), protocol=pickle.HIGHEST_PROTOCOL)

    output = _decode(model, predict, method, tmp_path)

    assert output is not None
    assert vars(model).keys() == snapshot[0].keys(), set(vars(model)) ^ set(snapshot[0])
    _assert_unchanged(model, snapshot)
    assert pickle.dumps(vars(model), protocol=pickle.HIGHEST_PROTOCOL) == values
    # Decoding again gives the same result: nothing carried over.
    if method == "compute_log_likelihood":
        again = _decode(model, predict, method, tmp_path / "again")
        np.testing.assert_array_equal(np.asarray(again), np.asarray(output))


@pytest.mark.unit
def test_loading_drops_the_degenerate_bins_older_predictions_recorded(
    checkpoint_recording,
):
    """Older versions recorded each prediction's degenerate bins on the
    detector; a detector they saved loads without them."""
    model, fit, _ = checkpoint_recording("sorted")
    model.fit(**fit)
    model._degenerate_timesteps_ = np.array([3, 5])

    loaded = pickle.loads(pickle.dumps(model))

    assert not hasattr(loaded, "_degenerate_timesteps_")
    assert vars(loaded).keys() == vars(model).keys() - {"_degenerate_timesteps_"}
