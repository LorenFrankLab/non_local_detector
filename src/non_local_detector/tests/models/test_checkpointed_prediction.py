"""Public performance modes preserve the original conditioned prediction."""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    RandomWalk,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("representation", ["dense", "structured", "auto"])
def test_compact_prediction_matches_full_conditioning(
    checkpoint_recording, tmp_path, family, representation
):
    model, fit, predict = checkpoint_recording(family)
    model.fit(**fit)
    reference = model.predict(**predict, return_outputs="all")
    model.fit(**fit, transition_representation=representation)
    compact = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode="compact",
        chunk_size=13,
        checkpoint_dir=tmp_path / "checkpoints",
        return_outputs=["filter", "predictive"],
    )
    assert "state_bins" not in compact.dims
    for name in [
        "acausal_state_probabilities",
        "causal_state_probabilities",
        "predictive_state_probabilities",
    ]:
        np.testing.assert_allclose(compact[name], reference[name], rtol=1e-6, atol=1e-6)
    for name in ["time", "time_bin_start", "time_bin_end", "is_missing"]:
        np.testing.assert_array_equal(compact[name], reference[name])
    np.testing.assert_allclose(
        compact.attrs["marginal_log_likelihoods"],
        reference.attrs["marginal_log_likelihoods"],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_incremental_selected_rows_keep_entire_recording_context(
    checkpoint_recording, tmp_path, family
):
    model, fit, predict = checkpoint_recording(family)
    model.fit(**fit)
    reference = model.predict(**predict, return_outputs="all")
    model.fit(**fit, transition_representation="structured")
    result = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode="spatial",
        result_path=tmp_path / "result",
        chunk_size=13,
        selected_intervals=[[0.1, 0.3], [1.1, 1.4]],
        return_outputs="all",
    )
    selected = reference.sel(time=result.time)
    for name in [
        "acausal_posterior",
        "causal_posterior",
        "predictive_posterior",
        "log_likelihood",
    ]:
        np.testing.assert_allclose(
            result[name].values, selected[name].values, rtol=1e-6, atol=1e-6
        )
    assert (tmp_path / "result" / "manifest.json").exists()
    assert result.attrs["conditioning"] == "whole_recording"
    loaded = model.load_results(tmp_path / "result")
    for name in result.data_vars:
        np.testing.assert_allclose(
            loaded[name].values, result[name].values, rtol=1e-6, atol=1e-6
        )
    assert loaded.indexes["state_bins"].equals(result.indexes["state_bins"])
    for name in [
        "time_bin_start",
        "time_bin_end",
        "time_bin_end_inclusive",
        "is_missing",
    ]:
        np.testing.assert_array_equal(loaded[name], result[name])


def test_checkpoint_options_do_not_silently_use_dense_prediction(
    checkpoint_recording, tmp_path
):
    model, fit, predict = checkpoint_recording("sorted")
    model.fit(**fit)
    with pytest.raises(ValueError, match="checkpointed"):
        model.predict(**predict, output_mode="compact")
    with pytest.raises(ValueError, match="result_path"):
        model.predict(**predict, inference_mode="checkpointed", output_mode="spatial")


@pytest.mark.parametrize("options", [{"cache_likelihood": True}, {"n_chunks": 2}])
def test_checkpointed_prediction_rejects_dense_chunk_controls(
    checkpoint_recording, options
):
    model, fit, predict = checkpoint_recording("sorted")
    model.fit(**fit)
    with pytest.raises(ValueError, match="chunk_size"):
        model.predict(
            **predict, inference_mode="checkpointed", output_mode="compact", **options
        )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_structured_model_pickle_roundtrip(checkpoint_recording, tmp_path, family):
    model, fit, predict = checkpoint_recording(family)
    model.fit(**fit, transition_representation="structured")
    path = tmp_path / "model.pkl"
    model.save_model(path)
    restored = model.load_model(path)
    expected = model.predict(
        **predict, inference_mode="checkpointed", output_mode="compact", chunk_size=13
    )
    actual = restored.predict(
        **predict, inference_mode="checkpointed", output_mode="compact", chunk_size=13
    )
    np.testing.assert_array_equal(
        actual.acausal_state_probabilities, expected.acausal_state_probabilities
    )
    assert type(restored) is type(model)
    assert path.stat().st_size < model.n_state_bins_**2 * 8 + 100_000


def test_compact_mode_rejects_requested_spatial_arrays(checkpoint_recording, tmp_path):
    model, fit, predict = checkpoint_recording("sorted")
    model.fit(**fit)
    with pytest.raises(ValueError, match="Spatial outputs"):
        model.predict(
            **predict,
            inference_mode="checkpointed",
            output_mode="compact",
            return_outputs="log_likelihood",
        )


def test_compact_predictive_shorthand_returns_state_probabilities(checkpoint_recording):
    model, fit, predict = checkpoint_recording("sorted")
    model.fit(**fit, transition_representation="structured")
    expected = model.predict(**predict, return_outputs="predictive")
    actual = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode="compact",
        chunk_size=13,
        return_outputs="predictive",
    )
    assert "predictive_posterior" not in actual
    np.testing.assert_allclose(
        actual.predictive_state_probabilities,
        expected.predictive_state_probabilities,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("output_mode", ["compact", "spatial"])
def test_small_checkpointed_results_export_to_netcdf(
    checkpoint_recording, tmp_path, output_mode
):
    model, fit, predict = checkpoint_recording("sorted")
    model.fit(**fit, transition_representation="structured")
    result = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode=output_mode,
        result_path=tmp_path / "store" if output_mode == "spatial" else None,
        chunk_size=13,
    )
    path = tmp_path / "export.nc"
    model.save_results(result, path)
    restored = model.load_results(path)
    for name in result.data_vars:
        np.testing.assert_array_equal(restored[name].values, result[name].values)
    if output_mode == "spatial":
        assert restored.indexes["state_bins"].equals(result.indexes["state_bins"])
    else:
        assert "state_bins" not in restored.dims
    assert restored.attrs["conditioning"] == "whole_recording"


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_checkpointed_covariate_transitions_keep_global_rows(
    checkpoint_recording, tmp_path, family
):
    model, fit, predict = checkpoint_recording(family)
    model.discrete_transition_type = DiscreteNonStationaryDiagonal(
        np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + speed"
    )
    fit["discrete_transition_covariate_data"] = {"speed": np.linspace(0, 3, 61)}
    predict["discrete_transition_covariate_data"] = {
        "speed": np.sin(np.arange(100) / 5)
    }
    model.fit(**fit)
    model.discrete_transition_coefficients_[1] = np.arange(12).reshape(4, 3) / 10
    reference = model.predict(**predict, return_outputs="all")
    model.fit(**fit, transition_representation="structured")
    model.discrete_transition_coefficients_[1] = np.arange(12).reshape(4, 3) / 10
    fitted_transitions = model.discrete_state_transitions_.copy()
    result = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode="spatial",
        chunk_size=13,
        result_path=tmp_path / "covariate",
        return_outputs="all",
    )
    for name in reference.data_vars:
        np.testing.assert_allclose(result[name], reference[name], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(model.discrete_state_transitions_, fitted_transitions)


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("local_std", [0.0, 1.5])
def test_structured_local_kernel_and_holes_match_dense(
    checkpoint_recording, tmp_path, family, local_std
):
    model, fit, predict = checkpoint_recording(family)
    mask = np.ones((7, 7), dtype=bool)
    mask[2, 3] = mask[4, 4] = False
    model.environments[0].is_track_interior = mask
    model.environments[0].is_track_interior_ = mask
    model.local_position_std = local_std
    model.fit(**fit)
    reference = model.predict(**predict)
    model.fit(**fit, transition_representation="structured")
    result = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode="spatial",
        chunk_size=13,
        result_path=tmp_path / "holes",
    )
    np.testing.assert_allclose(
        result.acausal_posterior, reference.acausal_posterior, rtol=1e-6, atol=1e-6
    )
    assert np.isnan(
        result.acausal_posterior[:, ~model.is_track_interior_state_bins_]
    ).all()


def test_structured_fit_avoids_all_pairs_graph_and_dense_gaussian(
    checkpoint_recording, monkeypatch
):
    import networkx as nx

    import non_local_detector.continuous_state_transitions as transitions

    model, fit, _ = checkpoint_recording("sorted")

    def forbidden(*args, **kwargs):
        raise AssertionError("quadratic setup used")

    monkeypatch.setattr(nx, "shortest_path_length", forbidden)
    monkeypatch.setattr(transitions, "_euclidean_random_walk", forbidden)
    model.fit(**fit, transition_representation="structured")


def test_nonseparable_transition_is_explicit_and_auto_fallback_is_budgeted(
    checkpoint_recording,
):
    from non_local_detector.transition_operators import (
        DenseTransitionBudgetError,
        UnsupportedTransitionError,
    )

    model, fit, predict = checkpoint_recording("sorted")
    for row in model.continuous_transition_types:
        for transition in row:
            if isinstance(transition, RandomWalk):
                transition.movement_var = np.array([[6.0, 0.5], [0.5, 6.0]])
    with pytest.raises(UnsupportedTransitionError):
        model.fit(**fit, transition_representation="structured")
    with pytest.raises(DenseTransitionBudgetError):
        model.fit(**fit, transition_representation="auto", max_dense_transition_bytes=1)
    model.fit(**fit)
    reference = model.predict(**predict)
    model.fit(**fit, transition_representation="auto")
    result = model.predict(
        **predict, inference_mode="checkpointed", output_mode="compact", chunk_size=13
    )
    np.testing.assert_allclose(
        result.acausal_state_probabilities,
        reference.acausal_state_probabilities,
        rtol=1e-6,
        atol=1e-6,
    )


def test_native_store_read_budget_can_be_set_when_reopening(
    checkpoint_recording, tmp_path
):
    model, fit, predict = checkpoint_recording("sorted")
    model.fit(**fit, transition_representation="structured")
    store = tmp_path / "posterior"
    result = model.predict(
        **predict,
        inference_mode="checkpointed",
        output_mode="spatial",
        result_path=store,
        chunk_size=13,
    )
    limited = model.load_results(store, max_read_bytes=64)
    with pytest.raises(MemoryError, match="read"):
        _ = limited.acausal_posterior.values
    np.testing.assert_array_equal(
        limited.acausal_posterior[:1, :3], result.acausal_posterior[:1, :3]
    )
    np.testing.assert_array_equal(
        model.load_results(store, max_read_bytes=100_000).acausal_posterior,
        result.acausal_posterior,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"transition_representation": "Dense"},
        {"transition_representation": "structurd"},
        {"max_dense_transition_bytes": True},
        {"max_dense_transition_bytes": 0},
        {"max_dense_transition_bytes": -1},
    ],
)
def test_fit_rejects_invalid_transition_arguments_before_refit(
    checkpoint_recording, arguments, monkeypatch
):
    from non_local_detector.exceptions import ValidationError

    model, fit, _ = checkpoint_recording("sorted")

    def refit(*args, **kwargs):
        raise AssertionError("environments were refit before validation")

    monkeypatch.setattr(model, "initialize_environments", refit)
    with pytest.raises(ValidationError):
        model.fit(**fit, **arguments)


def test_fit_accepts_numpy_integer_transition_budget(checkpoint_recording):
    model, fit, _ = checkpoint_recording("sorted")
    model.fit(**fit, max_dense_transition_bytes=np.int64(2**20))
    assert model._max_dense_transition_bytes_ == 2**20
