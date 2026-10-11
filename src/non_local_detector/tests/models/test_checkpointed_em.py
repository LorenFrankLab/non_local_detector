"""Checkpointed EM over structured transitions reproduces dense EM."""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)

pytestmark = pytest.mark.integration

N_BINS = 300


def _recording(family):
    """Three place cells on a 2-D arena, 6 s, 20 ms decode bins."""
    rng = np.random.default_rng(4)
    position_time = np.linspace(0, 6, 301)
    position = np.column_stack(
        (5 + 4 * np.sin(position_time), 5 + 4 * np.cos(position_time / 2))
    )
    centers = np.array([[2.0, 5.0], [8.0, 5.0], [5.0, 8.0]])
    spike_times = []
    for center in centers:
        rate = 40 * np.exp(-np.sum((position - center) ** 2, axis=1) / 4)
        spike_times.append(np.sort(position_time[rng.poisson(rate / 50) > 0]))
    environment = Environment(place_bin_size=2, position_range=((0, 10), (0, 10)))
    kwargs = {"environments": environment, "infer_track_interior": False}
    if family == "sorted":
        model = NonLocalSortedSpikesDetector(**kwargs)
        data = {"spike_times": spike_times}
    else:
        model = NonLocalClusterlessDetector(**kwargs)
        data = {
            "spike_times": spike_times,
            "spike_waveform_features": [
                rng.normal(10 * cell, 3, (len(times), 2))
                for cell, times in enumerate(spike_times)
            ],
        }
    data.update(
        position_time=position_time,
        position=position,
        time_edges=np.linspace(0, 6, N_BINS + 1),
    )
    return model, data


def _fit(model, data, **kwargs):
    results = model.estimate_parameters(**data, max_iter=3, tolerance=1e-12, **kwargs)
    # A structured or auto fit must have run on its transition operator.
    assert (getattr(model, "_continuous_transition_operator_", None) is not None) == (
        "transition_representation" in kwargs
    )
    return results, {
        "discrete_state_transitions_": np.array(model.discrete_state_transitions_),
        "initial_conditions_": np.array(model.initial_conditions_),
        "n_iter_": model.n_iter_,
    }


def _assert_same_fit(structured, dense, rtol=1e-5, atol=1e-6):
    (results, fitted), (reference, reference_fitted) = structured, dense
    assert fitted["n_iter_"] == reference_fitted["n_iter_"]
    for name in ("discrete_state_transitions_", "initial_conditions_"):
        np.testing.assert_allclose(
            fitted[name], reference_fitted[name], rtol=rtol, atol=atol
        )
    np.testing.assert_allclose(
        results.attrs["marginal_log_likelihoods"],
        reference.attrs["marginal_log_likelihoods"],
        rtol=rtol,
    )
    assert set(results.data_vars) == set(reference.data_vars)
    for name in reference.data_vars:
        np.testing.assert_allclose(
            results[name].values, reference[name].values, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_checkpointed_em_matches_dense_em(family, monkeypatch):
    model, data = _recording(family)
    dense = _fit(model, data)

    from non_local_detector import checkpointed_inference

    calls = []
    driver = checkpointed_inference.checkpointed_forward_backward

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return driver(*args, **kwargs)

    monkeypatch.setattr(checkpointed_inference, "checkpointed_forward_backward", spy)
    model, data = _recording(family)
    structured = _fit(
        model, data, transition_representation="structured", chunk_size=37
    )
    _assert_same_fit(structured, dense)
    # Every E-step, including the final one, ran on the structured operator.
    assert len(calls) == model.n_iter_ + 1
    assert all("transition_operator" in kwargs for kwargs in calls)
    assert [kwargs["transition_statistics"] for kwargs in calls] == [
        "counts"
    ] * model.n_iter_ + [None]


def test_checkpointed_em_returns_the_requested_outputs():
    model, data = _recording("sorted")
    dense = _fit(model, data, return_outputs="all", store_log_likelihood=True)
    reference_log_likelihood = np.asarray(model.log_likelihood_)
    model, data = _recording("sorted")
    structured = _fit(
        model,
        data,
        return_outputs="all",
        store_log_likelihood=True,
        transition_representation="structured",
        # The last chunk holds only the final recording row.
        chunk_size=N_BINS - 1,
    )
    _assert_same_fit(structured, dense)
    np.testing.assert_allclose(
        model.log_likelihood_, reference_log_likelihood, rtol=1e-6, atol=1e-6
    )


def test_checkpointed_em_with_covariate_transitions():
    covariates = {"speed": np.sin(np.arange(N_BINS) / 7)}

    def run(**kwargs):
        model, data = _recording("sorted")
        model.discrete_transition_type = DiscreteNonStationaryDiagonal(
            np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + speed"
        )
        return _fit(
            model, data, discrete_transition_covariate_data=covariates, **kwargs
        )

    _assert_same_fit(run(transition_representation="structured", chunk_size=50), run())


@pytest.mark.parametrize(
    "flags",
    [
        {"estimate_discrete_transition": False},
        {"estimate_initial_conditions": False},
        {"estimate_encoding_model": False},
    ],
)
def test_checkpointed_em_honors_estimation_flags(flags):
    model, data = _recording("sorted")
    dense = _fit(model, data, **flags)
    model, data = _recording("sorted")
    structured = _fit(
        model, data, transition_representation="structured", chunk_size=41, **flags
    )
    _assert_same_fit(structured, dense)


def test_checkpointed_em_with_missing_bins():
    is_missing = np.zeros(N_BINS, dtype=bool)
    is_missing[100:130] = True
    model, data = _recording("sorted")
    dense = _fit(model, data, is_missing=is_missing)
    model, data = _recording("sorted")
    structured = _fit(
        model,
        data,
        is_missing=is_missing,
        transition_representation="structured",
        chunk_size=29,
    )
    _assert_same_fit(structured, dense)


@pytest.mark.parametrize("representation", ["structured", "auto"])
def test_structured_em_rejects_row_chunks(representation):
    model, data = _recording("sorted")
    with pytest.raises(ValueError, match="chunk_size"):
        model.estimate_parameters(
            **data, max_iter=1, n_chunks=4, transition_representation=representation
        )


def test_dense_em_rejects_checkpoint_chunks():
    """chunk_size must not silently run unbounded dense EM."""
    model, data = _recording("sorted")
    with pytest.raises(ValueError, match="structured"):
        model.estimate_parameters(**data, max_iter=1, chunk_size=50)


def _track_recording():
    """An L-shaped track graph and one sorted cell per arm."""
    import networkx as nx

    rng = np.random.default_rng(9)
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(20.0, 0.0))
    graph.add_node(2, pos=(20.0, 20.0))
    graph.add_edge(0, 1, distance=20.0, edge_id=0)
    graph.add_edge(1, 2, distance=20.0, edge_id=1)
    position_time = np.linspace(0, 6, 301)
    linear = 20 + 19 * np.sin(position_time)
    position = np.column_stack(
        (np.minimum(linear, 20.0), np.clip(linear - 20.0, 0.0, None))
    )
    spike_times = []
    for center in (8.0, 32.0):
        rate = 40 * np.exp(-((linear - center) ** 2) / 20)
        spike_times.append(np.sort(position_time[rng.poisson(rate / 50) > 0]))
    environment = Environment(
        place_bin_size=2.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
    )
    model = NonLocalSortedSpikesDetector(environments=environment)
    return model, {
        "spike_times": spike_times,
        "position_time": position_time,
        "position": position,
        "time_edges": np.linspace(0, 6, N_BINS + 1),
    }


@pytest.mark.parametrize("case", ["no_local_state", "frozen_rows", "track_graph"])
def test_checkpointed_em_other_models(case):
    from non_local_detector import ContFragSortedSpikesClassifier

    def build():
        if case == "track_graph":
            return _track_recording()
        model, data = _recording("sorted")
        if case == "no_local_state":
            model = ContFragSortedSpikesClassifier(
                environments=model.environments, infer_track_interior=False
            )
        else:
            model = NonLocalSortedSpikesDetector(
                environments=model.environments,
                infer_track_interior=False,
                frozen_discrete_transition_rows=[1],
            )
        return model, data

    representation = "auto" if case == "track_graph" else "structured"
    dense = _fit(*build())
    structured = _fit(*build(), transition_representation=representation, chunk_size=45)
    _assert_same_fit(structured, dense)


@pytest.mark.parametrize(
    "options",
    [
        {"chunk_size": 50},
        {"n_chunks": 4, "transition_representation": "structured"},
    ],
)
def test_rejected_chunk_options_leave_the_fitted_model_unchanged(options):
    """Invalid chunking is rejected before the internal fit replaces state."""
    model, data = _recording("sorted")
    model.estimate_parameters(**data, max_iter=1, store_log_likelihood=True)
    transitions = np.array(model.discrete_state_transitions_)
    encoding = model.encoding_model_
    log_likelihood = model.log_likelihood_
    with pytest.raises(ValueError, match="chunk_size"):
        model.estimate_parameters(**data, max_iter=1, **options)
    np.testing.assert_array_equal(model.discrete_state_transitions_, transitions)
    assert model.encoding_model_ is encoding
    assert model.log_likelihood_ is log_likelihood
