"""Checkpointed EM over structured transitions reproduces dense EM."""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.exceptions import ValidationError

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
    kwargs.setdefault("max_iter", 3)
    results = model.estimate_parameters(**data, tolerance=1e-12, **kwargs)
    # Structured EM, or auto with a checkpoint option, runs on the operator.
    checkpoint_options = {"chunk_size", "result_path", "replay_cache_bytes"}
    if kwargs.get("transition_representation") == "structured" or (
        "transition_representation" in kwargs and checkpoint_options & set(kwargs)
    ):
        assert getattr(model, "_continuous_transition_operator_", None) is not None
    return results, {
        "discrete_state_transitions_": np.array(model.discrete_state_transitions_),
        "initial_conditions_": np.array(model.initial_conditions_),
        "n_iter_": model.n_iter_,
    }


# float32 EM parity for posteriors and fitted parameters. Measured differences
# after three iterations were at most 1.2e-7 for posteriors (about 8x below
# atol) and 5e-10 for transitions (2,000x below it), leaving margin for other
# devices, reduction orders and JAX versions. The operator and driver tests check the per-step
# products and statistics more tightly.
def _assert_same_fit(structured, dense, rtol=1e-5, atol=1e-6):
    (results, fitted), (reference, reference_fitted) = structured, dense
    assert fitted["n_iter_"] == reference_fitted["n_iter_"]
    for name in ("discrete_state_transitions_", "initial_conditions_"):
        np.testing.assert_allclose(
            fitted[name], reference_fitted[name], rtol=rtol, atol=atol
        )
    # Marginal log likelihoods are checked separately: at magnitudes of 1e3 to
    # 1e6, rtol=1e-5 would allow 0.01 to 10 units and hide evidence that drifts
    # with recording length. Each row's float32 increment is accurate to about
    # 1e-7 of its size, so the bound grows with the rows (8x the measured 4e-5
    # here; a float32 running total drifted by 3.4e-4 on these fits).
    np.testing.assert_allclose(
        results.attrs["marginal_log_likelihoods"],
        reference.attrs["marginal_log_likelihoods"],
        rtol=0,
        atol=1e-6 * N_BINS,
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


@pytest.mark.parametrize("variant", ["plain", "missing_and_store"])
def test_checkpointed_em_with_covariate_transitions(variant, tmp_path):
    covariates = {"speed": np.sin(np.arange(N_BINS) / 7)}
    extra = {}
    structured_extra = {}
    if variant == "missing_and_store":
        is_missing = np.zeros(N_BINS, dtype=bool)
        is_missing[40:70] = True
        extra["is_missing"] = is_missing
        structured_extra["result_path"] = tmp_path / "covariate"

    def run(**kwargs):
        model, data = _recording("sorted")
        model.discrete_transition_type = DiscreteNonStationaryDiagonal(
            np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + speed"
        )
        return _fit(
            model,
            data,
            discrete_transition_covariate_data=covariates,
            **extra,
            **kwargs,
        )

    _assert_same_fit(
        run(transition_representation="structured", chunk_size=50, **structured_extra),
        run(),
    )


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


def test_structured_em_rejects_row_chunks():
    model, data = _recording("sorted")
    with pytest.raises(ValidationError, match="n_chunks"):
        model.estimate_parameters(
            **data, max_iter=1, n_chunks=4, transition_representation="structured"
        )


def test_auto_em_with_row_chunks_runs_dense_em():
    """n_chunks is a dense-EM option, so "auto" honors it with dense EM."""
    model, data = _recording("sorted")
    dense = _fit(model, data, n_chunks=4, transition_representation="dense")
    model, data = _recording("sorted")
    auto = _fit(model, data, n_chunks=4, transition_representation="auto")
    assert not _uses_operator(model)
    _assert_same_fit(auto, dense)
    model, data = _recording("sorted")
    with pytest.raises(ValidationError, match="n_chunks .* cannot be combined"):
        model.estimate_parameters(
            **data,
            max_iter=1,
            n_chunks=4,
            chunk_size=50,
            transition_representation="auto",
        )


def test_dense_em_rejects_checkpoint_chunks():
    """chunk_size must not silently run unbounded dense EM."""
    model, data = _recording("sorted")
    with pytest.raises(ValidationError, match="structured"):
        model.estimate_parameters(
            **data, max_iter=1, chunk_size=50, transition_representation="dense"
        )


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
        {"chunk_size": 50, "transition_representation": "dense"},
        {"n_chunks": 4, "transition_representation": "structured"},
        {"chunk_size": 0, "transition_representation": "structured"},
        {"replay_cache_bytes": -1, "transition_representation": "structured"},
    ],
)
def test_rejected_chunk_options_leave_the_fitted_model_unchanged(options):
    """Invalid chunking is rejected before the internal fit replaces state."""
    model, data = _recording("sorted")
    model.estimate_parameters(**data, max_iter=1, store_log_likelihood=True)
    transitions = np.array(model.discrete_state_transitions_)
    encoding = model.encoding_model_
    log_likelihood = model.log_likelihood_
    with pytest.raises(ValidationError, match="chunk_size|replay_cache_bytes|n_chunks"):
        model.estimate_parameters(**data, max_iter=1, **options)
    np.testing.assert_array_equal(model.discrete_state_transitions_, transitions)
    assert model.encoding_model_ is encoding
    assert model.log_likelihood_ is log_likelihood


def _assert_store_keeps_em_traces(in_memory, stored, loaded, n_iter):
    """A result store reloads EM's traces and options as the fit returned
    them, and as the same fit held in memory computed them."""
    shapes = {
        "marginal_log_likelihoods": (n_iter + 1,),
        "em_objectives": (n_iter + 1,),
        "em_previous_objectives": (n_iter,),
    }
    for name, shape in shapes.items():
        assert isinstance(loaded.attrs[name], np.ndarray), name
        assert loaded.attrs[name].shape == shape, name
        # The store's JSON metadata round-trips floats exactly.
        np.testing.assert_array_equal(loaded.attrs[name], stored.attrs[name])
        # The final E-step writing a store is a different compiled program.
        np.testing.assert_allclose(
            loaded.attrs[name], in_memory.attrs[name], rtol=1e-6, atol=0.0
        )
    for name in ("em_transition_representation", "em_chunk_size"):
        assert loaded.attrs[name] == stored.attrs[name] == in_memory.attrs[name]
    # The default replay cache follows the memory available to each fit.
    replay = "em_replay_cache_bytes"
    assert loaded.attrs[replay] == stored.attrs[replay]


def test_checkpointed_em_writes_the_final_posterior_to_disk(tmp_path):
    """With result_path the final E-step streams its outputs to a store, which
    reloads with EM's traces."""
    model, data = _recording("sorted")
    in_memory = _fit(
        model,
        data,
        return_outputs="filter",
        transition_representation="structured",
        chunk_size=64,
    )[0]
    model, data = _recording("sorted")
    stored = _fit(
        model,
        data,
        return_outputs="filter",
        transition_representation="structured",
        chunk_size=64,
        result_path=tmp_path / "em",
    )[0]
    assert (tmp_path / "em" / "manifest.json").exists()
    # Lazily backed by the store, not held in memory.
    assert stored["acausal_posterior"].variable._in_memory is False
    for name in in_memory.data_vars:
        np.testing.assert_allclose(
            stored[name].values, in_memory[name].values, rtol=1e-6, atol=1e-7
        )
    np.testing.assert_array_equal(
        stored.attrs["marginal_log_likelihoods"],
        in_memory.attrs["marginal_log_likelihoods"],
    )
    loaded = model.load_results(tmp_path / "em")
    np.testing.assert_allclose(
        loaded["acausal_posterior"].values,
        in_memory["acausal_posterior"].values,
        rtol=1e-6,
        atol=1e-7,
    )
    _assert_store_keeps_em_traces(in_memory, stored, loaded, model.n_iter_)


@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_em_result_store_without_iterations_keeps_the_em_traces(tmp_path):
    """With max_iter=0 no M-step runs, and the store still reloads the final
    E-step's trace."""
    kwargs = {
        "max_iter": 0,
        "transition_representation": "structured",
        "chunk_size": 64,
    }
    model, data = _recording("sorted")
    in_memory = _fit(model, data, **kwargs)[0]
    model, data = _recording("sorted")
    stored = _fit(model, data, result_path=tmp_path / "em", **kwargs)[0]
    loaded = model.load_results(tmp_path / "em")
    _assert_store_keeps_em_traces(in_memory, stored, loaded, n_iter=0)


def test_result_path_rejects_dense_em_and_a_stored_likelihood(tmp_path):
    model, data = _recording("sorted")
    with pytest.raises(ValidationError, match="structured"):
        model.estimate_parameters(
            **data,
            max_iter=1,
            result_path=tmp_path / "a",
            transition_representation="dense",
        )
    with pytest.raises(ValidationError, match="store_log_likelihood"):
        model.estimate_parameters(
            **data,
            max_iter=1,
            transition_representation="structured",
            result_path=tmp_path / "b",
            store_log_likelihood=True,
        )
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_checkpointed_em_replay_cache_does_not_change_results(monkeypatch, family):
    """The replay reuses forward-pass rows; results are bitwise equal."""
    from non_local_detector import checkpointed_inference

    driver = checkpointed_inference.checkpointed_forward_backward

    def run(**kwargs):
        calls = []

        def counting_driver(time_edges, initial, likelihood, **driver_kwargs):
            def counted(*args, **kw):
                calls.append(kw["row_slice"].start)
                return likelihood(*args, **kw)

            return driver(time_edges, initial, counted, **driver_kwargs)

        monkeypatch.setattr(
            checkpointed_inference, "checkpointed_forward_backward", counting_driver
        )
        model, data = _recording(family)
        results = _fit(
            model,
            data,
            transition_representation="structured",
            chunk_size=50,
            **kwargs,
        )
        return results, len(calls)

    (uncached, _), uncached_calls = run(replay_cache_bytes=0)
    (cached, _), cached_calls = run()
    # Six chunks per pass: three in-loop E-steps and the final one.
    assert uncached_calls == 4 * 12
    assert cached_calls == 4 * 6
    for name in cached.data_vars:
        np.testing.assert_array_equal(cached[name].values, uncached[name].values)
    np.testing.assert_array_equal(
        cached.attrs["marginal_log_likelihoods"],
        uncached.attrs["marginal_log_likelihoods"],
    )


def test_dense_em_rejects_a_replay_cache():
    model, data = _recording("sorted")
    with pytest.raises(ValidationError, match="replay_cache_bytes"):
        model.estimate_parameters(
            **data,
            max_iter=1,
            replay_cache_bytes=2**20,
            transition_representation="dense",
        )


def _uses_operator(model):
    return getattr(model, "_continuous_transition_operator_", None) is not None


def test_auto_em_uses_dense_em_for_a_small_state_space():
    """Dense EM is faster where its transition matrix is small."""
    model, data = _recording("sorted")
    dense = _fit(model, data)
    model, data = _recording("sorted")
    auto = _fit(model, data, transition_representation="auto")
    assert not _uses_operator(model)
    assert model._transition_representation_ == "dense"
    for name in dense[0].data_vars:
        np.testing.assert_array_equal(auto[0][name].values, dense[0][name].values)
    np.testing.assert_array_equal(
        auto[1]["discrete_state_transitions_"], dense[1]["discrete_state_transitions_"]
    )


@pytest.mark.parametrize(
    "reason", ["large_transition", "dense_does_not_fit", "checkpoint_option"]
)
def test_auto_em_uses_structured_em_otherwise(reason, monkeypatch, tmp_path):
    from non_local_detector.models import base

    kwargs = {}
    if reason == "large_transition":
        monkeypatch.setattr(base, "_AUTO_DENSE_TRANSITION_BYTES", 1)
    elif reason == "dense_does_not_fit":
        monkeypatch.setattr(base, "_device_memory_limit", lambda: 1)
    else:
        kwargs["result_path"] = tmp_path / "store"
    model, data = _recording("sorted")
    dense = _fit(model, data)
    model, data = _recording("sorted")
    auto = _fit(model, data, transition_representation="auto", **kwargs)
    assert _uses_operator(model)
    _assert_same_fit(auto, dense)


def test_auto_em_keeps_structured_em_when_the_full_dense_matrix_is_over_budget():
    """Few interior bins on a large grid: the full dense transition is too big.

    Dense EM stores the transition over every state bin, so a small interior
    does not make it affordable; auto must keep the operator instead of
    exceeding max_dense_transition_bytes.
    """
    rng = np.random.default_rng(2)
    position_time = np.linspace(0, 6, 301)
    position = np.column_stack(
        (100 + 5 * np.sin(position_time), 100 + 5 * np.cos(position_time))
    )
    spike_times = [np.sort(position_time[rng.uniform(size=301) < 0.2])]
    model = NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=2, position_range=((0, 200), (0, 200)))
    )
    results = model.estimate_parameters(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        time_edges=np.linspace(0, 6, N_BINS + 1),
        max_iter=1,
        transition_representation="auto",
    )
    n_bins = model.state_ind_.shape[0]
    assert n_bins**2 * 8 > model._max_dense_transition_bytes_
    assert model.is_track_interior_state_bins_.sum() ** 2 * 4 < 2**20
    assert _uses_operator(model)
    assert np.all(np.isfinite(results["acausal_state_probabilities"].values))


@pytest.mark.parametrize("available", [8 * 2**30, 64 * 2**30, None])
def test_default_replay_cache_is_capped_by_available_memory(available, monkeypatch):
    """The default budget is min(16 GiB, half the available host memory)."""
    from non_local_detector import checkpointed_inference

    monkeypatch.setattr(
        checkpointed_inference, "_available_host_bytes", lambda: available
    )
    budgets = []
    driver = checkpointed_inference.checkpointed_forward_backward

    def recording_driver(*args, **kwargs):
        budgets.append(kwargs["replay_cache_bytes"])
        return driver(*args, **kwargs)

    monkeypatch.setattr(
        checkpointed_inference, "checkpointed_forward_backward", recording_driver
    )
    model, data = _recording("sorted")
    model.estimate_parameters(
        **data, max_iter=1, transition_representation="structured"
    )
    expected = 4 * 2**30 if available == 8 * 2**30 else 16 * 2**30
    assert budgets and set(budgets) == {expected}


def test_auto_em_device_memory_boundary(monkeypatch):
    """Dense EM needs its arrays to fit in half the device memory."""
    from non_local_detector.models import base

    model, data = _recording("sorted")
    n_time = N_BINS
    model.fit(
        data["position_time"],
        data["position"],
        data["spike_times"],
        transition_representation="auto",
    )
    n_bins = int(model.is_track_interior_state_bins_.sum())
    working = base._DENSE_EM_WORKING_ARRAYS * n_time * n_bins * 4
    for limit, dense in ((2 * working, True), (2 * working - 2, False)):
        model.fit(
            data["position_time"],
            data["position"],
            data["spike_times"],
            transition_representation="auto",
        )
        monkeypatch.setattr(base, "_device_memory_limit", lambda limit=limit: limit)
        model._choose_em_representation(n_time)
        assert _uses_operator(model) is not dense


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"transition_representation": "dense"},
            {"em_transition_representation": "dense", "em_n_chunks": 1},
        ),
        (
            {"transition_representation": "dense", "n_chunks": 3},
            {"em_transition_representation": "dense", "em_n_chunks": 3},
        ),
        (
            {"transition_representation": "auto"},
            {"em_transition_representation": "dense", "em_n_chunks": 1},
        ),
        (
            {"transition_representation": "structured"},
            {
                "em_transition_representation": "structured",
                "em_chunk_size": None,
                "em_replay_cache_bytes": None,
            },
        ),
        (
            {
                "transition_representation": "structured",
                "chunk_size": 40,
                "replay_cache_bytes": 0,
            },
            {
                "em_transition_representation": "structured",
                "em_chunk_size": 40,
                "em_replay_cache_bytes": 0,
            },
        ),
    ],
)
def test_results_record_the_em_path(kwargs, expected, tmp_path):
    """Users can tell from the results which EM path ran and with what chunks."""
    model, data = _recording("sorted")
    results = model.estimate_parameters(**data, max_iter=1, **kwargs)
    for name, value in expected.items():
        # None: a resolved default, recorded as a positive integer.
        if value is None:
            assert results.attrs[name] > 0
        else:
            assert results.attrs[name] == value
    other = {"em_n_chunks", "em_chunk_size", "em_replay_cache_bytes"} - set(expected)
    assert not other & set(results.attrs)
    if kwargs["transition_representation"] == "structured":
        model, data = _recording("sorted")
        stored = model.estimate_parameters(
            **data, max_iter=1, result_path=tmp_path / "store", **kwargs
        )
        # A default replay cache depends on the memory available at the call.
        for name, value in expected.items():
            if value is None:
                assert stored.attrs[name] > 0
            else:
                assert stored.attrs[name] == value


@pytest.mark.parametrize("n_chunks", [1, 3])
def test_dense_em_warns_before_exceeding_device_memory(n_chunks, monkeypatch):
    """A dense EM that cannot fit says so before it runs, and names the fix."""
    import warnings

    from non_local_detector.models import base

    monkeypatch.setattr(base, "_device_memory_limit", lambda: 1)
    model, data = _recording("sorted")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.estimate_parameters(
            **data, max_iter=1, n_chunks=n_chunks, transition_representation="dense"
        )
    messages = [str(w.message) for w in caught if "Dense EM needs" in str(w.message)]
    if n_chunks == 1:
        assert len(messages) == 1
        assert 'transition_representation="auto"' in messages[0]
    else:
        assert not messages


def test_dense_em_within_memory_does_not_warn(monkeypatch):
    import warnings

    from non_local_detector.models import base

    monkeypatch.setattr(base, "_device_memory_limit", lambda: 2**50)
    model, data = _recording("sorted")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.estimate_parameters(**data, max_iter=1, transition_representation="dense")
    assert not [w for w in caught if "Dense EM needs" in str(w.message)]


def _with_opaque_random_walks(model):
    """Swap RandomWalk blocks for a subclass the operator builder cannot read."""
    from dataclasses import dataclass, fields

    from non_local_detector.continuous_state_transitions import RandomWalk

    @dataclass
    class OpaqueRandomWalk(RandomWalk):
        pass

    model.continuous_transition_types = [
        [
            OpaqueRandomWalk(
                **{f.name: getattr(block, f.name) for f in fields(block) if f.init}
            )
            if type(block) is RandomWalk
            else block
            for block in row
        ]
        for row in model.continuous_transition_types
    ]
    return model


def test_auto_em_falls_back_to_dense_for_transitions_without_an_operator():
    """A transition "auto" cannot build in bounded form runs dense EM, as the
    dense constructor accepts it; asking for structured EM still fails."""
    from non_local_detector.transition_operators import UnsupportedTransitionError

    model, data = _recording("sorted")
    results = _with_opaque_random_walks(model).estimate_parameters(
        **data, max_iter=1, transition_representation="auto"
    )
    assert results.attrs["em_transition_representation"] == "dense"
    assert not _uses_operator(model)
    model, data = _recording("sorted")
    with pytest.raises(UnsupportedTransitionError):
        _with_opaque_random_walks(model).estimate_parameters(
            **data, max_iter=1, transition_representation="auto", chunk_size=50
        )
