"""Encoding EM weights follow local observation models independently of labels."""

import numpy as np
import pytest

from non_local_detector import Environment, SortedSpikesDecoder

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("n_local_states", [1, 2])
@pytest.mark.parametrize("representation", ["dense", "structured"])
@pytest.mark.parametrize(
    "state_names",
    [
        ["Observed", "Silent", "Moving", "Fragmented"],
        ["Observed", "Silent", "Local", "Fragmented"],
    ],
)
def test_encoding_update_uses_local_observation_posterior(
    checkpoint_recording,
    monkeypatch,
    family,
    state_names,
    n_local_states,
    representation,
):
    """Renaming Local or assigning its label to another state cannot change weights."""
    model, fit_kwargs, predict_kwargs = checkpoint_recording(family)
    model.state_names = state_names
    if n_local_states == 2:
        model.observation_models = [
            model.observation_models[0],
            model.observation_models[0],
            *model.observation_models[2:],
        ]
        model.discrete_initial_conditions = np.array([0.5, 0.5, 0, 0])
    posterior_rows = []
    encoding_weights = []
    original_predict = model._predict
    original_checkpointed_e_step = model._checkpointed_e_step
    original_fit_encoding = model.fit_encoding_model

    def record_prediction(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        posterior_rows.append(result[1])
        return result

    def record_checkpointed_e_step(*args, **kwargs):
        result = original_checkpointed_e_step(*args, **kwargs)
        posterior_rows.append(result.dataset.acausal_state_probabilities.values)
        return result

    def record_encoding(*args, **kwargs):
        if kwargs.get("weights") is not None:
            encoding_weights.append(np.array(kwargs["weights"]))
        return original_fit_encoding(*args, **kwargs)

    monkeypatch.setattr(model, "_predict", record_prediction)
    monkeypatch.setattr(model, "_checkpointed_e_step", record_checkpointed_e_step)
    monkeypatch.setattr(model, "fit_encoding_model", record_encoding)
    edges = predict_kwargs["time_edges"]
    model.estimate_parameters(
        **fit_kwargs,
        time_edges=edges,
        estimate_encoding_model=True,
        estimate_discrete_transition=False,
        estimate_initial_conditions=False,
        max_iter=1,
        min_encoding_local_mass=0,
        min_encoding_local_ess=0,
        transition_representation=representation,
        **({"chunk_size": 37} if representation == "structured" else {}),
    )

    assert len(encoding_weights) == 1
    local_indices = [
        i for i, obs in enumerate(model.observation_models) if obs.is_local
    ]
    expected = np.interp(
        fit_kwargs["position_time"],
        edges[:-1] + np.diff(edges) / 2,
        posterior_rows[0][:, local_indices].sum(axis=1),
    )
    np.testing.assert_allclose(encoding_weights[0], expected, rtol=1e-6, atol=1e-8)


def test_encoding_update_skips_models_without_local_observations(monkeypatch):
    """A continuous decoder named Local still has no local emission to weight."""
    time = np.linspace(0, 1, 31)
    model = SortedSpikesDecoder(
        state_names=["Local"],
        environments=Environment(position_range=[(0, 10)], place_bin_size=2),
        infer_track_interior=False,
    )
    weights = []
    original_fit_encoding = model.fit_encoding_model

    def record_encoding(*args, **kwargs):
        if kwargs.get("weights") is not None:
            weights.append(kwargs["weights"])
        return original_fit_encoding(*args, **kwargs)

    monkeypatch.setattr(model, "fit_encoding_model", record_encoding)
    model.estimate_parameters(
        position_time=time,
        position=np.linspace(1, 9, len(time)),
        spike_times=[np.array([0.1, 0.4, 0.8])],
        time_edges=np.linspace(0, 1, 51),
        estimate_encoding_model=True,
        estimate_discrete_transition=False,
        estimate_initial_conditions=False,
        max_iter=1,
    )

    assert weights == []
