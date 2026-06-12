"""Encoding models are shared by ``(environment_name, encoding_group)``.

Regression guard for the ``ObservationModel`` equality change. Equality now
compares all four fields (``environment_name``, ``encoding_group``,
``is_local``, ``is_no_spike``), so ``np.unique(observation_models)`` can return
several entries that share a single ``(environment_name, encoding_group)``
encoding key — the default non-local config is exactly such a case. The
encoding-fit loops deduplicate by that key, so the number of fitted encoding
models (and each model's value) must be unchanged: one model per distinct
``(environment_name, encoding_group)`` pair, never one per observation-model
entry. See PR #34.
"""

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


def _distinct_encoding_keys(observation_models):
    return {(o.environment_name, o.encoding_group) for o in observation_models}


@pytest.mark.unit
def test_default_nonlocal_obs_models_expand_under_unique():
    """The default non-local config triggers the duplicate-key scenario.

    This documents *why* the dedup matters: ``np.unique`` returns more entries
    than there are distinct encoding keys, because the four-state default has
    ``is_local`` / ``is_no_spike``-distinct models plus a repeated plain
    ``ObservationModel()`` that all share the single ``("", 0)`` key.
    """
    detector = NonLocalClusterlessDetector()

    n_unique = len(np.unique(detector.observation_models))
    n_keys = len(_distinct_encoding_keys(detector.observation_models))

    # 3 unique observation models (is_local, is_no_spike, plain) collapse onto
    # a single encoding key; if this stops being true the dedup test below is
    # no longer exercising the duplicate path.
    assert n_unique > n_keys, (
        f"expected the default non-local config to have more unique observation "
        f"models ({n_unique}) than encoding keys ({n_keys})"
    )
    assert n_keys == 1


@pytest.mark.unit
def test_nonlocal_clusterless_fits_one_encoding_model_per_env_group():
    """Fitting a default ``NonLocalClusterlessDetector`` yields exactly one
    encoding model per distinct ``(environment_name, encoding_group)`` pair,
    not one per observation-model entry.

    ``fit`` builds ``encoding_model_`` without running EM, so this is fast.
    """
    sim = make_simulated_run_data(
        n_tetrodes=2,
        place_field_means=np.arange(0, 80, 20),
        n_runs=2,
        seed=0,
    )

    detector = NonLocalClusterlessDetector()
    detector.fit(
        position_time=sim.position_time,
        position=sim.position,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
    )

    expected_keys = _distinct_encoding_keys(detector.observation_models)
    assert len(detector.encoding_model_) == len(expected_keys)
    assert set(detector.encoding_model_) == expected_keys


@pytest.mark.unit
def test_distinct_encoding_groups_fit_separate_models():
    """Two distinct encoding groups produce two separate encoding models.

    This is the adversarial guard for the dedup *key*: the loops skip repeated
    ``(environment_name, encoding_group)`` pairs but must keep genuinely
    distinct keys. A dedup that collapsed on ``environment_name`` alone (or any
    key dropping ``encoding_group``) would silently drop the second group and
    leave a single encoding model — caught here. The count-only tests above do
    not catch this because the default config has a single encoding key.
    """
    sim = make_simulated_run_data(
        n_tetrodes=2,
        place_field_means=np.arange(0, 80, 20),
        n_runs=2,
        seed=0,
    )

    detector = ClusterlessDecoder()
    # Fit once to initialize the environment grid (and the default single-group
    # encoding model).
    detector.fit(
        position_time=sim.position_time,
        position=sim.position,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
    )

    # Reconfigure for two encoding groups on the same environment and split the
    # position timeline between them, then refit only the encoding models.
    detector.observation_models = [
        ObservationModel(encoding_group=0),
        ObservationModel(encoding_group=1),
    ]
    n = sim.position_time.shape[0]
    encoding_group_labels = np.zeros(n, dtype=int)
    encoding_group_labels[n // 2 :] = 1

    detector.fit_encoding_model(
        position_time=sim.position_time,
        position=sim.position,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
        encoding_group_labels=encoding_group_labels,
    )

    # Both distinct keys survive the dedup; neither is dropped or merged.
    assert set(detector.encoding_model_) == {("", 0), ("", 1)}
    assert detector.encoding_model_[("", 0)] is not None
    assert detector.encoding_model_[("", 1)] is not None


@pytest.mark.unit
def test_nonlocal_sorted_spikes_fits_one_encoding_model_per_env_group():
    """Same encoding-model-sharing invariant for the sorted-spikes path."""
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_freq,
        _is_event,
        _,
    ) = make_simulated_data(seed=0, n_neurons=4)

    detector = NonLocalSortedSpikesDetector(sorted_spikes_algorithm="sorted_spikes_kde")
    detector.fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
    )

    expected_keys = _distinct_encoding_keys(detector.observation_models)
    assert len(detector.encoding_model_) == len(expected_keys)
    assert set(detector.encoding_model_) == expected_keys
