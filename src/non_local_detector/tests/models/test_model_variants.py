"""Tests for model variant classes (MultiEnvironment, NoSpikeContFrag).

These are thin wrappers around base detector classes. Tests focus on
correct default parameter initialization since the fitting/predicting
logic is tested via the base class tests.
"""

import numpy as np
import pytest

from non_local_detector.models.multienvironment_model import (
    MultiEnvironmentClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
)
from non_local_detector.models.nospike_cont_frag_model import (
    NoSpikeContFragClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
)


@pytest.mark.unit
class TestMultiEnvironmentDefaults:
    """Test default parameter construction for multi-environment models."""

    def test_sorted_spikes_default_state_names(self):
        """Should have 2 environment states by default."""
        model = MultiEnvironmentSortedSpikesClassifier()
        assert len(model.state_names) == 2

    def test_sorted_spikes_initial_conditions_valid(self):
        """Initial conditions must be a valid probability distribution."""
        model = MultiEnvironmentSortedSpikesClassifier()
        ic = model.discrete_initial_conditions
        assert np.all(ic >= 0)
        assert np.isclose(ic.sum(), 1.0)

    def test_sorted_spikes_observation_models_have_environment_names(self):
        """Each observation model should reference an environment."""
        model = MultiEnvironmentSortedSpikesClassifier()
        env_names = {obs.environment_name for obs in model.observation_models}
        assert env_names == {"env1", "env2"}

    def test_clusterless_default_construction(self):
        """Clusterless variant should construct with correct defaults."""
        model = MultiEnvironmentClusterlessClassifier()
        assert len(model.state_names) == 2
        assert np.isclose(model.discrete_initial_conditions.sum(), 1.0)

    def test_custom_initial_conditions_override(self):
        """Custom initial conditions should override defaults."""
        ic = np.array([0.5, 0.5])
        model = MultiEnvironmentSortedSpikesClassifier(
            discrete_initial_conditions=ic,
        )
        np.testing.assert_array_equal(model.discrete_initial_conditions, ic)


@pytest.mark.unit
class TestNoSpikeContFragDefaults:
    """Test default parameter construction for no-spike cont-frag models."""

    def test_sorted_spikes_three_states(self):
        """Should have 3 states: No-Spike, Continuous, Fragmented."""
        model = NoSpikeContFragSortedSpikesClassifier()
        assert len(model.state_names) == 3

    def test_sorted_spikes_initial_conditions_valid(self):
        """Initial conditions must be a valid probability distribution."""
        model = NoSpikeContFragSortedSpikesClassifier()
        ic = model.discrete_initial_conditions
        assert np.all(ic >= 0)
        assert np.isclose(ic.sum(), 1.0)

    def test_sorted_spikes_has_no_spike_observation(self):
        """Should have at least one no-spike observation model."""
        model = NoSpikeContFragSortedSpikesClassifier()
        assert any(obs.is_no_spike for obs in model.observation_models)

    def test_clusterless_default_construction(self):
        """Clusterless variant should construct with correct defaults."""
        model = NoSpikeContFragClusterlessClassifier()
        assert len(model.state_names) == 3
        assert np.isclose(model.discrete_initial_conditions.sum(), 1.0)

    def test_clusterless_has_no_spike_observation(self):
        """Clusterless variant should also have a no-spike observation."""
        model = NoSpikeContFragClusterlessClassifier()
        assert any(obs.is_no_spike for obs in model.observation_models)
