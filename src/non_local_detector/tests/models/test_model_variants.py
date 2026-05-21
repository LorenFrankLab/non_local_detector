"""Tests for model variant classes (MultiEnvironment, NoSpikeContFrag).

These are thin wrappers around base detector classes. Tests focus on
correct default parameter initialization since the fitting/predicting
logic is tested via the base class tests.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.models.multienvironment_model import (
    MultiEnvironmentClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
)
from non_local_detector.models.nospike_cont_frag_model import (
    NoSpikeContFragClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
)


def _make_cont_frag_results(
    *,
    n_time: int,
    state_names: tuple[str, ...],
    position_grid: dict[str, np.ndarray],
    seed: int = 0,
) -> xr.Dataset:
    """Build a synthetic results dataset matching ``_convert_results_to_xarray``.

    The acausal_posterior has dim ``state_bins`` indexed by a MultiIndex over
    ``("state", *position_dim_names)`` — the same layout produced by the
    detector classes. Mass is normalized per time step.

    Parameters
    ----------
    n_time : int
        Number of time steps.
    state_names : tuple of str
        Discrete state labels, one per state.
    position_grid : dict[str, np.ndarray]
        Mapping from position-dim name (e.g., ``"position"`` for 1D or
        ``"x_position"``/``"y_position"`` for 2D) to coordinate values.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with ``acausal_posterior`` (time, state_bins).
    """
    rng = np.random.default_rng(seed)
    position_dim_names = list(position_grid.keys())
    grid_arrays = [position_grid[name] for name in position_dim_names]
    mesh = np.meshgrid(*grid_arrays, indexing="ij")
    flat_positions = [m.ravel() for m in mesh]
    n_pos_bins = flat_positions[0].size

    state_col: list[str] = []
    position_cols: list[list[float]] = [[] for _ in position_dim_names]
    for s in state_names:
        state_col.extend([s] * n_pos_bins)
        for i, fp in enumerate(flat_positions):
            position_cols[i].extend(fp.tolist())

    mindex = pd.MultiIndex.from_arrays(
        [np.asarray(state_col), *(np.asarray(c) for c in position_cols)],
        names=("state", *position_dim_names),
    )

    raw = rng.random((n_time, len(mindex)))
    posterior = raw / raw.sum(axis=1, keepdims=True)

    if hasattr(xr.Coordinates, "from_pandas_multiindex"):
        mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "state_bins")
        ds = xr.Dataset(
            data_vars={"acausal_posterior": (("time", "state_bins"), posterior)},
            coords={"time": np.arange(n_time), **mindex_coords},
        )
    else:  # pragma: no cover - older xarray fallback
        ds = xr.Dataset(
            data_vars={"acausal_posterior": (("time", "state_bins"), posterior)},
            coords={"time": np.arange(n_time), "state_bins": mindex},
        )
    return ds


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


@pytest.mark.unit
class TestContFragGetPosterior:
    """``get_posterior`` must marginalize all position dims across env shapes."""

    def test_cont_frag_get_posterior_2d(self):
        """A 2D environment uses ``x_position``/``y_position`` dim names; the
        static method must collapse both, returning shape (n_time, n_states)
        with the state dim named ``state``.
        """
        x_centers = np.linspace(0.0, 50.0, 10)
        y_centers = np.linspace(0.0, 50.0, 10)
        n_time = 7
        state_names = ("Continuous", "Fragmented")

        results = _make_cont_frag_results(
            n_time=n_time,
            state_names=state_names,
            position_grid={"x_position": x_centers, "y_position": y_centers},
        )

        for cls in (ContFragSortedSpikesClassifier, ContFragClusterlessClassifier):
            state_probs = cls.get_posterior(results)
            assert state_probs.shape == (n_time, len(state_names)), (
                f"{cls.__name__}.get_posterior returned shape {state_probs.shape}, "
                f"expected (n_time, n_states) = ({n_time}, {len(state_names)})"
            )
            assert "state" in state_probs.dims, (
                f"{cls.__name__}.get_posterior should expose a 'state' dim; got "
                f"dims {state_probs.dims}"
            )
            # Position dims must be fully collapsed.
            assert "x_position" not in state_probs.dims
            assert "y_position" not in state_probs.dims
            # Each time row must sum to ~1 (probabilities marginalized over space).
            np.testing.assert_allclose(state_probs.sum("state").values, 1.0, atol=1e-10)
            # Selection by state label works and returns a per-time series.
            continuous_prob = state_probs.sel(state="Continuous")
            assert continuous_prob.shape == (n_time,)

    def test_cont_frag_get_posterior_1d_dim_name_unchanged(self):
        """The 1D environment uses dim name ``position``; the static method
        must still produce a state-named output dim so ``sel(state=...)`` keeps
        working as documented.
        """
        position_centers = np.linspace(0.0, 100.0, 20)
        n_time = 6
        state_names = ("Continuous", "Fragmented")

        results = _make_cont_frag_results(
            n_time=n_time,
            state_names=state_names,
            position_grid={"position": position_centers},
        )

        for cls in (ContFragSortedSpikesClassifier, ContFragClusterlessClassifier):
            state_probs = cls.get_posterior(results)
            assert state_probs.shape == (n_time, len(state_names))
            assert "state" in state_probs.dims
            assert "position" not in state_probs.dims
            np.testing.assert_allclose(state_probs.sum("state").values, 1.0, atol=1e-10)
            # The docstring example must keep working.
            assert state_probs.sel(state="Continuous").shape == (n_time,)
