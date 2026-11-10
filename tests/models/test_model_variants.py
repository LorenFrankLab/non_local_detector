"""Tests for model variant classes (MultiEnvironment, ContFrag).

These are thin wrappers around base detector classes; their defaults are
pinned in test_constructor_contract.py and fitting/predicting is tested via
the base classes. These tests cover the variants' own behavior.
"""

import numpy as np
import pytest
import xarray as xr

from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.models.multienvironment_model import (
    MultiEnvironmentSortedSpikesClassifier,
)
from tests.conftest import make_state_bins_results


def _make_cont_frag_results(
    *,
    n_time: int,
    state_names: tuple[str, ...],
    position_grid: dict[str, np.ndarray],
    seed: int = 0,
) -> xr.Dataset:
    """Build a synthetic ContFrag results dataset over a full position grid.

    Unlike the paired-column ``make_state_bins_results`` factory, this takes a
    per-axis ``position_grid`` and forms the full Cartesian product (one bin per
    grid cell), matching the layout ``convert_results_to_xarray`` produces.

    Parameters
    ----------
    n_time : int
        Number of time steps.
    state_names : tuple of str
        Discrete state labels, one per state.
    position_grid : dict[str, np.ndarray]
        Mapping from position-dim name (e.g., ``"position"`` for 1D or
        ``"x_position"``/``"y_position"`` for 2D) to that axis's coordinate
        values. The full meshgrid of these axes becomes the position bins.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with ``acausal_posterior`` (time, state_bins).
    """
    mesh = np.meshgrid(*position_grid.values(), indexing="ij")
    position_columns = {
        name: grid.ravel() for name, grid in zip(position_grid, mesh, strict=False)
    }
    return make_state_bins_results(
        state_names, position_columns, n_time=n_time, seed=seed
    )


@pytest.mark.unit
def test_custom_initial_conditions_override():
    """Custom initial conditions override the variant's defaults."""
    ic = np.array([0.5, 0.5])
    model = MultiEnvironmentSortedSpikesClassifier(
        discrete_initial_conditions=ic,
    )
    np.testing.assert_array_equal(model.discrete_initial_conditions, ic)


@pytest.mark.unit
class TestContFragGetPosterior:
    """``get_posterior`` must marginalize all position dims across env shapes."""

    @pytest.mark.parametrize(
        "position_grid",
        [
            pytest.param({"position": np.linspace(0.0, 100.0, 20)}, id="1d"),
            pytest.param(
                {
                    "x_position": np.linspace(0.0, 50.0, 10),
                    "y_position": np.linspace(0.0, 50.0, 10),
                },
                id="2d",
            ),
            pytest.param(
                {f"dim{i}_position": np.linspace(0.0, 10.0, 4) for i in range(3)},
                id="nd",
            ),
        ],
    )
    def test_cont_frag_get_posterior_collapses_all_position_dims(self, position_grid):
        """``get_posterior`` marginalizes every position dim regardless of name.

        Covers 1D (``position``), 2D (``x_position``/``y_position``), and the
        >6D fallback (``dim{i}_position``), returning ``(time, state)`` with the
        state dim named ``state`` so the documented ``sel(state=...)`` keeps
        working.
        """
        n_time = 7
        state_names = ("Continuous", "Fragmented")
        results = _make_cont_frag_results(
            n_time=n_time, state_names=state_names, position_grid=position_grid
        )

        for cls in (ContFragSortedSpikesClassifier, ContFragClusterlessClassifier):
            state_probs = cls.get_posterior(results)
            assert state_probs.shape == (n_time, len(state_names)), (
                f"{cls.__name__}.get_posterior returned shape {state_probs.shape}, "
                f"expected (n_time, n_states) = ({n_time}, {len(state_names)})"
            )
            assert "state" in state_probs.dims
            # Every position dim must be collapsed (none left in the output).
            assert not any(
                d == "position" or d.endswith("_position") for d in state_probs.dims
            )
            # Each time row sums to ~1 (probability marginalized over space).
            np.testing.assert_allclose(state_probs.sum("state").values, 1.0, atol=1e-10)
            # The documented selection API keeps working.
            assert state_probs.sel(state="Continuous").shape == (n_time,)
