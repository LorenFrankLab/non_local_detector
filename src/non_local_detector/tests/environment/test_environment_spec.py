"""Contract tests for the ``Environment`` (spec) / ``FittedEnvironment`` split.

``Environment`` holds only the user-provided specification; calling
``fit_place_grid(position)`` returns a separate ``FittedEnvironment`` carrying
the resolved spatial grid and the spatial-query methods. The spec is never
mutated, and there is no ``_is_fitted`` flag — "fitted" is a type, not a state.
"""

import copy
import dataclasses

import numpy as np
import pytest

from non_local_detector.environment import Environment, FittedEnvironment

FITTED_ATTRS = (
    "edges_",
    "place_bin_edges_",
    "place_bin_centers_",
    "centers_shape_",
    "is_track_interior_",
    "is_track_boundary_",
    "track_graphDD",
    "distance_between_nodes_",
)


@pytest.fixture
def position_2d():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 10.0, size=(200, 2))


@pytest.mark.unit
class TestEnvironmentSpec:
    def test_spec_constructs_with_valid_inputs(self):
        spec = Environment(environment_name="env", place_bin_size=2.0)
        assert spec.environment_name == "env"
        assert spec.place_bin_size == 2.0

    def test_spec_does_not_have_fitted_attributes(self):
        """The spec exposes none of the fitted grid attributes nor _is_fitted."""
        spec = Environment(place_bin_size=2.0)
        assert not hasattr(spec, "_is_fitted")
        for attr in FITTED_ATTRS:
            assert not hasattr(spec, attr), f"spec should not have {attr}"

    def test_spec_is_hashable(self):
        """Specs are hashable (identity-based), so they work as dict keys/sets."""
        spec = Environment(place_bin_size=2.0)
        assert hash(spec) == hash(spec)
        assert spec in {spec}


@pytest.mark.unit
class TestFitPlaceGrid:
    def test_returns_fitted_environment(self, position_2d):
        fitted = Environment(place_bin_size=2.0).fit_place_grid(position_2d)
        assert isinstance(fitted, FittedEnvironment)
        # A 2D grid environment populates every grid attribute.
        for attr in FITTED_ATTRS:
            assert getattr(fitted, attr) is not None, f"{attr} should be set"
        assert fitted.place_bin_centers_.shape[1] == 2

    def test_does_not_mutate_spec(self, position_2d):
        """fit_place_grid must not mutate the spec it was called on."""
        spec = Environment(place_bin_size=2.0)
        spec_before = copy.deepcopy(spec)
        spec.fit_place_grid(position_2d)
        # Declared input fields are unchanged (catches e.g. an infer_track_interior
        # write-back).
        assert spec == spec_before
        # No fitted attribute leaked onto the spec as a new instance attribute
        # (dataclass __eq__ above only compares declared fields, so check each).
        assert not hasattr(spec, "_is_fitted")
        for attr in FITTED_ATTRS:
            assert not hasattr(spec, attr), f"spec leaked fitted attr {attr}"

    def test_fitted_forwards_spec_parameters(self, position_2d):
        """Spec parameters are readable on the fitted object (read-only)."""
        spec = Environment(environment_name="track", place_bin_size=2.0)
        fitted = spec.fit_place_grid(position_2d)
        assert fitted.environment_name == "track"
        assert fitted.place_bin_size == 2.0
        assert fitted.track_graph is None
        assert fitted.spec is spec

    def test_fitted_spec_parameters_are_read_only(self, position_2d):
        """Forwarded spec parameters cannot be assigned on the fitted object."""
        fitted = Environment(place_bin_size=2.0).fit_place_grid(position_2d)
        with pytest.raises(AttributeError):
            fitted.place_bin_size = 5.0

    def test_repeated_fit_from_spec_is_independent(self, position_2d):
        """Each fit produces an independent FittedEnvironment."""
        spec = Environment(place_bin_size=2.0)
        fitted_a = spec.fit_place_grid(position_2d)
        fitted_b = spec.fit_place_grid(position_2d)
        assert fitted_a is not fitted_b
        np.testing.assert_array_equal(
            fitted_a.place_bin_centers_, fitted_b.place_bin_centers_
        )

    def test_replace_overrides_fitted_field_without_refit(self, position_2d):
        """dataclasses.replace yields a new fitted env with overridden fields."""
        fitted = Environment(place_bin_size=2.0).fit_place_grid(position_2d)
        degenerate = dataclasses.replace(fitted, track_graphDD=None)
        assert degenerate.track_graphDD is None
        # Original is untouched.
        assert fitted.track_graphDD is not None
