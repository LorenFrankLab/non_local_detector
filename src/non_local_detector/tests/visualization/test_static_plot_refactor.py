"""Refactor regression tests for ``visualization/static.py``.

Phase 1a refactored two pieces of inline algorithm in
``plot_non_local_model``:

1. The neuron-sort line that hardcoded
   ``detector.encoding_model_[("", 0)]["place_fields"]``.
2. The ``conditional_non_local_acausal_posterior`` block at lines
   155–169.

Both now route through helpers in ``non_local_detector.analysis``.
These tests paste the original inline algorithms verbatim and assert
the helper output is bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.analysis.place_fields import extract_per_cell_place_fields
from non_local_detector.analysis.posterior import conditional_non_local_posterior
from non_local_detector.tests._simulated_detectors import FittedDetector


@pytest.mark.unit
def test_conditional_non_local_posterior_matches_inline(
    nl_fitted: FittedDetector,
) -> None:
    """Helper output bit-identical to the pre-refactor inline algorithm.

    Pasted from the pre-refactor ``static.py`` lines 154–169.
    """
    detector = nl_fitted.detector
    results = nl_fitted.results
    env = detector.environments[0]
    state_ind = detector.state_ind_
    acausal_posterior = results.acausal_posterior.values
    results_time = results.time.values

    non_local_inds = np.nonzero(
        ["Non-Local" in state for state in detector.state_names]
    )[0]
    inline = np.zeros((len(results_time), len(env.place_bin_centers_)))
    for non_local_ind in non_local_inds:
        inline += acausal_posterior[:, state_ind == non_local_ind]
    inline /= np.nansum(inline, axis=1)[:, np.newaxis]
    inline[:, ~env.is_track_interior_] = np.nan

    helper = conditional_non_local_posterior(results, detector).values
    np.testing.assert_allclose(helper, inline, atol=1e-14, equal_nan=True)


@pytest.mark.unit
def test_place_field_peak_sort_matches_inline(
    nl_fitted: FittedDetector,
) -> None:
    """``extract_per_cell_place_fields`` driving the sort matches the inline pattern."""
    detector = nl_fitted.detector
    env = detector.environments[0]

    inline_place_fields = detector.encoding_model_[("", 0)]["place_fields"]
    inline_sort = np.argsort(
        env.place_bin_centers_[np.nanargmax(inline_place_fields, axis=1)].squeeze()
    )

    helper_place_fields = extract_per_cell_place_fields(detector)
    helper_sort = np.argsort(
        env.place_bin_centers_[np.nanargmax(helper_place_fields, axis=1)].squeeze()
    )

    np.testing.assert_array_equal(helper_sort, inline_sort)
